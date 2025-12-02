"""
HOPE Model Training Script (Production-Ready)
Features:
- Streaming Dataset (FineWeb, RefinedWeb, Wiki, Books)
- 2-Phase Training (Stateless -> Stateful)
- Mixed Precision (AMP)
- Torch Compile
- Gradient Checkpointing
"""
import os
import time
import math
import torch
import tiktoken
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset, interleave_datasets

try:
    from .config import HOPEConfig
    from .model import HOPE
except ImportError:
    from config import HOPEConfig
    from model import HOPE

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Hyperparameters
BATCH_SIZE = 16
SEQ_LENGTH = 2048
MAX_STEPS = 150000
WARMUP_STEPS = 2000
BASE_LR = 1.5e-4
MIN_LR = 2e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Phase Switching
PHASE2_START_STEP = 50000
STATE_RESET_INTERVAL = 2000

# System
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
COMPILE = True

# Logging
LOG_INTERVAL = 100
SAVE_INTERVAL = 5000
OUT_DIR = 'checkpoints'

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Dataset Pipeline
# -----------------------------------------------------------------------------
class StreamingTextDataset(IterableDataset):
    def __init__(self, tokenizer, seq_length=2048):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Load datasets in streaming mode
        # Note: In a real run, ensure you have internet access and HF token if needed
        print("Loading datasets...")
        ds_fineweb = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
        ds_refined = load_dataset("tiiuae/falcon-refinedweb", split="train", streaming=True)
        ds_wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        ds_books = load_dataset("bookcorpus", split="train", streaming=True)
        
        # Interleave with probabilities (adjust as needed)
        self.dataset = interleave_datasets(
            [ds_fineweb, ds_refined, ds_wiki, ds_books],
            probabilities=[0.4, 0.3, 0.2, 0.1],
            seed=42
        )
        
    def __iter__(self):
        buffer = []
        for item in self.dataset:
            # Handle different column names
            text = item.get('text', item.get('content', ''))
            if not text:
                continue
                
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            
            while len(buffer) >= self.seq_length + 1:
                chunk = buffer[:self.seq_length + 1]
                buffer = buffer[self.seq_length:] # Sliding window overlap could be implemented here
                
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def get_lr(it):
    # 1) Linear warmup
    if it < WARMUP_STEPS:
        return BASE_LR * (it + 1) / WARMUP_STEPS
    # 2) Constant min learning rate after decay
    if it > MAX_STEPS:
        return MIN_LR
    # 3) Cosine decay
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (BASE_LR - MIN_LR)

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
def main():
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    
    # 1. Initialize Model
    config = HOPEConfig()
    config.block_size = SEQ_LENGTH
    
    print(f"Initializing HOPE model ({config.n_layer} layers, {config.n_embd} dim)...")
    model = HOPE(config)
    model.to(DEVICE)
    
    # Enable Gradient Checkpointing
    model.enable_gradient_checkpointing()
    print("Gradient checkpointing enabled.")
    
    # Compile
    if COMPILE and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
        
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=BASE_LR, 
        weight_decay=WEIGHT_DECAY,
        fused=True if torch.cuda.is_available() else False
    )
    
    # Mixed Precision
    scaler = GradScaler(enabled=(DTYPE == 'float16'))
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DTYPE]
    
    # Dataset
    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = StreamingTextDataset(tokenizer, seq_length=SEQ_LENGTH)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        pin_memory=True,
        num_workers=2 # Adjust based on CPU
    )
    
    # Training State
    iter_num = 0
    tokens_processed = 0
    t0 = time.time()
    
    # Persistent states for Phase 2
    persistent_states = None
    
    print("Starting training...")
    train_iter = iter(train_loader)
    
    while iter_num < MAX_STEPS:
        # Determine Phase
        is_stateful_phase = iter_num >= PHASE2_START_STEP
        
        # Fetch Batch
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
            
        X, Y = X.to(DEVICE, non_blocking=True), Y.to(DEVICE, non_blocking=True)
        
        # LR Schedule
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Forward Pass
        # Phase 1: Stateless (always pass None, ignore returned states)
        # Phase 2: Stateful (pass persistent_states, update them)
        
        current_states = persistent_states if is_stateful_phase else None
        
        # Reset states periodically in Phase 2
        if is_stateful_phase and (iter_num - PHASE2_START_STEP) % STATE_RESET_INTERVAL == 0:
            current_states = None
            
        with autocast(enabled=True, dtype=pt_dtype):
            logits, loss, new_states = model(X, Y, states=current_states)
            
        # Handle States
        if is_stateful_phase:
            # Detach states to truncate BPTT
            persistent_states = [s.detach() for s in new_states]
        else:
            persistent_states = None
            
        # Backward Pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Clip Gradient
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        # Step
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        tokens_processed += X.numel()
        if iter_num % LOG_INTERVAL == 0:
            dt = time.time() - t0
            t0 = time.time()
            phase_str = "STATEFUL" if is_stateful_phase else "STATELESS"
            print(f"step {iter_num} | loss {loss.item():.4f} | lr {lr:.2e} | {phase_str} | tokens {tokens_processed/1e9:.2f}B | {dt*1000/LOG_INTERVAL:.2f}ms/step")
            
        # Checkpointing
        if iter_num > 0 and iter_num % SAVE_INTERVAL == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'config': config,
                'persistent_states': persistent_states
            }
            torch.save(checkpoint, os.path.join(OUT_DIR, f"ckpt_{iter_num}.pt"))
            print(f"Saved checkpoint to {OUT_DIR}/ckpt_{iter_num}.pt")
            
        iter_num += 1

if __name__ == "__main__":
    main()
