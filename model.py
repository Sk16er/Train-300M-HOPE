"""
HOPE Model Implementation (Production-Ready)
TitansL2 + CMS Memory | Gradient Checkpointing | Stateful/Stateless
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List, Dict

try:
    from .config import HOPEConfig
except ImportError:
    from config import HOPEConfig

class TitansL2(nn.Module):
    """
    Titans Memory Module with L2/Delta Rule Update.
    
    Implements: M_{t+1} = M_t (I - alpha * k_t k_t^T) + beta * v_t k_t^T
    
    Features:
    - Chunkwise Parallel Scan for efficient training
    - Recurrent mode for inference
    - Bounded alpha/beta for stability
    """
    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.chunk_size = 128 # Tunable chunk size for parallel scan
        
        # Projections
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Learnable parameters (bounded)
        self.alpha_raw = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        self.beta_raw = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        
        # Dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw) * 0.5 # Bound to [0, 0.5]

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw) * 0.5 # Bound to [0, 0.5]

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, H, T, D)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Normalize keys for stability
        k = F.normalize(k, dim=-1)
        
        if state is not None and T == 1:
            # Inference mode (step-by-step)
            y, new_state = self.forward_inference(q, k, v, state)
        else:
            # Training mode (chunkwise parallel scan)
            # Note: If state is provided in training (e.g. BPTT), it acts as initial state
            y, new_state = self.forward_train_chunkwise(q, k, v, initial_state=state)
            
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        y = self.resid_dropout(self.c_proj(y))
        
        return y, new_state

    def forward_inference(self, q, k, v, state):
        # q, k, v: (B, H, 1, D)
        # state: (B, H, D, D)
        
        # 1. Read: y = q @ M^T
        y = torch.matmul(q, state.transpose(-1, -2)) 
        
        # 2. Update
        k_t = k.transpose(-1, -2) # (B, H, D, 1)
        v_t = v.transpose(-1, -2) # (B, H, D, 1)
        
        # M_new = M - alpha * (M k) k^T + beta * v k^T
        Mk = torch.matmul(state, k_t) # (B, H, D, 1)
        forget_term = torch.matmul(Mk, k) # (B, H, D, D)
        write_term = torch.matmul(v_t, k) # (B, H, D, D)
        
        new_state = state - self.alpha * forget_term + self.beta * write_term
        
        return y, new_state

    def forward_train_chunkwise(self, q, k, v, initial_state=None):
        B, H, T, D = q.shape
        chunk_size = self.chunk_size
        
        # Pad if necessary
        if T % chunk_size != 0:
            pad_len = chunk_size - (T % chunk_size)
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            T_padded = T + pad_len
        else:
            T_padded = T
            
        num_chunks = T_padded // chunk_size
        
        # Reshape to chunks
        q_chunks = q.view(B, H, num_chunks, chunk_size, D)
        k_chunks = k.view(B, H, num_chunks, chunk_size, D)
        v_chunks = v.view(B, H, num_chunks, chunk_size, D)
        
        # 1. Compute Chunk Operators (A, B)
        A_chunks, B_chunks = self._compute_chunk_operators(k_chunks, v_chunks)
        
        # 2. Global Scan
        if initial_state is None:
            curr_M = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        else:
            curr_M = initial_state
            
        M_starts = [curr_M]
        
        for i in range(num_chunks):
            A = A_chunks[:, :, i]
            B_op = B_chunks[:, :, i]
            next_M = torch.matmul(curr_M, A) + B_op
            M_starts.append(next_M)
            curr_M = next_M
            
        M_starts_tensor = torch.stack(M_starts[:-1], dim=2) # (B, H, num_chunks, D, D)
        
        # 3. Intra-Chunk Processing
        y_chunks = self._process_chunks(q_chunks, k_chunks, v_chunks, M_starts_tensor)
        
        # Reshape back
        y = y_chunks.view(B, H, T_padded, D)
        if T != T_padded:
            y = y[:, :, :T, :]
            
        return y, M_starts[-1]

    def _compute_chunk_operators(self, k_chunks, v_chunks):
        B, H, num_chunks, chunk_size, D = k_chunks.shape
        k_flat = k_chunks.reshape(-1, chunk_size, D)
        v_flat = v_chunks.reshape(-1, chunk_size, D)
        
        A = torch.eye(D, device=k_chunks.device).unsqueeze(0).expand(k_flat.size(0), D, D).clone()
        B_op = torch.zeros_like(A)
        
        alpha = self.alpha.view(1, H, 1, 1).expand(B, H, num_chunks, 1).reshape(-1, 1, 1)
        beta = self.beta.view(1, H, 1, 1).expand(B, H, num_chunks, 1).reshape(-1, 1, 1)
        
        for t in range(chunk_size):
            kt = k_flat[:, t, :].unsqueeze(2)
            vt = v_flat[:, t, :].unsqueeze(2)
            kt_T = kt.transpose(1, 2)
            
            # A_new = A_old (I - alpha k k^T) = A_old - alpha (A_old k) k^T
            Ak = torch.matmul(A, kt)
            A = A - alpha * torch.matmul(Ak, kt_T)
            
            # B_new = B_old (I - alpha k k^T) + beta v k^T
            Bk = torch.matmul(B_op, kt)
            B_op = B_op - alpha * torch.matmul(Bk, kt_T) + beta * torch.matmul(vt, kt_T)
            
        return A.view(B, H, num_chunks, D, D), B_op.view(B, H, num_chunks, D, D)

    def _process_chunks(self, q_chunks, k_chunks, v_chunks, M_starts):
        B, H, num_chunks, chunk_size, D = q_chunks.shape
        q_flat = q_chunks.reshape(-1, chunk_size, D)
        k_flat = k_chunks.reshape(-1, chunk_size, D)
        v_flat = v_chunks.reshape(-1, chunk_size, D)
        M_curr = M_starts.reshape(-1, D, D).clone()
        
        alpha = self.alpha.view(1, H, 1, 1).expand(B, H, num_chunks, 1).reshape(-1, 1, 1)
        beta = self.beta.view(1, H, 1, 1).expand(B, H, num_chunks, 1).reshape(-1, 1, 1)
        
        ys = []
        for t in range(chunk_size):
            qt = q_flat[:, t, :].unsqueeze(2)
            kt = k_flat[:, t, :].unsqueeze(2)
            vt = v_flat[:, t, :].unsqueeze(2)
            
            # Read
            yt = torch.matmul(qt, M_curr.transpose(1, 2))
            ys.append(yt)
            
            # Update
            kt_T = kt.transpose(1, 2)
            Mk = torch.matmul(M_curr, kt)
            M_curr = M_curr - alpha * torch.matmul(Mk, kt_T) + beta * torch.matmul(vt, kt_T)
            
        y = torch.cat(ys, dim=2)
        return y.view(B, H, num_chunks, chunk_size, D)

class CMSBlock(nn.Module):
    """Continuum Memory System MLP Block"""
    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class HOPEBlock(nn.Module):
    """
    Standard HOPE Block: TitansL2 + CMS
    """
    def __init__(self, config: HOPEConfig, layer_idx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.titans = TitansL2(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.cms = CMSBlock(config)
        self.layer_idx = layer_idx

    def forward(self, x, state: Optional[torch.Tensor] = None):
        # Titans
        res, new_state = self.titans(self.ln1(x), state)
        x = x + res
        # CMS
        x = x + self.cms(self.ln2(x))
        return x, new_state

class HOPE(nn.Module):
    """
    Full HOPE Model
    """
    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([HOPEBlock(config, i) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Weight tying
        
        self.apply(self._init_weights)
        
        # Gradient Checkpointing flag
        self.gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(self, idx, targets=None, states=None, pos_offset=0):
        device = idx.device
        b, t = idx.size()
        
        # Positional Encoding
        pos = torch.arange(pos_offset, pos_offset + t, dtype=torch.long, device=device)
        if pos_offset + t > self.config.block_size:
             # Handle overflow if needed, though usually we clip or rotate
             pos = torch.clamp(pos, max=self.config.block_size - 1)
             
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        new_states = []
        
        for i, block in enumerate(self.transformer.h):
            block_state = states[i] if states is not None else None
            
            if self.gradient_checkpointing and self.training:
                # Checkpointing requires inputs to require grad, so we might need a dummy
                # But here x usually requires grad.
                # Note: checkpointing with state is tricky. 
                # We wrap the block call.
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                # Checkpoint only supports tensors. If state is None, we need to handle it.
                # Simplified: only checkpoint if we can.
                x, new_block_state = checkpoint(
                    create_custom_forward(block),
                    x,
                    block_state,
                    use_reentrant=False
                )
            else:
                x, new_block_state = block(x, state=block_state)
                
            new_states.append(new_block_state)
            
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss, new_states

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stream=False):
        """
        Stateful generation.
        If stream=True, yields tokens one by one.
        """
        # 1. Prefill
        logits, _, states = self(idx, pos_offset=0)
        
        # Last token
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        if stream:
            yield idx_next
        else:
            out = torch.cat((idx, idx_next), dim=1)
            
        current_pos = idx.size(1)
        
        for _ in range(max_new_tokens - 1):
            logits, _, states = self(idx_next, states=states, pos_offset=current_pos)
            
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            current_pos += 1
            
            if stream:
                yield idx_next
            else:
                out = torch.cat((out, idx_next), dim=1)
                
        if not stream:
            return out
