"""
HOPE Model Configuration
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class HOPEConfig:
    # Model dimensions (approx 300M params)
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 22
    block_size: int = 2048
    vocab_size: int = 50257  # GPT-2 tokenizer
    dropout: float = 0.1
    bias: bool = False
    
    # HOPE specific
    cms_update_periods: List[int] = field(default_factory=lambda: [1, 4, 16])
    learning_rate_memory: float = 1e-2  # Learning rate for the inner loop (Titans/CMS)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
