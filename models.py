import torch
import torch.nn as nn
from torch.nn import functional as F
from self_attention import Block

class SmallLanguageModel(nn.Module):
    """Small Language Model with token and position embeddings."""

    def __init__(self,
                vocab_size: int,
                n_embed: int,
                block_size: int,
                num_heads: int,
                num_layers: int,
                dropout=0.1) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(num_heads=num_heads, block_size=block_size, embed_dim=n_embed, dropout=dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self,
                idx: torch.Tensor,
                targets: torch.Tensor = None):
        
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C=n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C=n_embed)
        x = tok_emb + pos_emb # (B,T,C=n_embed)
        x = self.blocks(x) # (B,T,C=n_embed)
        x = self.ln_f(x) # (B,T,C=n_embed)
        logits = self.lm_head(x) # (B,T,C=vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx