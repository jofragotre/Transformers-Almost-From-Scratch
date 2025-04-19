import torch
import torch.nn as nn
from torch import Tensor
from typing import List

class SelfAttentionHead(nn.Module):
    
    def __init__(self,
                 embed_dim: int,
                 head_size: int,
                 block_size: int,
                 dropout: float = 0.1,
                 ) -> None:
        super().__init__()
        
        self.head_size = head_size
        self.block_size = block_size
        self.embed_dim = embed_dim

        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C), where
               B = batch size, T = sequence length, C = embedding dimension.
        
        Returns:
            Tensor of shape (B, T, head_size).
        """
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = torch.softmax(wei, dim=-1)  # (B, T, T)

        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    
    def __init__(self,
                 embed_dim: int,
                 head_size: int,
                 num_heads: int,
                 block_size: int,
                 dropout: float = 0.1,
                 ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(embed_dim, head_size, block_size, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C), where
               B = batch size, T = sequence length, C = embedding dimension.
        
        Returns:
            Tensor of shape (B, T, embed_dim).
        """
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, num_heads * head_size)
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    
    def __init__(self,
                 n_embed: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embed, 4 * n_embed)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C), where
               B = batch size, T = sequence length, C = embedding dimension.
        
        Returns:
            Tensor of shape (B, T, embed_dim).
        """
        out = self.fc(x)  # (B, T, ff_dim)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    
    def __init__(self,
                 num_heads: int,
                 block_size: int,
                 embed_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()

        # Make output of each multi head the same size as the input
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        head_size = embed_dim // num_heads

        self.sa = MultiHeadAttention(embed_dim, head_size, num_heads, block_size, dropout=dropout) # Output shape: (B, T, embed_dim)
        self.ff = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C), where
               B = batch size, T = sequence length, C = embedding dimension.
        
        Returns:
            Tensor of shape (B, T, embed_dim).
        """
        x = x + self.sa(self.ln1(x))  # (B, T, embed_dim)
        x = x + self.ff(self.ln2(x))  # (B, T, embed_dim)
        return x

if __name__ == "__main__":

    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)

    HEAD_SIZE = 8
    NUM_HEADS = 4

    self_attention_head = SelfAttentionHead(head_size=HEAD_SIZE, block_size=T, embed_dim=C)
    out = self_attention_head(x)
    print("SelfAttentionHead output shape:", out.shape, f" Expected shape: {[B, T, HEAD_SIZE]}")

    multi_head_attention = MultiHeadAttention(head_size=HEAD_SIZE, num_heads=NUM_HEADS, block_size=T, embed_dim=C)
    out = multi_head_attention(x)
    print("MultiHeadAttention output shape:", out.shape, f" Expected shape: {[B, T, HEAD_SIZE * NUM_HEADS]}")

    block = Block(num_heads=NUM_HEADS, block_size=T, embed_dim=C)
    out = block(x)
    print("Block output shape:", out.shape, f" Expected shape: {[B, T, C]}")
