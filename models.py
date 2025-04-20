import torch
import torch.nn as nn
from torch.nn import functional as F
from self_attention import Block

class SmallLanguageModel(nn.Module):
    """Small Language Model (Decoder only) following the attention is all you need paper."""

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

class SmallVisionTransformer(nn.Module):
    """Small Vision Transformer following the ViT paper (encoder only)."""

    def __init__(self,
                patch_size: int = 16,
                n_embed: int = 768,
                num_heads: int = 8,
                num_layers: int = 6,
                dropout: float = 0.1,
                num_classes: int = 10,
                ):
        super().__init__()

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.patch_len = patch_size * patch_size * 3  # Assuming 3 channels (RGB)
        self.n_embed = n_embed
        
        # Projection layer to map patches to the embedding dimension
        self.projection = nn.Linear(self.patch_len, self.n_embed)  # (patch_len, n_embed)

        # Learnable parameters for classification token and position embedding
        self.classification_token = nn.Parameter(torch.randn(1, self.n_embed))  # (1, patch_len)
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_embed))  # (1D  position embedding)

        # Here use_tril is set to False because we are not using a causal mask, all tokens can attend to all other tokens
        # in the self-attention block.
        self.blocks = nn.Sequential(*[Block(num_heads=num_heads, block_size=None, embed_dim=n_embed, dropout=dropout, use_tril=False) for _ in range(num_layers)])

        # Classification head
        self.classification_head = nn.Linear(n_embed, num_classes)  # Assuming 10 classes for classification

    def patchify(self, x):
        """
        Convert an image into patches.
        Args:
            x: Input tensor of shape (B, C, H, W), where
               B = batch size, C = number of channels, H = height, W = width.
        
        Returns:
            Tensor of shape (B, num_patches, patch_size * patch_size * C).
        """
        # Assuming x is of shape (B, C, H, W)
        B, C, H, W = x.shape
        patch_size = 16
        assert H % patch_size == 0, "Height must be divisible by patch size"
        assert W % patch_size == 0, "Width must be divisible by patch size"

        # Unfold the image into patches (sometimes done with einops)
        patches = self.unfold(x) # (B, C * patch_size * patch_size, num_patches)
        patches = patches.transpose(1, 2) # (B, num_patches, C * patch_size * patch_size)
        
        return patches

    def forward(self, x):

        # x is of shape (B, C, H, W)
        B, C, H, W = x.shape

        # Convert the image into patches
        patches = self.patchify(x)
        
        # Project patches to the embedding dimension
        patches = self.projection(patches) # B and N_patches are not affected by the projection
        
        # Append the classification token to the patches
        classification_token = self.classification_token.expand(B, -1, -1)
        patches = torch.cat((classification_token, patches), dim=1) # (B, num_patches + 1, n_embed)

        # Add position embedding
        position_embedding = self.position_embedding.expand(B, -1, -1)
        patches = patches + position_embedding # (B, num_patches + 1, n_embed)
        
        # Pass through the transformer blocks
        x = self.blocks(patches)
        
        # Take the output of the classification token
        x = x[:, 0, :]  # (B, n_embed)

        # Classification head
        logits = self.classification_head(x)

        return logits


    

if __name__ == "__main__":
    
    x = torch.randn(10, 3, 64, 64) # (B, C, H, W)
    
    model = SmallVisionTransformer(patch_size=16, n_embed=768, num_heads=8, num_layers=6, dropout=0.1, num_classes=10)

    logits = model(x) # (B, num_classes)
    print(logits.shape) # (B, num_classes)