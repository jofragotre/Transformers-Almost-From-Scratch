import torch
import tqdm
from tokenizers import SimpleTokenizer
from dataset import TextDataset
from models import SmallLanguageModel

# Hyper parameters (TODO: move to config file like hydra)
DATASET_PATH = './dataset/input.txt'
BATCH_SIZE = 64
BLOCK_SIZE = 256
TRAIN_SPLIT = 0.9
LEARNING_RATE = 3e-4
STEPS = 5000
EVAL_INTERVAL = 200
EVAL_STEPS = 200

N_EMBED = 128
N_HEADS = 4
N_LAYERS = 4
DROPOUT = 0.15

# Define device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print("Using device:", device)

# Initialize the tokenizer
with open(DATASET_PATH, 'r') as f:
    text = f.read()
    chars = sorted(list(set(text)))
    tokenizer = SimpleTokenizer(vocab=chars)

# Initialize the TextDataset
dataset = TextDataset(file_path=DATASET_PATH,
                      tokenizer=tokenizer,
                      block_size=BLOCK_SIZE,
                      batch_size=BATCH_SIZE,
                      train_split=TRAIN_SPLIT)

# Get a batch of data
x, y = dataset.get_batch('train')
print("Input batch shape:", x.shape)
print("Target batch shape:", y.shape)

# Initialize the model
model = SmallLanguageModel(vocab_size=dataset.vocab_size,
                           n_embed=N_EMBED,
                           block_size=BLOCK_SIZE,
                           num_heads=N_HEADS,
                           num_layers=N_LAYERS).to(device)

# Create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Generate new text
print(tokenizer.decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=100)[0].tolist()))

# Estimate the loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_STEPS)
        for k in range(EVAL_STEPS):
            x, y = dataset.get_batch(split)
            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in tqdm.tqdm(range(STEPS)):

    # Print progress
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get a batch of data
    x, y = dataset.get_batch('train')
    x, y= x.to(device), y.to(device)

    # Forward pass
    logits, loss = model(x, y)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate new text
print(tokenizer.decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long).to(device), max_new_tokens=500)[0].tolist()))