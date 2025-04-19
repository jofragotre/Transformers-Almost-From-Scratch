import torch
from tokenizers import BaseTokenizer

class TextDataset:
    def __init__(self, file_path: str,
                tokenizer: BaseTokenizer,
                block_size: int = 128,
                batch_size: int = 32,
                train_split: float = 0.9):
        """
        Initialize the TextDataset.
        Args:
            file_path (str): Path to the text file.
            block_size (int): Size of each block of text.
            batch_size (int): Number of sequences to process in parallel.
        """
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Check the diferent characters in the text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Create a tokenizer
        self.tokenizer = tokenizer

        # Encode the text to integers
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # Split the data into training and validation sets
        self.train_data = data[:int(train_split*len(data))]
        self.val_data = data[int(train_split*len(data)):]
        
        # Set batch size and block size
        self.batch_size = batch_size
        self.block_size = block_size
    
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y