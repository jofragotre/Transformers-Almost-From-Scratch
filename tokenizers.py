from typing import List

class BaseTokenizer:
    def __init__(self):
        pass

    def encode(self, text):
        raise NotImplementedError("Subclasses should implement this method!")

    def decode(self, tokens):
        raise NotImplementedError("Subclasses should implement this method!")

class SimpleTokenizer(BaseTokenizer):

    def __init__(self, vocab=List[str]):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }

    def encode(self, text):
        return [self.stoi[char] for char in text]

    def decode(self, tokens):
        return ''.join([self.itos[token] for token in tokens])