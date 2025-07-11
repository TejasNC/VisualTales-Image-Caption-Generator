import os
import re
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import pickle


class CapCollat:
    """
    Pads a batch of (image, caption) pairs where captions are variable-length.

    Args:
        pad_seq (int): Index used for <PAD> token.
        batch_first (bool): Whether to return tensors with batch as the first dimension.
    """

    def __init__(self, pad_seq: int, batch_first: bool = False):
        self.pad_seq = pad_seq
        self.batch_first = batch_first

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate a batch of (image, caption) pairs into padded tensors.

        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor]]): A list of (image, caption) pairs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of images with shape [batch_size, 3, H, W].
                - A tensor of padded captions with shape [batch_size, max_seq_len] (if batch_first=True).
        """
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            raise ValueError("All items in the batch are None.")

        # Separate images and captions
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        caps = [item[1] for item in batch]
        padded_caps = pad_sequence(
            caps, batch_first=self.batch_first, padding_value=self.pad_seq
        )

        # Optionally return lengths
        lengths = torch.tensor([len(cap) for cap in caps])

        return imgs, padded_caps, lengths


### Vocabulary Class

import spacy
from collections import Counter
from typing import List

# Load the spaCy model with error handling
try:
    spacy_eng = spacy.load(
        "en_core_web_sm", disable=["ner", "parser"]
    )  # Disable unnecessary components for speed
except OSError:
    raise RuntimeError(
        "The spaCy model 'en_core_web_sm' is not installed. Please install it using 'python -m spacy download en_core_web_sm'."
    )


class Vocabulary:
    """
    A class to build and manage a vocabulary for text data.
    """

    def __init__(self, freq_threshold: int):
        """
        Initialize the Vocabulary object.

        Args:
            freq_threshold (int): Minimum frequency for a word to be included in the vocabulary.
        """
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self) -> int:
        """
        Return the size of the vocabulary.
        """
        return len(self.itos)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize a given text into a list of lowercase tokens.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list: List[str]) -> None:
        """
        Build the vocabulary from a list of sentences.

        Args:
            sentence_list (List[str]): A list of sentences to build the vocabulary from.
        """
        frequencies = Counter()

        for sentence in sentence_list:
            for word in self.tokenize(str(sentence)):
                frequencies[word] += 1

        idx = 4  # Start indexing after special tokens
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text: str) -> List[int]:
        """
        Convert a text into a list of numerical tokens based on the vocabulary.

        Args:
            text (str): The input text to numericalize.

        Returns:
            List[int]: A list of numerical tokens.
        """
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

    def save_vocab(self, path: str) -> None:
        """
        Save the vocabulary to a file.

        Args:
            path (str): Path to save the vocabulary file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "freq_threshold": self.freq_threshold,
                    "itos": self.itos,
                    "stoi": self.stoi,
                },
                f,
            )
        print(f"Vocabulary saved to {path}")

    @classmethod
    def load_vocab(cls, path: str):
        """
        Load a vocabulary from a file.

        Args:
            path (str): Path to the vocabulary file.

        Returns:
            Vocabulary: A loaded Vocabulary object.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        vocab = cls(data["freq_threshold"])
        vocab.itos = data["itos"]
        vocab.stoi = data["stoi"]
        return vocab


import os
import re


def get_latest_checkpoint(checkpoint_dir):
    """Returns the latest checkpoint file path based on epoch number in filename."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

    def extract_epoch(filename):
        match = re.search(r"(\d+)(?=\.pth$)", filename)
        return int(match.group(1)) if match else -1

    checkpoints = sorted(checkpoints, key=extract_epoch, reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0]) if checkpoints else None


def calculate_bleu(references, hypotheses, max_n=4):
    """
    Calculate BLEU-1 to BLEU-4 scores for a batch of hypotheses against their references.

    Args:
        references (List[List[str]]): List of reference sentences, each as a list of tokens
        hypotheses (List[List[str]]): List of hypothesis sentences, each as a list of tokens
        max_n (int): Maximum n-gram size for BLEU calculation

    Returns:
        Dict[str, float]: Dictionary with BLEU-1 to BLEU-4 scores
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    # Ensure we have nltk's smoothing function
    smoothie = SmoothingFunction().method1

    # Calculate corpus BLEU scores for each n-gram size
    scores = {}
    for n in range(1, max_n + 1):
        scores[f"BLEU-{n}"] = corpus_bleu(
            [[ref] for ref in references],
            hypotheses,
            weights=tuple([1 / n] * n + [0] * (max_n - n)),
            smoothing_function=smoothie,
        )

    # Individual sentence BLEU scores if needed for detailed analysis
    sentence_scores = []
    for ref, hyp in zip(references, hypotheses):
        sentence_bleus = {}
        for n in range(1, max_n + 1):
            sentence_bleus[f"BLEU-{n}"] = sentence_bleu(
                [ref],
                hyp,
                weights=tuple([1 / n] * n + [0] * (max_n - n)),
                smoothing_function=smoothie,
            )
        sentence_scores.append(sentence_bleus)

    return scores, sentence_scores
