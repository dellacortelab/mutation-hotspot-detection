
from torch.utils.data import IterableDataset
import os
import numpy as np
import sentencepiece as spm
import torch
import ast

from ..gen_dataset.gen_uniparc_dataset import UniparcDatasetPreprocessor

class ProtSeqDataset(IterableDataset):
    """Generic protein sequence dataset yielding protein sequences as strings"""
    def __init__(
        self,
        dataset_dir='/data/uniparc',
        seq_filename='sequences.txt',
        mode='train',
        tokenizer_prefix='uniparc_tokenizer',
        vocab_size=8000,
        no_verification=False
        ):
        super().__init__()

        preproc = UniparcDatasetPreprocessor(
            dataset_dir=dataset_dir,
            seq_filename=seq_filename,
            tokenizer_prefix=tokenizer_prefix,
            vocab_size=vocab_size,
            no_verification=no_verification
        )
        preproc.prepare_dataset()

        self.data_file = os.path.join(dataset_dir, mode + '.txt')

        # Load tokenizer
        tokenizer_file = os.path.join(dataset_dir, tokenizer_prefix + '.model')
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_file)
        self.true_vocab_size = self.tokenizer.vocab_size()

    def get_vocab_size(self):
        """Get the true vocab size, which, for small datasets, may be smaller than
        the specified vocab size
        Returns:
            (int): the true vocab size
        """
        return self.true_vocab_size

    def tokenize(self, sequence):
        """Tokenize the sequence
        Args:
            sequence (str): the sequence to tokenize
        Returns:
            (torch.Tensor (long)): the integer-encoded sequence, with start/end tokens
        """
        # Tokenize sequence
        encoded_seq = self.tokenizer.encode(sequence, out_type=int, add_bos=True, add_eos=True)
        encoded_seq = torch.tensor(encoded_seq, dtype=torch.long)
        return encoded_seq

    def crop_seq(self, sequence):
        """Crop the sequence to a given max length
        Args:
            sequence (torch.Tensor (long)): the integer-encoded sequence
        Returns:
            sequence (torch.Tensor (long)): the sequence, cropped to crop_length
        """
        # Only crop if necessary
        if self.crop_length is not None and len(sequence) > self.crop_length:
            max_start_idx = len(sequence) - self.crop_length
            start_idx = np.random.randint(max_start_idx + 1)
            return sequence[start_idx:start_idx + self.crop_length]

        return sequence

    def preprocess_seq(self, sequence):
        """Conduct preprocessing on each sequence
        Args:
            sequence (str): a sequence read from file
        Returns:
            sequence (torch.Tensor (long)): the processed sequence
        """
        sequence = self.tokenize(sequence)
        sequence = self.crop_seq(sequence)
        return sequence

    def __iter__(self):
        """Return an iterator over the dataset
        Returns:
            (Iterable): the iterator
        """
        # Follow this pattern: https://medium.com/swlh/how-to-use-pytorch-dataloaders-to-work-with-enormously-large-text-files-bbd672e955a0
        seq_iter = open(self.data_file)
        return map(self.preprocess_seq, seq_iter)