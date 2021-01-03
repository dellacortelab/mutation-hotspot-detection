#######################################################################################################
# Sequence Dataset classes
#######################################################################################################

from torch.utils.data import IterableDataset
import os
import numpy as np
import sentencepiece as spm
import torch
import ast

from ..logger import Logger

class ManyToOneDataset(IterableDataset):
    """Generic sequence dataset yielding a sequence (as a Tensor of indices) and its corresponding scalar label."""
    def __init__(self,
        dataset_dir='/data/',
        seq_prefix='sequences',
        label_prefix='labels',
        tokenizer_prefix='tokenizer',
        mode='train',
        vocab_size=8000,
        crop_length=None,
        no_verification=False,
        dataset_generator_class=None,
        n_seq=None
        ):
        super().__init__()

        self.crop_length=crop_length

        if dataset_generator_class is not None:
            dataset_generator = dataset_generator_class(
                dataset_dir=dataset_dir,
                seq_prefix=seq_prefix,
                label_prefix=label_prefix,
                tokenizer_prefix=tokenizer_prefix,
                vocab_size=vocab_size,
                no_verification=no_verification,
                n_seq=n_seq
            )
            dataset_generator.prepare_dataset()

        self.set_length(dataset_generator=dataset_generator, mode=mode)

        self.seq_file = os.path.join(dataset_dir, seq_prefix + '_' + mode + '.txt')
        self.label_file = os.path.join(dataset_dir, label_prefix + '_' + mode + '.txt')

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

    def set_length(self, dataset_generator, mode):
        # Get number of sequences from the dataset metadata
        dataset_generator.load_dataset_metadata()
        if mode == 'train':
            self.n_seq = len(dataset_generator.metadata.train_idx)
        if mode == 'val':
            self.n_seq = len(dataset_generator.metadata.val_idx)
        if mode == 'test':
            self.n_seq = len(dataset_generator.metadata.test_idx)

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


    def preprocess_label(self, label):
        """Conduct preprocessing on each label
        Args:
            sequence (str): a sequence read from file
        Returns:
            sequence (torch.Tensor (long)): the processed sequence
        """
        # Interpret string from file as a scalar or a list of scalars,
        # then cast it to a Tensor
        label = ast.literal_eval(label.rstrip())
        # We do this because we need a 1d tensor
        return torch.tensor(label if isinstance(label, list) else [label])

    def __iter__(self):
        """Return an iterator over the dataset
        Returns:
            (Iterable): the iterator
        """
        seq_iter = open(self.seq_file, 'r')
        seq_iter = map(self.preprocess_seq, seq_iter)
        label_iter = open(self.label_file, 'r')
        label_iter = map(self.preprocess_label, label_iter)
        return zip(seq_iter, label_iter)

    def __len__(self):
        return self.n_seq

class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

        # Initialize the random buffer
        self.initialize_buffer()

    def initialize_buffer(self):
        self.shufbuf = []
        try:
            # Fill buffer of desired size with dataset
            self.dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                self.shufbuf.append(next(self.dataset_iter))
        except (StopIteration, MemoryError):
            # End when we run out of items or run out of memory
            # In case we hit a memory error, correct the length of the buffer.
            # Going until a MemoryError is still probably a bad idea. Better to leave some memory left over.
            self.buffer_size = len(self.shufbuf)

        # Get indices to yield the remaining items in a random order
        self.last_indices = np.random.choice(self.buffer_size, self.buffer_size, replace=False)
        self.last_indices_cnt = 0
        self.i = 0

    def __iter__(self):
        # import traceback; traceback.print_stack()
        # print("CALLED ITER------------------------------------------")
        # import pdb; pdb.set_trace()
        self.initialize_buffer()
        return self

    def __next__(self):
        try:
            # Return an item from a random position
            evict_idx = np.random.randint(self.buffer_size)
            item = self.shufbuf[evict_idx]
            # Get item NOT in buffer
            replace_item = next(self.dataset_iter)
            # Insert the new item into that random position
            self.shufbuf[evict_idx] = replace_item
            # No guarantees on pure randomness, but at least you are always sampling
            # from index 0 to index (self.buffer_size + n_yielded items)
            self.i += 1
            # print("I1", self.i)
            # print(self.i)
            return item

        except StopIteration:

            if self.last_indices_cnt == len(self.last_indices):
                raise StopIteration
            
            item = self.shufbuf[self.last_indices[self.last_indices_cnt]]
            self.last_indices_cnt += 1
            self.i += 1
            # print("I2", self.i)
            # print("LASTIDX", self.last_indices_cnt)

            return item

    def __len__(self):
        return len(self.dataset)


def padding_collate_fn(batch):
    batch_size = len(batch)
    # Get the max length of an input sequence (each item is an input seq and a label)
    max_length = max([len(item[0]) for item in batch])
    # Get the length of a label (assumes fixed size)
    out_dim = len(batch[0][1])

    batch_in_seq = torch.zeros((batch_size, max_length), dtype=torch.long)
    batch_out = torch.zeros((batch_size, out_dim), dtype=torch.float)

    for i, (in_sequence, out_sequence) in enumerate(batch):
        batch_in_seq[i, :len(in_sequence)] = in_sequence
        batch_out[i, :] = out_sequence

    return batch_in_seq, batch_out

def shuffle_dataset_test():
    from transformer_autoencoder.dataset.seq_dataset import ShuffleDataset
    from transformer_autoencoder.dataset.mutation_activity_dataset import MutationActivityDataset
    train_dataset = MutationActivityDataset(mode='train', vocab_size=200, no_verification=True, dataset_dir='/data/mutation_activity/dataset', n_seq=int(50))
    train_dataset = ShuffleDataset(train_dataset, buffer_size=5)
    while True:
        next(train_dataset)