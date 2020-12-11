#######################################################################################################
# Sequence Dataset classes
#######################################################################################################

from torch.utils.data import IterableDataset
import os
import numpy as np
import sentencepiece as spm
import torch

from .gen_seq_dataset import UniparcDatasetPreprocessor

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
		super(ProtSeqDataset, self).__init__()

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



class ShuffleDataset(IterableDataset):
	def __init__(self, dataset, buffer_size):
		super().__init__()
		self.dataset = dataset
		self.buffer_size = buffer_size

		# Initialize the random buffer
		self.initialize_buffer()
		# Get indices to yield the last items in a random order
		self.indices = np.random.choice(self.buffer_size, self.buffer_size, replace=False)

	def initialize_buffer(self):
		self.shufbuf = []
		try:
			# Fill buffer of desired size with dataset
			self.dataset_iter = iter(self.dataset)
			for i in range(self.buffer_size):
				self.shufbuf.append(next(self.dataset_iter))
		except Exception as e:
			# In case we hit a memory error, correct the length of the buffer.
			# This is still probably a bad idea. Better to leave some memory left over.
			self.buffer_size = len(self.shufbuf)
	

	def __iter__(self):

		try:

			while True:
				try:
					# Get item NOT in buffer
					item = next(self.dataset_iter)
					# Return an item from a random position
					evict_idx = np.random.randint(self.buffer_size)
					yield self.shufbuf[evict_idx]
					# Insert the new item into that random position
					self.shufbuf[evict_idx] = item
					# No guarantees on pure randomness, but at least you are always sampling
					# from index 0 to index (self.buffer_size + n_yielded items)
				except StopIteration:
					# Once all items from the dataset are in the shuffle buffer, break
					break

			# yield the remaining items one-by-one in a random order
			for idx in self.indices:
				yield self.shufbuf[idx]

		except GeneratorExit:
			pass


def padding_collate_fn(batch):
	batch_size = len(batch)
	# Get the max length of an input sequence (each item is an input seq, output seq, and mask indices)
	max_length = max([len(seq[0]) for seq in batch])

	batch_in_seq = torch.zeros((batch_size, max_length), dtype=torch.long)
	batch_out_seq = torch.zeros((batch_size, max_length), dtype=torch.long)
	batch_mask_idx = torch.zeros((batch_size, max_length), dtype=torch.long)

	for i, (in_sequence, out_sequence, mask_idx) in enumerate(batch):
		batch_in_seq[i, :len(in_sequence)] = in_sequence
		batch_out_seq[i, :len(out_sequence)] = out_sequence
		batch_mask_idx[i, mask_idx] = 1

	return batch_in_seq, batch_out_seq, batch_mask_idx