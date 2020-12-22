#######################################################################################################
# Classes for downloading and preparing the dataset
#######################################################################################################


from torch.utils.data import IterableDataset
import torch
import sentencepiece as spm
import os
import subprocess
import re
from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl

class DatasetGenerator():
    """Base class for dataset generators"""
    def __init__(self):
        pass

    def prepare_dataset(self):
        pass

class SeqDatasetGenerator(DatasetGenerator):
	"""Base class for generating a sequence dataset"""
	def __init__(self, 
		dataset_dir="/data/uniparc",
		seq_prefix="sequences",
		tokenizer_prefix='tokenizer',
		vocab_size=8000,
		input_sentence_size=int(1e9),
		shuffle_input_sentence=True,
		train_val_test_split=[.8, .1, .1],
		no_verification=False
		):
		super().__init__()
		self.dataset_dir = dataset_dir
		self.seq_file = os.path.join(dataset_dir, seq_prefix + '.txt')
		self.train_seq_file = os.path.join(dataset_dir, seq_prefix + '_train.txt')
		self.val_seq_file = os.path.join(dataset_dir, seq_prefix + '_val.txt')
		self.test_seq_file = os.path.join(dataset_dir, seq_prefix + '_test.txt')
		self.dataset_metadata_file = os.path.join(dataset_dir, 'dataset_metadata.pkl')
		self.tokenizer_file_prefix = os.path.join(dataset_dir, tokenizer_prefix)
		self.vocab_size = vocab_size
		self.input_sentence_size = input_sentence_size
		self.shuffle_input_sentence = shuffle_input_sentence
		self.train_val_test_split = train_val_test_split
		self.no_verification = no_verification

	def prepare_dataset(self):
		"""Download, unzip, and preprocess sequences, collecting aggregate statistics about the 
		dataset along the way"""

		if self.no_verification:
			print("Performing all steps of dataset generation without user verification. \
				This could take a full day.")
		else:
			print("Confirming with user at each significant dataset generation step.")

		if not os.path.exists(self.seq_file):
			if not self.no_verification:
				response = input("About to create sequence file. This could take several hours. Continue? [Y/n]")
				if response != 'Y':
					raise ValueError("Sequence file doesn't exist")
			else:
				print("About to create sequence file. This could take several hours.")
			self.prepare_sequences_file()

		if not os.path.exists(self.train_seq_file):
			if not self.no_verification:
				response = input("About to create train/test/validation files. This could take several hours. Continue? [Y/n]")
				if response != 'Y':
					raise ValueError("Train/test/val split doesn't exist")
			else:
				print("About to create train/test/validation files. This could take several hours.")
			self.split_train_val_test()

		if not os.path.exists(self.tokenizer_file_prefix + '.model'):
			if not self.no_verification:
				response = input("About to create tokenizer. This could take several hours. Continue? [Y/n]")
				if response != 'Y':
					raise ValueError("Tokenizer doesn't exist")
			else:
				print("About to train tokenizer. This could take several hours.")
			self.train_tokenizer()

	def prepare_sequences_file():
		"""To be implemented by child class. May include downloading and unzipping, or generating
		synthetic sequences, then placing them in self.seq_file, one sequence on each line."""
		pass

	def plot_histogram(self, seq_lengths):
		"""Create histogram of sequence lengths and save to file
		Args:
			seq_lengths (np.ndarray): array of sequence lengths
		"""			
		bin_size = 25
		max_bin = 750
		bins = np.arange(0, max_bin + bin_size, bin_size)
		fig, ax = plt.subplots(figsize=(15, 5))
		_, bins, patches = plt.hist(np.clip(seq_lengths, bins[0], bins[-1]), bins=bins)
		xlabels = bins[1:].astype(str)
		xlabels[-1] += '+'
		N_labels = len(xlabels)
		plt.xlim([0, max_bin])
		plt.xticks(bin_size * np.arange(N_labels) + bin_size/2.)
		ax.set_xticklabels(xlabels)
		plt.yticks()
		plt.title(f"Sequence Length Histogram - Total Sequences: {len(seq_lengths)}")
		plt.xlabel("Sequence Length")
		plt.ylabel("Number of Sequences")
		plt.setp(patches, linewidth=0)
		fig.tight_layout()
		seq_lengths_fig_file = os.path.join(self.dataset_dir, "seq_lengths.png")
		plt.savefig(seq_lengths_fig_file)

	def log_sequence_data(self, seq_lengths):
		"""Log aggregate sequence data for data validation
		Args:
			seq_lengths (np.ndarray): array of sequence lengths	
		"""
		metadata = SeqDatasetMetadata(n_seq=len(seq_lengths), seq_lengths=seq_lengths)
		# Log sequence data
		with open(self.dataset_metadata_file, 'wb') as metadata_file:
			pkl.dump(metadata, metadata_file)
		# Create a histogram figure
		self.plot_histogram(seq_lengths)

	def load_dataset_metadata(self):
		# Find total number of sequences
		with open(self.dataset_metadata_file, 'rb') as metadata_file:
			self.metadata = pkl.load(metadata_file)

	def split_train_val_test(self):
		"""Split the dataset betweet training, test, and validation sets"""
		# Find total number of sequences
		with open(self.dataset_metadata_file, 'rb') as metadata_file:
			metadata = pkl.load(metadata_file)
		n_seq = metadata.n_seq
		indices = np.arange(n_seq)
		# Get train indices
		n_train = int( self.train_val_test_split[0] * n_seq )
		train_indices = np.random.choice(indices, size=n_train, replace=False)
		self.train_indices_set = set(train_indices)
		remaining_indices = np.setxor1d(indices, train_indices)
		# Get test indices
		n_test = int( self.train_val_test_split[1] * n_seq )
		test_indices = np.random.choice(remaining_indices, size=n_test, replace=False)
		self.test_indices_set = set(test_indices)
		# Get val indices
		remaining_indices = np.setxor1d(remaining_indices, test_indices)
		val_indices = remaining_indices
		self.val_indices_set = set(val_indices)
		metadata.train_idx = train_indices
		metadata.test_idx = test_indices
		metadata.val_idx = val_indices
		# Write some metadata about train/test/val split
		with open(self.dataset_metadata_file, 'wb') as metadata_file:
			pkl.dump(metadata, metadata_file)

		# I once hit a segfault here. If you do, try uncommenting the line below for debugging
		# import faulthandler; faulthandler.enable()
		with open(self.seq_file, 'r') as seq_file:
			with open(self.train_seq_file, 'w') as train_file:
				with open(self.test_seq_file, 'w') as test_file:
					with open(self.val_seq_file, 'w') as val_file:
						for i, line in enumerate(seq_file):
							if i in self.train_indices_set:
								train_file.write(line)
							elif i in self.test_indices_set:
								test_file.write(line)
							elif i in self.val_indices_set:
								val_file.write(line)
							else:
								raise ValueError("All sequences should be assigned to a train/test/val set")				

	def split_other_dataset_train_val_test(self, other_dataset_file):
		"""Split another dataset according to the same train/val/test indices. Only works if the dataset
		is the exact same number of lines"""
		# Parse file names for other dataset
		basename = os.path.splitext(other_dataset_file)[0]
		other_dataset_train_file = basename + '_train.txt'
		other_dataset_val_file = basename + '_val.txt'
		other_dataset_test_file = basename + '_test.txt'

		try:
			self.train_indices_set
		except AttributeError:
			self.load_dataset_metadata()
			self.train_indices_set, self.val_indices_set, self.test_indices_set = set(self.metadata.train_idx), set(self.metadata.val_idx), set(self.metadata.test_idx)

		if not os.path.exists(other_dataset_train_file):
			# I once hit a segfault here. If you do, try uncommenting the line below for debugging
			# import faulthandler; faulthandler.enable()
			with open(other_dataset_file, 'r') as other_file:
				with open(other_dataset_train_file, 'w') as train_file:
					with open(other_dataset_val_file, 'w') as val_file:
						with open(other_dataset_test_file, 'w') as test_file:
							for i, line in enumerate(other_file):
								if i in self.train_indices_set:
									train_file.write(line)
								elif i in self.val_indices_set:
									val_file.write(line)
								elif i in self.test_indices_set:
									test_file.write(line)
								else:
									raise ValueError("All sequences should be assigned to a train/val/test set")	

	def train_tokenizer(self):
		"""Train a tokenizer on the training data if a trained tokenizer doesn't exist"""
		spm.SentencePieceTrainer.train(
			input=self.train_seq_file, 
			model_prefix=self.tokenizer_file_prefix, 
			vocab_size=self.vocab_size, 
			hard_vocab_limit=False,
			add_dummy_prefix=False,
			input_sentence_size=self.input_sentence_size,
			shuffle_input_sentence=self.shuffle_input_sentence,
			pad_id=0,
			bos_id=1,
			eos_id=2,
			unk_id=3
		)

class SeqDatasetMetadata():
	"""Data class that holds metadata about the dataset"""
	def __init__(
		self,
		seq_lengths=None,
		n_seq=None,
		train_idx=None,
		test_idx=None,
		val_idx=None
	):
		self.seq_lengths = seq_lengths
		self.n_seq = n_seq
		self.train_idx = train_idx
		self.test_idx = test_idx
		self.val_idx = val_idx

	def __str__(self):
		train_len = str(len(self.train_idx)) if self.train_idx is not None else "None"
		test_len = str(len(self.test_idx)) if self.test_idx is not None else "None"
		val_len = str(len(self.val_idx)) if self.val_idx is not None else "None"
		n_seq = str(self.n_seq) if self.n_seq is not None else "None"
		
		return f"Number of sequences: {n_seq}\n" + \
				f"Number of train items: {train_len}\n" + \
				f"Number of test items: {test_len}\n" + \
				f"Number of val items: {val_len}"

def check_dataset_metadata(file_path='/data/uniparc/dataset_metadata.pkl'):
	# Find total number of sequences
	with open(file_path, 'rb') as metadata_file:
		metadata = pkl.load(metadata_file)
		print(metadata)

# Remove all files except fasta file
# Be careful using this! Make sure you are in the correct directory
# ls -1 | grep -v 'fasta' | xargs rm -f
# ls -1 | grep -v 'tokenizer' | xargs rm -f