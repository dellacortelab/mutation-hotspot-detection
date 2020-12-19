#######################################################################################################
# Classes for downloading and preparing the sequence dataset
#######################################################################################################

# TODO: Host these files in the cloud so you can just download them

from torch.utils.data import IterableDataset
import torch
import sentencepiece as spm
import os
import subprocess
import re
from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl

class UniparcDatasetPreprocessor():
	"""Class for downloading the UniParc dataset for the first time, preprocessing data
	to prepare inputs for the transformer. Use after CathDatasetDownloader to download
	the CATH data."""
	def __init__(self, 
		dataset_dir="/data/uniparc",
		uri="ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/uniparc/uniparc_active.fasta.gz",
		seq_filename="sequences.txt",
		tokenizer_prefix='uniparc_tokenizer',
		vocab_size=8000,
		max_n_sequences=int(4e8),
		train_test_val_split=[.8, .1, .1],
		no_verification=False
		):
		self.dataset_dir = dataset_dir
		self.seq_file = os.path.join(dataset_dir, seq_filename)
		self.train_seq_file = os.path.join(dataset_dir, 'train.txt')
		self.test_seq_file = os.path.join(dataset_dir, 'test.txt')
		self.val_seq_file = os.path.join(dataset_dir, 'val.txt')
		self.dataset_metadata_file = os.path.join(dataset_dir, 'dataset_metadata.pkl')
		self.uri = uri
		self.zipped_dataset_file = os.path.join(dataset_dir, os.path.basename(uri))
		self.unzipped_dataset_file = os.path.splitext(self.zipped_dataset_file)[0]
		self.tokenizer_file_prefix = os.path.join(dataset_dir, tokenizer_prefix)
		self.vocab_size = vocab_size
		self.max_n_sequences = max_n_sequences
		self.train_test_val_split = train_test_val_split
		self.no_verification = no_verification

	def prepare_dataset(self):
		"""Download, unzip, and preprocess sequences, collecting aggregate statistics about the 
		dataset along the way"""

		if self.no_verification:
			print("Performing all steps of dataset generation without user verification. \
			 	This could take a full day.")
		else:
			print("Confirming with user at each significant dataset generation step.")
		self.download_latest()
		self.unzip_data()
		self.prepare_unsupervised_sequences()
		self.split_train_test_val()
		self.train_tokenizer()

	def download(self, uri):
		"""Download the dataset to the given file path
		Args:
			uri (string): the uri of the data to download
		"""
		# Make dataset location 
		# TODO: Make this faster - this call is surprisingly slow due to call to os.lstat
		if not os.path.exists(self.dataset_dir):
			os.makedirs(self.dataset_dir)

			if not self.no_verification:
				response = input("About to download sequence data. This could take several hours. Continue? [Y/n]")
				if response != 'Y':
					raise ValueError("Data unavailable")
			else:
				print("About to download sequence data. This could take several hours.")
			# Download dataset
			cmd = f"wget -P {self.dataset_dir} -m {self.uri}"
			process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
			output, error = process.communicate()
			print("Process output:", output, "Process error:", error)

	def download_latest(self):
		"""Download the latest version of the UniParc dataset"""
		self.download(self.uri)

	def unzip_data(self):
		"""Unzip the data"""
		if not os.path.exists(self.unzipped_dataset_file):

			if not self.no_verification:
				response = input("About to unzip sequence data. This could take several hours. Continue? [Y/n]")
				if response != 'Y':
					raise ValueError("Data unavailable")
			else:
				print("About to unzip sequence data. This could take several hours.")
				
			# Move gzip file to the dataset_dir
			zipfile_path = self.uri.split('/')[2:]
			zipfile_path = os.path.join(self.dataset_dir, *zipfile_path)
			cmd = f"mv {zipfile_path} {self.dataset_dir}"
			process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)

			# Remove empty file path
			empty_zipfile_dir = os.path.join(self.dataset_dir, self.uri.split('/')[2])
			cmd = f"rm -rf {empty_zipfile_dir}"
			process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
			
			# Unzip file
			cmd = f"gunzip {self.zipped_dataset_file}"
			process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)

	def prepare_unsupervised_sequences(self):
		"""Extract sequences into a single file with a sequence on each line, ready
		to be tokenized by the SentencePiece tokenizer. Also, collect aggregate
		data for sanity check on data integrity."""
		n_seq = 0
		seq_lengths = np.zeros(self.max_n_sequences)
		if not os.path.exists(self.seq_file):
			
			if not self.no_verification:
				response = input("About to preprocess sequence data. This could take several hours. Continue? [Y/n]")
				if response != 'Y':
					raise ValueError("Data unavailable")
			else:
				print("About to preprocess sequence data. This could take several hours.")
			
			with open(self.unzipped_dataset_file, 'r') as data_file:
				with open(self.seq_file, 'w') as seq_file:
					# Make regexes for the first line and for a sequence
					first_line_regex = re.compile(r'\>UPI[0-9]*.*\n')
					sequence_regex = re.compile(r'\>UPI[0-9]*.*([^\>]*)')
					raw_seq = ""
					# Skip the first line
					next(data_file)
					for line in data_file:
						# See if we have come to a new sequence
						if bool(re.search(first_line_regex, line)):
							# Flush the current sequence buffer and reset it
							seq_file.write(raw_seq + '\n')
							# Increment n_seq counter and track line length
							seq_lengths[n_seq] = len(raw_seq)
							n_seq += 1
							raw_seq = ""
						else:
							raw_seq += line.strip('\n')

					# Flush the current sequence buffer one last time
					seq_file.write(raw_seq)
					seq_lengths[n_seq] = len(raw_seq)
					n_seq += 1

			# Log aggregate data for sanity check
			self.log_sequence_data(seq_lengths[:n_seq])
	
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

	def split_train_test_val(self):
		"""Split the dataset betweet training, test, and validation sets"""
		if not os.path.exists(self.train_seq_file):

			if not self.no_verification:
				response = input("About to create train/test/validation files. This could take several hours. Continue? [Y/n]")
				if response != 'Y':
					raise ValueError("Train/test/val split doesn't exist")
			else:
				print("About to create train/test/validation files. This could take several hours.")

			# Find total number of sequences
			with open(self.dataset_metadata_file, 'rb') as metadata_file:
				metadata = pkl.load(metadata_file)
			n_seq = metadata.n_seq
			indices = np.arange(n_seq)
			# Get train indices
			n_train = int( self.train_test_val_split[0] * n_seq )
			train_indices = np.random.choice(indices, size=n_train, replace=False)
			train_indices_set = set(train_indices)
			remaining_indices = np.setxor1d(indices, train_indices)
			# Get test indices
			n_test = int( self.train_test_val_split[1] * n_seq )
			test_indices = np.random.choice(remaining_indices, size=n_test, replace=False)
			test_indices_set = set(test_indices)
			# Get val indices
			remaining_indices = np.setxor1d(remaining_indices, test_indices)
			val_indices = remaining_indices
			val_indices_set = set(val_indices)
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
								if i in train_indices_set:
									train_file.write(line)
								elif i in test_indices_set:
									test_file.write(line)
								elif i in val_indices_set:
									val_file.write(line)
								else:
									raise ValueError("All sequences should be assigned to a train/test/val set")				


	def train_tokenizer(self):
		"""Train a tokenizer on the training data if a trained tokenizer doesn't exist"""
		if not os.path.exists(self.tokenizer_file_prefix + '.model'):
			
			if not self.no_verification:
				response = input("About to train tokenizer. This could take several hours. Continue? [Y/n]")
				if response != 'Y':
					raise ValueError("Tokenizer unavailable")
			print("About to train tokenizer. This could take several hours.")
			
			spm.SentencePieceTrainer.train(
				input=self.train_seq_file, 
				model_prefix=self.tokenizer_file_prefix, 
				vocab_size=self.vocab_size, 
				hard_vocab_limit=False,
				add_dummy_prefix=False,
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