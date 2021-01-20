#######################################################################################################
# Utility function to get dataloaders for sequence datasets
#######################################################################################################

import numpy as np
from torch.utils.data import DataLoader

from .dataset.seq_dataset import ShuffleDataset 

# To calculate a good buffer size for protein sequences:
# 1. For A-Z letters, it is 1 byte/letter
# 2. To get buffer_size, solve this equation for n_seq:
# 3. available_memory * desired_fraction_of_memory_per_dataset = avg_seq_length * bytes_per_char * n_seq
# 4. For example, 60e9 * 1/4 = 500 * 1 * n_seq -> n_seq = 30000000 (30 million sequences)
# 5. Let buffer_size=n_seq

def get_sequence_loaders(dataset_class, batch_size=32, vocab_size=8000, n_mutations=10, n_seq=None, dataset_dir='/data/uniparc', buffer_size=int(3e7), simple_data=True, no_verification=False, base_seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI')), amino_acids=np.array(list('ACDEFGHIKLMNPQRSTVWY'))):
    """Return dataloaders for the train, validation, and test datasets. These dataloaders implement
    shuffling to a limited extent, based on memory capacity.
    Args:
        dataset_class (type): a dataset_class type which will be instantiated
        batch_size (int): the batch size for training
        vocab_size (int): the vocabulary size (including <start>, <end>, <unk>, and <pad> tokens)
        n_mutations (int): the number of mutations in each sequence in the generated dataset
        n_seq (int): the number of sequences for the generated dataset
        dataset_dir (str): the path to the folder for the generated dataset
        buffer_size (int): the number of sequences to maintain in the ShuffleDataset buffer. Ideally,
            this should be as large as possible without giving you a memory error. See the heuristic 
            above for a suggested number.
        simple_data (bool): True for deterministic labels, False for randomized labels
        no_verification (bool): whether to ask for verification at each dataset generation step (for large datasets)
        base_seq (np.array of type str): the base sequence to mutate
        amino_acids (np.array of type str): the vocabulary of amino acids
    Returns:
        train_loader (DataLoader): a dataloader containing the training data
        val_loader (DataLoader): a dataloader containing the training data
        test_loader (DataLoader): a dataloader containing the training data
    """
    train_dataset = dataset_class(mode='train', vocab_size=vocab_size, base_seq=base_seq, amino_acids=amino_acids, n_mutations=n_mutations, simple_data=simple_data, no_verification=no_verification, dataset_dir=dataset_dir, n_seq=n_seq)
    val_dataset = dataset_class(mode='val', vocab_size=vocab_size, dataset_dir=dataset_dir)
    test_dataset = dataset_class(mode='test', vocab_size=vocab_size, dataset_dir=dataset_dir)
    # Create "Shuffle Datasets", which which allow for random sampling to the greatest possible extent
    # within your memory constraints. This is because you can't shuffle an IterableDataset
    train_dataset = ShuffleDataset(train_dataset, buffer_size=buffer_size)
    val_dataset = ShuffleDataset(val_dataset, buffer_size=buffer_size)
    test_dataset = ShuffleDataset(test_dataset, buffer_size=buffer_size)
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=padding_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=padding_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=padding_collate_fn
    )
    return train_loader, val_loader, test_loader
