#######################################################################################################
# Dataloaders for the Transformer AutoEncoder
#######################################################################################################

from torch.utils.data import DataLoader
from .dataset.uniparc_dataset import ProtSeqDataset
from .dataset.seq_dataset import ShuffleDataset, padding_collate_fn

# To calculate a good buffer size for protein sequences:
# 1. For A-Z letters, it is 1 byte/letter
# 2. To get buffer_size, solve this equation for n_seq.
# 3. available_memory * desired_fraction_of_memory_per_dataset = avg_seq_length * bytes_per_char * n_seq
# 4. For example, 60e9 * 1/4 = 500 * 1 * n_seq -> n_seq = 30000000 (30 million sequences)
# 5. Let buffer_size=n_seq

def get_sequence_loaders(dataset_class=ProtSeqDataset, batch_size=32, vocab_size=8000, dataset_dir='/data/uniparc', buffer_size=int(3e7), no_verification=False):
    train_dataset = dataset_class(mode='train', vocab_size=vocab_size, no_verification=no_verification, dataset_dir=dataset_dir)
    val_dataset = dataset_class(mode='val', vocab_size=vocab_size, dataset_dir=dataset_dir)
    test_dataset = dataset_class(mode='test', vocab_size=vocab_size, dataset_dir=dataset_dir)
    # Create "Shuffle Datasets", which which allow for random sampling to the greatest possible extent
    # within your memory constraints. This is because you can't shuffle an IterableDataset
    # import pdb; pdb.set_trace()
    train_dataset = ShuffleDataset(train_dataset, buffer_size=buffer_size)
    val_dataset = ShuffleDataset(val_dataset, buffer_size=buffer_size)
    test_dataset = ShuffleDataset(test_dataset, buffer_size=buffer_size)
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # num_workers=16,
        pin_memory=True,
        collate_fn=padding_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        # num_workers=16,
        pin_memory=True,
        collate_fn=padding_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        # num_workers=16,
        pin_memory=True,
        collate_fn=padding_collate_fn
    )
    return train_loader, val_loader, test_loader
