####################################
# Driver for various training setups
####################################

import torch
from torch import nn

# Local modules
from transformer.train import Trainer
from transformer.loader import get_sequence_loaders
from transformer.model import ProteinBert
from common.logger import Logger

def pretrain_transformer(
        batch_size=8,
        n_blocks=110,
        n_epochs=20,
        vocab_size=8000,
        dataset_dir='/data/uniparc',
        no_verification=True
    ):
    print("Building datasets")
    train_loader, test_loader, val_loader = get_sequence_loaders(batch_size=batch_size, vocab_size=vocab_size, dataset_dir=dataset_dir, no_verification=no_verification)
    print("Datasets built")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    objective = nn.CrossEntropyLoss()
    print("Building model")
    model = ProteinBert(vocab_size=vocab_size)
    # breakcode()
    trainer = Trainer(model, train_loader, val_loader, objective, model_name='autoencoder', device=device)

    print("Training model")
    trainer.train(n_epochs=n_epochs)
    print("Training complete")

def finetune_transformer():
    pass

if __name__ == '__main__':
    pretrain_transformer()
