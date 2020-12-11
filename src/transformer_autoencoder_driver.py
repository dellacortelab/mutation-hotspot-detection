#######################################################################################################
# Driver for the Transformer AutoEncoder
#######################################################################################################

import torch
from torch import nn

# Local modules
from transformer_autoencoder.train import Trainer
from transformer_autoencoder.loader import get_sequence_loaders
from transformer_autoencoder.model import TransformerAutoEncoder, TransformerAutoEncoderReactivityPredictor
from transformer_autoencoder.loss import AutoEncoderLoss, AutoEncoderFineTuneLoss
from transformer_autoencoder.logger import Logger

def train_transformer_autoencoder(
        batch_size=8,
        n_blocks=110,
        n_epochs=20,
        vocab_size=8000,
        seq_dataset_dir='/data/uniparc',
        no_verification=True
    ):
    print("Building datasets")
    train_loader, val_loader, test_loader = get_sequence_loaders(batch_size=batch_size, vocab_size=vocab_size, dataset_dir=seq_dataset_dir, no_verification=no_verification)
    # If the dataset is too small, vocab_size will be smaller than specified
    true_vocab_size = train_loader.dataset.get_vocab_size()
    print("Datasets built")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Setting objective")
    objective = AutoEncoderLoss(n_classes=true_vocab_size)
    print("Building model")
    # Continue HERE
    model = TransformerAutoEncoder(vocab_size=true_vocab_size)
    print("Building trainer")
    trainer = Trainer(model, train_loader, val_loader, objective, model_name='transformer_autoencoder', device=device)
    print("Training model")
    trainer.train(n_epochs=n_epochs)
    print("Training complete")

def finetune_transformer_autoencoder(
        batch_size=8,
        n_blocks=110,
        n_epochs=20,
        vocab_size=8000,
        seq_dataset_dir='/data/uniparc',
        reactivity_dataset_dir='/data/reactivity',
        no_verification=True
    ):
    print("Building datasets")
    train_loader, val_loader, test_loader  = get_sequence_loaders(batch_size=batch_size, vocab_size=vocab_size, seq_dataset_dir=dataset_dir, no_verification=no_verification)
    print("Datasets built")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Setting objective")
    objective = AutoEncoderFineTuneLoss()
    print("Building model")
    model = TransformerAutoEncoderReactivityPredictor(vocab_size=vocab_size)
    print("Building trainer")
    trainer = Trainer(model, train_loader, val_loader, objective, model_name='transformer_autoencoder_fine_tune', device=device)
    print("Training model")
    trainer.train(n_epochs=n_epochs)
    print("Training complete")

if __name__ == '__main__':
    pretrain_transformer()
