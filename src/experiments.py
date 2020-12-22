#############################################
# Experiments
#############################################

from transformer_autoencoder.dataset.mutation_activity_dataset import MutationActivityDataset
from transformer_autoencoder.loader import get_sequence_loaders
from transformer_autoencoder.model import TransformerReactivityPredictor
from transformer_autoencoder.train import Trainer

import torch
from torch.nn import MSELoss

def hotspot_experiment(
        base_savepath='/models/hydrolase_design/transformer_reactivity_predictor',
        model_name='transformer_reactivity_predictor',
        dataset_dir='/data/mutation_activity',
        vocab_size=200,
        d_model=768,
        n_epochs=10,
        no_verification=True
    ):
    # Dataset
    # 100k amino acid sequences of length 100k amino acid sequences of length 200
    # Each is the same sequence as a wild type, but with 10 mutations
    # The enzyme activity is based on the sequence
    # There are 50 "hotspot" locations where mutations affect enzyme activity - 10 beneficial,
    # 10 detrimental, 30 neutral
    # Each mutated sequence is assigned an activity based on whether its mutations
    # occur at hotspots

    # Hypothesis: 
    # 1. We can train a network to map from sequence to activity
    # 2. We can identify mutation hotspots and whether they are beneficial, detrimental, or 
    # neutral by examining the variance of the network's predictions when we mutate that residue
    # compared to the variance of mutating other residues
    
    # Load many_to_one dataset into dataset class
    train_loader, val_loader, test_loader = get_sequence_loaders(dataset_class=MutationActivityDataset, dataset_dir=dataset_dir, no_verification=no_verification, vocab_size=vocab_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Setting objective")
    objective = MSELoss()
    print("Building model")
    # Load network
    model = TransformerReactivityPredictor(vocab_size=vocab_size, d_model=d_model)
    print("Building trainer")
    # TODO: specify model path
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, objective=objective, 
        base_savepath=base_savepath, model_name=model_name, device=device)
    print("Training model")
    trainer.train(n_epochs=n_epochs)
    print("Training complete")
    # Results
    # Plot loss over time
    # Plot activity variance across residues
    # Plot mean across residues with high variance
    # Assign designation: beneficial to top 10 beneficial positions, designation: detrimental 
    # to top 10 detrimental positions, and designation: flexible to top 30 variance positions not
    # in beneficial/detrimental
    # Compare 

if __name__ == "__main__":
    hotspot_experiment()