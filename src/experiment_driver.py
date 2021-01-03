#############################################
# Experiments
#############################################

from transformer_autoencoder.dataset.mutation_activity_dataset import MutationActivityDataset
from transformer_autoencoder.loader import get_sequence_loaders
from transformer_autoencoder.model import TransformerActivityPredictor
from transformer_autoencoder.train import Trainer
from transformer_autoencoder.experiments import get_summary_plots

import torch
from torch.nn import MSELoss




def hotspot_experiment(
        base_savepath='/models/hydrolase_design/transformer_activity_predictor',
        model_name='transformer_activity_predictor',
        # dataset_dir='/data/mutation_activity/dataset_mini',
        dataset_dir='/data/mutation_activity/dataset',
        # log_dir='/data/mutation_activity/logs_mini',
        log_dir='/data/mutation_activity/logs',
        n_epochs=4,
        batch_eval_freq=100,
        epoch_eval_freq=1,
        no_verification=True,
        # Hyperparameters
        # n_seq=int(1e2)
        vocab_size=20,
        d_model=768,
        batch_size=32,
        n_seq=int(1e4)
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
    
    # Set seeds
    import torch
    import numpy as np
    np.random.seed(0)
    torch.manual_seed(0)


    # Load many_to_one dataset into dataset class
    train_loader, val_loader, test_loader = get_sequence_loaders(dataset_class=MutationActivityDataset, dataset_dir=dataset_dir, batch_size=batch_size, no_verification=no_verification, vocab_size=vocab_size, n_seq=n_seq)
    # device = torch.device('cpu') 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Setting objective")
    objective = MSELoss()
    print("Building model")
    # Load network
    model = TransformerActivityPredictor(vocab_size=vocab_size, d_model=d_model)
    print("Building trainer")
    # TODO: specify model path
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, objective=objective, batch_size=batch_size, log_dir=log_dir, 
        base_savepath=base_savepath, model_name=model_name, device=device, batch_eval_freq=batch_eval_freq, epoch_eval_freq=epoch_eval_freq)
    print("Training model")
    trainer.train(n_epochs=n_epochs)
    print("Training complete") 
    get_summary_plots(model=model, device=device, log_dir=log_dir, dataset_dir=dataset_dir)
    # Results
    # Plot loss over time
    # Plot activity variance across residues
    # Plot mean across residues with high variance
    # Assign designation: beneficial to top 10 beneficial positions, designation: detrimental 
    # to top 10 detrimental positions, and designation: flexible to top 30 variance positions not
    # in beneficial/detrimental
    # Compare 

    # TODO: Analyze results, see if we can identify residues
    # TODO: Implement data parallelism

if __name__ == "__main__":
    hotspot_experiment()