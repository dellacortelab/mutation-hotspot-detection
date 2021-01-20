#############################################
# Driver for hotspot detection experiments
#
# Dataset
# 100k amino acid sequences of length 200
# Each is the same sequence as a wild type, specified by `base_seq`, but with 30 mutations.
# The enzyme activity is based on the sequence.
# There are 50 "hotspot" locations where mutations affect enzyme activity - 10 beneficial,
# 10 detrimental, 30 flexible. For beneficial locations, any mutation increases activity,
# for detrimental residues, any mutation decreases activity, and for flexible residues,
# hydrophobic amino acid mutations increase activity while hydrophilic amino acid mutations
# decrease activity.

# Hypothesis: 
# 1. We can train a network to map from sequence to activity
# 2. We can identify mutation hotspots and whether they are beneficial, detrimental, or 
# flexible by examining the mean and variance (across amino acids) of the network's 
# predictions when we mutate that residue
#############################################

from hotspot_detector.dataset.mutation_activity_dataset import MutationActivityDataset
from hotspot_detector.loader import get_sequence_loaders
from hotspot_detector.model import FullyConnectedActivityPredictor
from hotspot_detector.train import Trainer
from hotspot_detector.experiments import get_summary_plots

import numpy as np
import torch
from torch.nn import MSELoss

def hotspot_experiment(
        base_savepath='/models/hydrolase_design/transformer_activity_predictor',
        model_name='transformer_activity_predictor',
        dataset_dir='/data/mutation_activity/dataset_short_seq',
        log_dir='/data/mutation_activity/logs_w_bias_untrained',
        n_epochs=20,
        batch_eval_freq=100,
        epoch_eval_freq=1,
        no_verification=True,
        # Hyperparameters
        base_seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI')),
        amino_acids=np.array(list('AILMFWYVCGPRNDQEHKST')),
        n_mutations=30,
        d_model=100,
        batch_size=32,
        n_seq=int(1e5),
        simple_data=True,
        debug=False
    ):
    """Generate a hotspot detection dataset, train a network to identify hotspots, and create plots summarizing the result.
    Args:
        base_savepath (str): the path to the folder for saved models
        model_name (str): the model name for logged data
        dataset_dir (str): the path to the folder for the generated dataset
        log_dir (str): the path to the folder for the generated summary files
        n_epochs (int): the number of epochs to train
        batch_eval_freq (int): get validation loss every batch_eval_freq batches
        epoch_eval_freq (int): get validation loss every epoch_eval_freq epochs
        no_verification (bool): whether to ask for verification at each dataset generation step (for large datasets)
        base_seq (np.array of type str): the base sequence to mutate
        amino_acids (np.array of type str): the vocabulary of amino acids
        n_mutations (int): the number of mutations in each sequence in the generated dataset
        d_model (int): the dimension of hidden layers
        batch_size (int): the batch size for training
        n_seq (int): the number of sequences for the generated dataset
        simple_data (bool): True for deterministic labels, False for randomized labels
        debug (bool): whether to run in debug mode, with fewer, shorter sequences and faster training
    """
    # TODO: Check for adequate randomization - val_loader vs iterating through the dataset is currently the same
    
    # Debug setting for shorter sequences, training, and logging
    if debug:
        simple_data = True
        n_seq = int(1e2)
        log_dir = '/data/mutation_activity/logs_mini'
        dataset_dir = '/data/mutation_activity/dataset_mini'
        n_epochs = 1
        base_seq = base_seq[:75]

    # vocab_size includes <start>, <end>, <unk>, and <pad> tokens
    vocab_size = len(amino_acids) + 4
    
    # Set seeds
    np.random.seed(0)
    torch.manual_seed(0)
    # TODO: CUDA seed?

    # Load data into dataloaders
    train_loader, val_loader, test_loader = get_sequence_loaders(dataset_class=MutationActivityDataset, dataset_dir=dataset_dir, batch_size=batch_size, simple_data=simple_data, no_verification=no_verification, vocab_size=vocab_size, n_mutations=n_mutations, n_seq=n_seq, base_seq=base_seq, amino_acids=amino_acids)
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Setting objective")
    objective = MSELoss()
    print("Building model")
    # Load network
    model = FullyConnectedActivityPredictor(vocab_size=vocab_size, d_model=d_model, seq_length=len(base_seq) + 2, amino_acids=amino_acids, base_seq=base_seq, dataset_dir=dataset_dir)
    # print("Building trainer")
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, objective=objective, batch_size=batch_size, log_dir=log_dir, 
        base_savepath=base_savepath, model_name=model_name, device=device, batch_eval_freq=batch_eval_freq, epoch_eval_freq=epoch_eval_freq)
    print("Training model")
    trainer.train(n_epochs=n_epochs)
    print("Training complete") 
    get_summary_plots(model=model, base_seq=base_seq, amino_acids=amino_acids, device=device, log_dir=log_dir, dataset_dir=dataset_dir)
    
    # TODO: Implement data parallelism

if __name__ == "__main__":
    hotspot_experiment()
    



##########################
# Observations: This task relies on position much more than amino acid identity and context. The power of 
# Transformers comes mostly from their ability to combine information across the sequence, whereas this 
# task tests primarily the ability to use positional information (not what Transformers excel at)
#
# Ideas to simplify:
# - ^ = done
# - * = high priority
#
# - Dataset
# - ^No stochasticity in assignment of score. Only hotspot residues contribute to score (and thus the loss), not all residues
# - *Only train one mutation
# - - Mutation at one position - Hydrophilics score 2, hydrophobics score 0, other score 1
# - *Train classification model - classify high, medium, or low activity based on number of beneficial mutations
# - Train on all possible mutations
# - Reduce vocabulary to binary
# - 
# - Training/model
# - *Add learning rate hyperparameter
# - ^Train for longer
# - Substitute attention for Bert pooling
# - ^Implement correct version of attention
# - ^Figure out optimal score vs average score
# - - Optimal score: Predict 2 for hydrophilics, 0 for hydrophobics, 1 for other
# - - Optimal mean: 1 for most residues, 1.6 for beneficial, .4 for detrimental, a little under 1 for flexible
# - - Optimal variance: 0 for most residues, 0 for beneficial, 0 for detrimental, ~.36 for flexible
# - - Optimal loss: 0
# - - Average score: 1
# - - Average score loss: (0^2*6 + 1^2*7 + 1^2*7)/20 = .7
# - 
# - Evaluation
# - ^Get means and variances only for mutated amino acid, not for original
# - - ^Highlight beneficial bars in green, detrimental in red, and flexible in blue
# - - ^Visualize difference only at hotspot residues
# - ^Visualize attention in pooling layer
# - *Visualize distance matrix between embeddings (hopefully hydrophobic will be close to hydrophobic, etc.)
# - Visualize difference between prediction for mutation vs. original
# - Compare predictions for mutations present in the dataset to predictions for mutations not in the dataset
##########################


