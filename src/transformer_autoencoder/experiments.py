#############################################
# Experiments
#############################################

from dataset.mutation_activity_dataset import MutationActivityDataset

def hotspot_experiment():
    # Dataset
    # 100k amino acid sequences of lengths
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
    train_loader, val_loader, test_loader = get_sequence_loaders(dataset_class=MutationActivityDataset)
    # Load network
    
    # Train network

    # Results
    # Plot loss over time
    # Plot activity variance across residues
    # Plot mean across residues with high variance
    # Assign designation: beneficial to top 10 beneficial positions, designation: detrimental 
    # to top 10 detrimental positions, and designation: flexible to top 30 variance positions not
    # in beneficial/detrimental
    # Compare 
