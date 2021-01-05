###########################################################
# Functions to get result visualizations
# - Main function: get_summary_plots()
###########################################################

import copy
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle as box
import numpy as np
from .dataset.mutation_activity_dataset import MutationActivityDataset
import os

def get_predictions(
        model, 
        device,
        dataset_dir,
        base_seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI')),
        amino_acids="ACDEFGHIKLMNPQRSTVWY",
    ):
    """Mutate every residue in the model to every other amino acid. Get the mean and
    variance of the predictions for each residue position. We expect that flexible
    residues will have high variance across different amino acids (wrong - it should 
    predict 0 for all of them), beneficial mutations will have the highest predictions,
    and detrimental mutations will have the lowest predictions.
    Args:
        model (nn.Module): a torch model
        device (torch.device): the device on which to test the model
        dataset_dir (str): the directory in which the dataset is stored
        base_seq (np.ndarray): an array of characters in the base sequence
        amino_acids (str): a string containg the 1-letter abbreviations of all amino acids
    Returns:
        predicted_beneficial ((seq_length) np.ndarray): the predicted beneficial indices
        predicted_detrimental ((seq_length) np.ndarray): the predicted detrimental indices
        predicted_flexible ((seq_length) np.ndarray): the predicted flexible indices
    """
    # Create a dataset object to preprocess the sequences
    dataset = MutationActivityDataset(mode='train', no_verification=True, dataset_dir=dataset_dir, n_seq=100)

    model.to(device)

    scores = np.zeros((len(base_seq), len(amino_acids)))
    for i in range(len(base_seq)):
        for j, amino_acid in enumerate(amino_acids):
            mut_sequence = copy.deepcopy(base_seq)
            mut_sequence[i] = amino_acid
            mut_sequence = ''.join(mut_sequence)
            mut_seq_preproc = dataset.preprocess_seq(mut_sequence).to(device)
            scores[i, j] = model(mut_seq_preproc.unsqueeze(0)).cpu().item()

    return scores

def get_mean_var_for_mutations(
        scores,
        base_seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI')),
        amino_acids="ACDEFGHIKLMNPQRSTVWY"
    ):
    """Get the mean and variance of scores across residues not equal to the original residue
    Args:
        scores ((seq_length x n_amino_acids) np.ndarray): the predicted scores with each mutation
            for each amino acid
        base_seq ((seq_length) np.ndarray): an array of characters in the base sequence
    Returns:
        ((seq_length) np.ndarray): the average predicted scores for mutated residues
        ((seq_length) np.ndarray): the variance of predicted scores for mutated residues
    """
    # Scores without score of non-mutated amino acid
    abbreviated_scores = np.zeros( (len(base_seq), len(amino_acids)-1) )
    amino_acid_to_idx = { amino_acid : idx for idx, amino_acid in enumerate(amino_acids)}

    for i, (row, true_amino_acid) in enumerate(zip(scores, base_seq)):
        true_amino_acid_idx = amino_acid_to_idx[true_amino_acid]
        # Exclude the score for the true amino acid
        abbreviated_scores[i, :true_amino_acid_idx] = row[:true_amino_acid_idx]
        abbreviated_scores[i, true_amino_acid_idx:] = row[true_amino_acid_idx+1:]

    return np.mean(abbreviated_scores, axis=1), np.var(abbreviated_scores, axis=1)


def predict_activity(scores, means, variances, n_beneficial=10, n_detrimental=10, n_flexible=30):
    """Return the predicted beneficial residues, detrimental residues, and flexible residues
    Args:
        scores ((seq_length x n_amino_acids) np.ndarray): the predicted scores with each mutation
            for each amino acid
        n_beneficial (int): the number of beneficial residues
        n_detrimental (int): the number of detrimental residues
        n_flexible (int): the number of flexible residues
    Returns:
        predicted_beneficial ((seq_length) np.ndarray): the predicted beneficial indices
        predicted_detrimental ((seq_length) np.ndarray): the predicted detrimental indices
        predicted_flexible ((seq_length) np.ndarray): the predicted flexible indices
    """
    # Predict beneficial locations by top and bottom means
    # Sorted means, lowest to highest
    sorted_means = np.argsort(means)
    predicted_beneficial = sorted_means[-n_beneficial:]
    predicted_detrimental = sorted_means[:n_detrimental]
    # Sorted variances, highest to lowest
    sorted_variances = np.argsort(variances)[::-1]
    predicted_flexible = np.array([i for i in sorted_variances if i not in set(predicted_beneficial.tolist() + predicted_detrimental.tolist())])[:n_flexible]
    
    return predicted_beneficial, predicted_detrimental, predicted_flexible

def get_colors(
        dataset_dir,
        base_seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI'))
    ):
    """Return a list of colors indicating the true nature of each residue (green
    for beneficial, red for detrimental, blue for flexible, white for other)
    Args:
        dataset_dir (str): the directory in which the dataset is stored
        base_seq ((seq_length) np.ndarray): an array of characters in the base sequence
    Returns:
        colors (list of str): colors representing the type of each residue
    """
    # Get true indices
    true_indices = np.load(os.path.join(dataset_dir, 'good_bad_flex_indices.npz'))
    true_good = true_indices['beneficial']
    true_bad = true_indices['detrimental']
    true_flexible = true_indices['flexible']
    # Create a list of colors, in order
    colors = []
    for i in range(len(base_seq)):
        if i in true_good:
            colors.append('green')
        elif i in true_bad:
            colors.append('red')
        elif i in true_flexible:
            colors.append('lightblue')
        else:
            colors.append('white')

    return colors

def plot_hotspots(base_seq, pred_good, pred_bad, pred_flexible, dataset_dir, log_dir):
    """Plot predicted hotspots on top and true hotspots on bottom
    Args:
        predicted_beneficial ((seq_length) np.ndarray): the predicted beneficial indices
        predicted_detrimental ((seq_length) np.ndarray): the predicted detrimental indices
        predicted_flexible ((seq_length) np.ndarray): the predicted flexible indices
        log_dir (str): the directory in which to save the results
        dataset_dir (str): the directory in which the dataset is stored
    """
    fig, ax = plt.subplots(1,1,figsize=(30,5))

    # Get true indices
    true_indices = np.load(os.path.join(dataset_dir, 'good_bad_flex_indices.npz'))
    true_good = true_indices['beneficial']
    true_bad = true_indices['detrimental']
    true_flexible = true_indices['flexible']

    for i in range(len(base_seq)):
        if i in true_good:
            color = 'green'
        elif i in true_bad:
            color = 'red'
        elif i in true_flexible:
            color = 'lightblue'
        else:
            color = 'white'
        a = box((i,0),1,1,color = color)
        ax.add_patch(a)
        if i in pred_good:
            color = 'green'
        elif i in pred_bad:
            color = 'red'
        elif i in pred_flexible:
            color = 'lightblue'
        else:
            color = 'white'
        b = box((i,1),1,1,color = color)
        ax.add_patch(b)
    plt.plot([0,len(base_seq)],[1,1],color='black')
    #ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks([0.5, 1.5], ['Hidden Pattern', 'Predicted Pattern'], fontsize=16)
    plt.xticks(fontsize=22)
    plt.xlim([0,len(base_seq)])
    plt.ylim([0,2])

    plt.savefig(os.path.join(log_dir, 'hotspots.png'))

def plot_mean_var(scores, means, variances, colors, log_dir):
    """Plot the means and variances for each residue
    Args:
        scores ((seq_length x n_amino_acids) np.ndarray): the predicted scores with each mutation
            for each amino acid
        colors (list of str): colors representing the type of each residue
        log_dir (str): the directory in which to save the results
    """
    fig = plt.figure(figsize=(30, 10))
    plt.subplot(211)
    plt.bar(np.arange(len(means)), means, color=colors, edgecolor='black')
    plt.title("Mean score per residue mutation")
    plt.ylabel('Mean Activity Score')
    plt.xlabel('Residue Position')
    plt.subplot(212)
    plt.bar(np.arange(len(variances)), variances, color=colors, edgecolor='black')
    plt.title("Variance Across Amino Acids for each Residue")
    plt.ylabel('Variance of Activity Score')
    plt.xlabel('Residue Position')
    plt.savefig(os.path.join(log_dir, 'mean_variance.png'))

def get_average_attention(
        model, 
        device, 
        dataset_dir, 
        n_seq=100,
        base_seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI'))
        ):
    """Get the average attention weights of the last layer in the model, across N batches
    from the dataset, for each residue in the sequence. If the model has trained correctly, 
    we expect it to attend most heavily to the mutation hotspots, which are the residues
    that make the most difference in enzyme activity.
    Args:
        model (nn.Module): a pytorch model
        device (torch.device): the device on which to test the model
        dataset_dir (str): the directory in which the dataset is stored
        n_seq (int): the number of sequences to average over
    Returns:
        attention_weights_average ((seq_length) np.ndarray): the average attention weights for each residue
    """
    dataset = MutationActivityDataset(mode='train', no_verification=True, dataset_dir=dataset_dir)
    model.to(device)
    
    attention_weights_sum = None
    for i, (x, _) in enumerate(dataset):
        if i > n_seq:
            break

        if attention_weights_sum is None:
            attention_weights_sum = np.zeros(len(x[1:-1]))

        x = x.to(device).unsqueeze(0)
        _, attention_weights = model(x, return_attention_weights=True)
        attention_weights_sum += attention_weights.squeeze(0).detach().cpu().numpy()[1:-1]

    attention_weights_average = attention_weights_sum / n_seq

    return attention_weights_average

def plot_average_attention(attention, colors, log_dir):
    """Plot the attention weights of the last attention layer for each residue.
    Args:
        attention ((seq_length) np.ndarray): the attention weights for each residue
        colors (list of str): colors representing the type of each residue
        log_dir (str): the directory in which to save the results
    """
    fig = plt.figure(figsize=(30, 5))
    plt.bar(np.arange(len(attention)), attention, color=colors, edgecolor='black')
    plt.title("Attention Values per Residue")
    plt.ylabel('Attention Score')
    plt.xlabel('Residue Position')
    plt.savefig(os.path.join(log_dir, 'average_attention.png'))


def get_attention_samples(
        model, 
        device, 
        dataset_dir, 
        n_samples=4,
        base_seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI'))
        ):
    """Get the average attention weights of the last layer in the model, across N batches
    from the dataset, for each residue in the sequence. If the model has trained correctly, 
    we expect it to attend most heavily to the mutation hotspots, which are the residues
    that make the most difference in enzyme activity.
    Args:
        model (nn.Module): a pytorch model
        device (torch.device): the device on which to test the model
        dataset_dir (str): the directory in which the dataset is stored
        n_samples (int): the number of sequences to sample
    Returns:
        sequences (list of (seq_length) str's): the sequences sampled
        mut_indices_list (list of (seq_length) np.ndarray's): a list of ndarrays with 1's where each sequence is mutated
        attention_weights_list (list of (seq_length) np.ndarray's): the attention weights for each sample
    """
    dataset = MutationActivityDataset(mode='train', return_raw_seq=True, no_verification=True, dataset_dir=dataset_dir)
    model.to(device)

    sequences = []
    attention_weights_list = []
    mut_indices_list = []
    for i, (x, _, raw_seq) in enumerate(dataset):
        if i > n_samples:
            break

        # Get indices where this sequence differs from the original sequence
        mut_indices = ( np.array(list(raw_seq)) != base_seq )
        x = x.to(device).unsqueeze(0)
        _, attention_weights = model(x, return_attention_weights=True)
        attention_weights = attention_weights.squeeze(0).detach().cpu().numpy()[1:-1]
        
        sequences.append(raw_seq)
        mut_indices_list.append(mut_indices)
        attention_weights_list.append(attention_weights)

    return sequences, mut_indices_list, attention_weights_list

def plot_attention_samples(sequences, mut_indices_list, attention_samples, colors, log_dir):
    """Plot the attention weights of the last attention layer for each residue.
    Args:
        sequences (list of (seq_length) np.ndarrays): the sequences sampled
        attention (list of (seq_length) np.ndarrays): the attention weights for each residue
        colors (list of str): colors representing the type of each residue
        log_dir (str): the directory in which to save the results
    """
    n_samples = len(attention_samples)

    fig, ax = plt.subplots(n_samples, 1, figsize=(30, 5*n_samples))

    plt.suptitle("Attention Values per Residue")
    for i, (seq, mut_indices, attention) in enumerate(zip(sequences, mut_indices_list, attention_samples)):
        
        ax[i].bar(np.arange(len(attention)), attention, color=colors, edgecolor='black')

        # Set ticks
        # Use the pyplot interface to change just one subplot...
        plt.sca(ax[i])
        plt.xticks(np.arange(len(attention)), list(seq))
        ticklabels = ax[i].get_xticklabels()
        for j, idx_mutated in enumerate(mut_indices):
            if idx_mutated:
                ticklabels[j].set_color("red")
        
        ax[i].set_ylabel('Attention Score')
        ax[i].set_xlabel('Residue Position (ticks for mutated residues in red)')
    
    plt.savefig(os.path.join(log_dir, 'attention_samples.png'))


def get_summary_plots(model, base_seq, amino_acids, device, log_dir, dataset_dir):
    """Save summary plots of mean score for each residue
    Args:
        model (nn.Module): a pytorch model
        device (torch.device): the device on which to test the model
        log_dir (str): the directory in which to save the results
        dataset_dir (str): the directory in which the dataset is stored
    """
    # Score all residue substitutions
    scores = get_predictions(model=model, device=device, dataset_dir=dataset_dir, base_seq=base_seq, amino_acids=amino_acids)
    colors = get_colors(base_seq=base_seq, dataset_dir=dataset_dir)

    # Plot 0: average attention weights
    average_attention = get_average_attention(model=model, device=device, dataset_dir=dataset_dir, base_seq=base_seq)
    plot_average_attention(attention=average_attention, colors=colors, log_dir=log_dir)

    # Plot 0.5: attention weights for a couple of individual sequences
    sequences, mut_indices_list, attention_samples = get_attention_samples(model=model, device=device, dataset_dir=dataset_dir, base_seq=base_seq)
    plot_attention_samples(sequences=sequences, mut_indices_list=mut_indices_list, attention_samples=attention_samples, colors=colors, log_dir=log_dir)

    # Plot 1: means and variances
    means, variances = get_mean_var_for_mutations(scores=scores, base_seq=base_seq, amino_acids=amino_acids)
    plot_mean_var(scores=scores, means=means, variances=variances, colors=colors, log_dir=log_dir)

    # Plot 2: Predicted beneficial/detrimental/flexible indices vs. reality
    pred_good, pred_bad, pred_flexible = predict_activity(scores=scores, means=means, variances=variances)
    plot_hotspots(base_seq=base_seq, pred_good=pred_good, pred_bad=pred_bad, pred_flexible=pred_flexible, dataset_dir=dataset_dir, log_dir=log_dir)