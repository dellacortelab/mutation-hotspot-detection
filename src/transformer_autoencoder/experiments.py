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

def predict_activity(scores, n_beneficial=10, n_detrimental=10, n_flexible=30):
    """Return the predicted beneficial residues, detrimential residues, and flexible residues
    Args:
        scores ((seq_length x n_amino_acids) np.ndarray): the predicted scores with each mutation
            for each amino acid
    Returns:
        predicted_beneficial ((seq_length) np.ndarray): the predicted beneficial indices
        predicted_detrimental ((seq_length) np.ndarray): the predicted detrimental indices
        predicted_flexible ((seq_length) np.ndarray): the predicted flexible indices
    """
    # Predict beneficial locations by top and bottom means
    means = np.mean(scores, axis=1)
    # Sorted means, lowest to highest
    sorted_means = np.argsort(means)
    predicted_beneficial = sorted_means[-n_beneficial:]
    predicted_detrimental = sorted_means[:n_detrimental]
    # Predict flexible locations by variance across amino acids
    variances = np.var(scores, axis=1)
    # Sorted variances, highest to lowest
    sorted_variances = np.argsort(variances)[::-1]
    predicted_flexible = np.array([i for i in sorted_variances if i not in set(predicted_beneficial.tolist() + predicted_detrimental.tolist())])[:n_flexible]
    
    return predicted_beneficial, predicted_detrimental, predicted_flexible

def plot_hotspots(pred_good, pred_bad, pred_flexible, dataset_dir, log_dir):
    fig, ax = plt.subplots(1,1,figsize=(20,5))

    # Get true indices
    true_indices = np.load(os.path.join(dataset_dir, 'good_bad_flex_indices.npz'))
    true_good = true_indices['beneficial']
    true_bad = true_indices['detrimental']
    true_flexible = true_indices['flexible']
    # import pdb; pdb.set_trace()

    for i in range(200):
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
    plt.plot([0,200],[1,1],color='black')
    #ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks([0.5, 1.5], ['Hidden Pattern', 'Predicted Pattern'], fontsize=16)
    plt.xticks(fontsize=22)
    plt.xlim([0,200])
    plt.ylim([0,2])

    plt.savefig(os.path.join(log_dir, 'hotspots.png'))


def get_summary_plots(model, device, log_dir, dataset_dir):
    # Score all residue substitutions
    scores = get_predictions(model=model, device=device, dataset_dir=dataset_dir)

    # Plot 1: means
    means = np.mean(scores, axis=1)
    plt.figure()
    plt.bar(np.arange(len(means)), means)
    plt.title("Mean score per residue mutation")
    plt.ylabel('Mean Activity Score')
    plt.xlabel('Residue Position')
    plt.savefig(os.path.join(log_dir, 'means.png'))

    # Plot 2: Predicted beneficial/detrimental/flexible indices vs. reality
    pred_good, pred_bad, pred_flexible = predict_activity(scores)
    plot_hotspots(pred_good=pred_good, pred_bad=pred_bad, pred_flexible=pred_flexible, dataset_dir=dataset_dir, log_dir=log_dir)