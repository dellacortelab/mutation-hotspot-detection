#######################################################################################################
# Neural network model classes for the ActivityPredictor
#######################################################################################################

import os
import numpy as np
import torch
from torch import nn
from torch.nn import init
from transformers import BertModel, BertConfig

from .dataset.mutation_activity_dataset import MutationActivityDataset

class FullyConnectedActivityPredictor(nn.Module):
    """An architecture suited to finding patterns based on positions in a fixed-length vector
    """
    def __init__(self, d_model=768, vocab_size=8000, seq_length=200, set_deterministic_weights=False, from_pretrained=False, model_name='fc_activity_predictor', base_savepath='/models/mutation_activity/activity_predictor', dataset_dir=None, amino_acids=np.array(list('AILMFWYVCGPRNDQEHKST')), base_seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI'))):
    """Generate a hotspot detection dataset, train a network to identify hotspots, and create plots summarizing the result.
    Args:
        d_model (int): the dimension of hidden layers
        vocab_size (int): the vocabulary size (including <start>, <end>, <unk>, and <pad> tokens)
        seq_length (int): the length of the sequence to predict hotspots onnt_activity_predictor
        set_deterministic_weights (bool): whether or not to set optimal weights that we found analytically
        from_pretrained (bool): whether to use weights from pretrained model
        model_name (str): the model name for logged data
        base_savepath (str): the path to the folder for saved models
        dataset_dir (str): the path to the folder for the generated dataset
        amino_acids (np.array of type str): the vocabulary of amino acids
        base_seq (np.array of type str): the base sequence to mutate
    """
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.lin_1 = nn.Linear(d_model, 1, bias=False)
        self.act = nn.Sigmoid()
        self.weight_layer = nn.Parameter(data=torch.zeros(seq_length), requires_grad=True)
        self.bias_layer = nn.Parameter(data=torch.zeros(seq_length), requires_grad=True)
        
        #
        # Initialize weights
        #

        # Create a dataset object to get the tokenized indices of the characters
        dataset = MutationActivityDataset(mode='train', no_verification=True, dataset_dir=dataset_dir)
        self.base_seq = nn.Parameter(data=dataset.tokenize(''.join(list(base_seq))), requires_grad=False)
        # Re-order such that the hydrophobics and hydrophilics are grouped together.
        # [1] because the tokenizer adds start/stop tokens
        hydrophobic_tokenized_indices = torch.tensor([dataset.tokenize(character)[1] for character in amino_acids[:11]])
        hydrophilic_tokenized_indices = torch.tensor([dataset.tokenize(character)[1] for character in amino_acids[11:]])
        # Hydrophobics are positive, hydrophilics are negative
        with torch.no_grad():
            self.embedding.weight[hydrophilic_tokenized_indices] = -torch.ones(9, d_model)
            self.embedding.weight[hydrophobic_tokenized_indices] = torch.ones(11, d_model)

        # An option for the perfect weights that we determined analytically to have zero loss
        if set_deterministic_weights:
            self.lin_1.weight[:] = torch.ones(1, d_model)*6
            
            true_indices = np.load(os.path.join(dataset_dir, 'good_bad_flex_indices.npz'))
            true_ids = np.zeros(seq_length)
            # +1 because there is a start token
            true_ids = np.zeros(seq_length)
            true_ids[true_indices['flexible'] + 1] = 1
            true_ids[true_indices['beneficial'] + 1] = 2
            true_ids[true_indices['detrimental'] + 1] = 3
            true_ids = torch.tensor(true_ids)
            with torch.no_grad():
                # Set flexible indices to -1.2, beneficial and detrimental indices to 0.
                self.weight_layer.data[true_ids == 0] = 0.
                self.weight_layer.data[true_ids == 1] = -1.2
                self.weight_layer.data[true_ids == 2] = 0.
                self.weight_layer.data[true_ids == 3] = 0.
                # Set flexible and beneficial indices to .6, detrimental indices to -.6
                self.bias_layer.data[true_ids == 0] = 0.
                self.bias_layer.data[true_ids == 1] = 0.6
                self.bias_layer.data[true_ids == 2] = 0.6
                self.bias_layer.data[true_ids == 3] = -0.6

        if from_pretrained:
            activity_predictor_path = os.path.join(base_savepath, model_name + '.pt')
            if os.path.exists(activity_predictor_path):
                self.activity_predictor = torch.load(activity_predictor_path)
            else:
                print("Pretrained was set to True, but no pretrained activity network was found. \
                    Training activity predictor from scratch.")

    def forward(self, x, return_attention_weights=False):
        """Run a forward pass that outputs predicted activity and 
        reconstructed input.
        Args:
            x ((N x L) torch.Tensor): an item from the training set
            return_attention_weights (bool): whether to return the tensor weights
                with the network output
        """
        out = self.embedding(x)
        out = self.lin_1(out).squeeze(-1)
        out = self.act(out)
        out = out * self.weight_layer + self.bias_layer
        # Mask out values that didn't change from the base_seq - we don't think these should contribute
        mask = self.base_seq == x
        out[mask] = 0

        out = torch.sum(out, dim=-1)

        if return_attention_weights:
            return out, self.weight_layer.data

        return out

