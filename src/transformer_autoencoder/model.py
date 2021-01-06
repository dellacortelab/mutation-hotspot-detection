#######################################################################################################
# Neural network model classes for the Transformer AutoEncoder
#######################################################################################################

import os
import numpy as np
import torch
from torch import nn
from torch.nn import init
from transformers import BertModel, BertConfig

from .dataset.mutation_activity_dataset import MutationActivityDataset

class AttentionLayer(nn.Module):
    """Attention layer implemented as in self-attention, but with a trainable prototype query
    vector instead of a query that is a transformation of the input. Justification: for the 
    purposes of autoencoding and predicting, the prototype vector for summarizing the sequence 
    does not depend on different tokens - it is always has the same job: summarize the sequence 
    for e.g. an autoencoding task (as opposed to the job of predicting the next character, in 
    which the prototype vector changes based on the character preceding the character to predict.)
    """
    def __init__(self, d_model):
        super().__init__()

        # TODO: add multiple heads
        # The query should be a trainable prototype vector of weights, such that multiplying Q by K^T is
        # just multiplying K by a linear layer from d_model to 1
        self.lin_q = nn.Linear(d_model, 1)
        self.lin_k = nn.Linear(d_model, d_model)
        self.lin_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)
        self.scale_factor = torch.sqrt(torch.tensor(d_model, dtype=torch.float))

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
            return_attention_weights (bool): whether to return the tensor weights
                with the network output
        """
        # # Previous attention setup
        # # x: N x L x d_model
        # q = self.lin_q(x)
        # # q: N x L x d_model
        # q = torch.sum(q, dim=1, keepdim=True)
        # # q: N x 1 x d_model
        # k = self.lin_k(x)
        # k = torch.transpose(k, 1, 2)
        # x: N x L x d_model
        k = self.lin_k(x)
        # k: N x L x d_model
        # This is where we differ from self-attention: we use a learnable prototype Q vector, 
        # implemented as a linear layer, instead of transforming the input to get queries
        attn = self.lin_q(k)
        # attn: N x L x 1
        attn = torch.transpose(attn, 1, 2)
        # attn: N x 1 x L
        attn = attn / self.scale_factor
        attn = self.softmax(attn)
        # attn: N x 1 x L
        v = self.lin_v(x)
        # v: N x L x d_model
        out = torch.bmm(attn, v).squeeze(1)
        # out: N x d_model

        if return_attention_weights:
            return out, attn.squeeze(1)
        
        return out

class ManyToOneAttentionBlock(nn.Module):
    def __init__(self, d_model=768, vocab_size=8000):
        super().__init__()

        config = BertConfig(vocab_size=vocab_size, hidden_size=d_model)
        self.attn = AttentionLayer(d_model)
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(d_model, eps=config.layer_norm_eps)

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
            return_attention_weights (bool): whether to return the tensor weights
                with the network output
        """
        if return_attention_weights:
            # x: N x L x d_model
            out, attn_weights = self.attn(x, return_attention_weights=return_attention_weights)
            # out: N x 1 x d_model
            out = self.dropout(out)
            # out: N x 1 x d_model
            # out = self.layer_norm(out)
            # Experimenting with skip connection
            out = self.layer_norm(torch.mean(x, dim=1) + out)
            # out: N x 1 x d_model
            return out, attn_weights

        out = self.attn(x)
        out = self.dropout(out)
        # out = self.layer_norm(out)
        # Experimenting with skip connection
        out = self.layer_norm(torch.mean(x, dim=1) + out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=768, vocab_size=8000):
        super().__init__()

        config = BertConfig(vocab_size=vocab_size, hidden_size=d_model)
        self.bert = BertModel(config)
        self.attn = ManyToOneAttentionBlock(d_model=d_model, vocab_size=vocab_size)
        
    def compute_attention(self, x):
        return self.attn.compute_attention(out)

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
            return_attention_weights (bool): whether to return the tensor weights
                with the network output
        """
        # x: N x L x d_model
        out = self.bert(x)[0]
        # out: N x L x d_model

        if return_attention_weights:
            out, attn_weights = self.attn(out, return_attention_weights=return_attention_weights)
            # out: N x 1 x d_model
            return out, attn_weights

        out = self.attn(out)
        # out: N x 1 x d_model
        return out


class CrossAttentionBlock(nn.Module):
    """A 'Cross Attention' block. This is the part of the TransformerDecoder that 
    differs from the original. Rather than attending separately over the output 
    sequence and the input sequence, it attends over both sequences jointly.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.lin_q = nn.Linear(d_model, d_model)
        self.lin_k = nn.Linear(d_model, d_model)
        self.lin_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x, context=None):
        """Run a sequence forward through the network
        Args:
            x ((N, L, d_model) torch.Tensor): embedding of the previous tokens
            context ((N, 1, d_model)): context vector from the encoder
        Returns:
            x ((N, L, d_model) torch.Tensor): deeper embedding of the previous tokens
        """
        if context is None:
            context = torch.zeros(x.shape[0], 1, self.d_model)

        batch_size, L, _ = x.shape
        # TODO: Figure out how to implement mask
        mask = torch.ones(L, L)

        mask = torch.concat(torch.ones(L), mask, dim=1)
        context_x = torch.concat(context, x, dim=1)
        Q = self.lin_q(x).reshape(batch_size, L, self.n_heads, self.d_k)
        K = self.lin_k(context_x).reshape(batch_size, L+1, self.n_heads, self.d_k)
        V = self.lin_v(context_x).reshape(batch_size, L+1, self.n_heads, self.d_k)
        # N x L x L+1 x n_heads cross attention matrix
        attn = torch.bmm(Q, torch.transpose(K, (1, 3)), dim=1)
        attn = self.softmax(attn)
        attn = attn / torch.sqrt(self.d_k)
        # N x L x d_k x n_heads attention output
        out = torch.bmm(attn, V)
        # N x L x d_model output
        out = torch.concat(out, dim=(2, 3))
        return out

class TransformerDecoderBlock(nn.Module):
    """A single TransformerDecoder block, including attention, a linear layer,
    a norm, and a skip connection.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.attention = CrossAttentionBlock(d_model, n_heads)
        self.linear = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm()

    def forward(self, x, context=None):
        """Run a sequence forward through the network
        Args:
            x ((N, L, d_model) torch.Tensor): embedding of the previous tokens
            context ((N, 1, d_model)): context vector from the encoder
        Returns:
            x ((N, L, d_model) torch.Tensor): deeper embedding of the previous tokens
        """
        if context is None:
            context = torch.zeros(x.shape[0], 1, self.d_model)
        out = self.attention(x, context)
        out = self.linear(x)
        out = self.layer_norm(x)
        # Skip connection
        out = out + x
        return out
        

class TransformerDecoder(nn.Module):
    """Decoder from the Transformer (Vaswani 2017) with a slight modification - 
    masked self-attention and cross attention are merged into a single attention
    block that attends across both sequences.
    """
    def __init__(self, vocab_size=8000, n_blocks=100, d_model=128, n_heads=8):
        """Constructor for the TransformerDecoder.
        Args:
            vocab_size (int): the number of classes in the input and in the output
            n_blocks (int): the number of self-attention blocks to include
            d_model (int): the hidden dimension of the model
            n_heads (int): the number of different attention heads to use
        """
        super().__init__()

        self.d_model = d_model

        # Embedding and positional encoding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encod = nn.PositionalEncoding()
        # Cross attention layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_blocks)
        ])
        # Classifier layer
        self.final_linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, context=None):
        """Run a sequence forward through the network
        Args:
            x ((N, L, d_model) torch.Tensor): indices of the previous tokens
            context ((N, 1, d_model)): context vector from the encoder
        Returns:
            out ((N, L, vocab_size) torch.Tensor): the predictions for the next tokens
        """
        if context is None:
            context = torch.zeros(x.shape[0], 1, self.d_model)
        # Embedding and positional encoding
        out = self.embed(x)
        out += self.pos_encode(out)
        # Cross attention layers
        for block in self.layers:
            out = block(out, context)
        # Classifier layer
        out = self.final_linear(out)
        return out

class TransformerAutoEncoderActivityPredictor(nn.Module):
    def __init__(self, vocab_size=8000, d_model=768, pred_dim=1, from_pretrained=False, transformer_autoencoder_path='/models/hydrolase_design/transformer_autoencoder'):
        super().__init__()

        encoder_path = os.path.join(transformer_autoencoder_path, 'encoder.pt')
        decoder_path = os.path.join(transformer_autoencoder_path, 'decoder.pt')
        activity_predictor_path = os.path.join(transformer_autoencoder_path, 'activity_predictor.pt')

        if self.from_pretrained:
            if os.path.exists(encoder_path):
                self.encoder = torch.load(encoder_path)
            else:
                print("Pretrained was set to True, but no pretrained encoder was found. \
                    Training encoder from scratch.")
        else:
            self.encoder = TransformerEncoder(d_model=d_model, vocab_size=vocab_size)

        if self.from_pretrained:
            if os.path.exists(decoder_path):
                self.decoder = torch.load(decoder_path)
            else:
                print("Pretrained was set to True, but no pretrained decoder was found. \
                    Training decoder from scratch.")

        if self.from_pretrained:
            if os.path.exists(activity_predictor_path):
                self.activity_predictor = torch.load(activity_predictor_path)
            else:
                print("Pretrained was set to True, but no pretrained activity network was found. \
                    Training activity predictor from scratch.")
        else:
            self.last_lin = nn.Linear(d_model, pred_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def autoencode(self, x):
        return self.decode(self.encode(x))

    def predict_activity(self, x):
        return self.last_lin((self.encode(x)))

    def forward(self, x):
        """Run a forward pass that outputs predicted activity and 
        reconstructed input.
        Args:
            x ((N x L) torch.Tensor): an item from the training set
        """
        latent = self.encode(x)        
        return self.last_lin(latent), self.decode(latent)

class TransformerAutoEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.encoder = TransformerEncoder(d_model=d_model)
        self.decoder = TransformerDecoder(d_model=d_model)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        return self.decode(self.encode(x))

    def infer(self, x):
        out = x
        out_list = [out]

        while out is not self.end_token:
            out = self.decode(out)
            out_list.append(out)
            
        return ''.join(out_list)

class FullyConnectedActivityPredictor1(nn.Module):
    """An architecture more suited to finding patterns based on positions in a fixed-length vector
    """
    def __init__(self, d_model=768, vocab_size=8000, seq_length=200, from_pretrained=False, base_savepath='/models/hydrolase_design/fc_activity_predictor'):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.lin_0 = nn.Linear(d_model, d_model)
        self.lin_1 = nn.Linear(d_model, 1)
        # self.act = nn.ReLU()
        # self.act = nn.Tanh()
        self.act = nn.Sigmoid()
        self.bias_layer = nn.Parameter(data=torch.ones(seq_length), requires_grad=True)
        self.lin_2 = nn.Linear(seq_length, 1)

        if from_pretrained:
            activity_predictor_path = os.path.join(base_savepath, 'fc_activity_predictor.pt')
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
        # Positivity(r) = -(flexible(r) and hydrophobic(r))
        out = self.embedding(x)
        out = self.lin_0(out)
        out = self.act(out)
        out = self.lin_1(out).squeeze(-1)
        out = self.act(out)
        out = out.clone() + self.bias_layer
        out = self.lin_2(out)

        if return_attention_weights:
            return out, self.lin_2.weight.squeeze(-1)

        return out

class FullyConnectedActivityPredictor(nn.Module):
    """An architecture more suited to finding patterns based on positions in a fixed-length vector
    """
    def __init__(self, d_model=768, vocab_size=8000, seq_length=200, from_pretrained=False, base_savepath='/models/hydrolase_design/nt_activity_predictor', dataset_dir=None, amino_acids=np.array(list('AILMFWYVCGPRNDQEHKST')), base_seq=np.array(list('MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLI'))):
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
            activity_predictor_path = os.path.join(base_savepath, 'nt_activity_predictor.pt')
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

class TransformerActivityPredictor(nn.Module):
    def __init__(self, d_model=768, vocab_size=8000, pred_dim=1, from_pretrained=False, model_name='model', base_savepath='/models/hydrolase_design/transformer_activity_predictor'):
        super().__init__()

        encoder_path = os.path.join(base_savepath, 'encoder.pt')
        activity_predictor_path = os.path.join(base_savepath, 'activity_predictor.pt')

        if from_pretrained:
            if os.path.exists(encoder_path):
                self.encoder = torch.load(encoder_path)
            else:
                print("Pretrained was set to True, but no pretrained encoder was found. \
                    Training encoder from scratch.")
        else:
            self.encoder = TransformerEncoder(d_model=d_model, vocab_size=vocab_size)

        if from_pretrained:
            if os.path.exists(activity_predictor_path):
                self.activity_predictor = torch.load(activity_predictor_path)
            else:
                print("Pretrained was set to True, but no pretrained activity network was found. \
                    Training activity predictor from scratch.")
        else:
            self.last_lin = nn.Linear(d_model, pred_dim)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, return_attention_weights=False):
        """Run a forward pass that outputs predicted activity and 
        reconstructed input.
        Args:
            x ((N x L) torch.Tensor): an item from the training set
            return_attention_weights (bool): whether to return the tensor weights
                with the network output
        """
        if return_attention_weights:
            out, attn_weights = self.encoder(x, return_attention_weights=return_attention_weights)
            out = self.last_lin(out)
            return out, attn_weights
        
        out = self.encoder(x)
        out = self.last_lin(out)
        return out