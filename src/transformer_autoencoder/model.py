#######################################################################################################
# Neural network model classes for the Transformer AutoEncoder
#######################################################################################################

import os
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertConfig

class AttentionLayer(nn.Module):
    # TODO: Figure out how to get queries
    def __init__(self, bert_config):
        super().__init__()

        d_model = bert_config.hidden_size
        # TODO: add multiple heads
        self.lin_q = nn.Linear(d_model, d_model)
        self.lin_k = nn.Linear(d_model, d_model)
        self.lin_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)
        self.scale_factor = torch.sqrt(torch.tensor(d_model, dtype=torch.float))

    def forward(self, x):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
        """
        # N x L x d_model
        q = self.lin_q(x)
        # Sum over L dimension - output N x 1 x d_model
        q = torch.sum(q, dim=1, keepdim=True)
        # N x L x d_model
        k = self.lin_k(x)
        # N x d_model x L
        k = torch.transpose(k, 1, 2)
        # N x 1 x L
        attn = torch.bmm(q, k)
        attn = self.softmax(attn)

        attn = attn / self.scale_factor
        # N x L x d_model
        v = self.lin_v(x)
        # N x d_model
        out = torch.bmm(attn, v).squeeze(1)
        return out

class ManyToOneAttentionBlock(nn.Module):
    def __init__(self, d_model=768, vocab_size=8000):
        super().__init__()

        config = BertConfig(vocab_size=vocab_size, hidden_size=d_model)
        self.attn = AttentionLayer(config)
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=config.layer_norm_eps)

    def forward(self, x):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
        """
        # Attention Block
        out = self.attn(x)
        out = self.dropout(out)
        block_1_out = self.layer_norm_1(x + out)
        # Last Linear Block
        out = self.lin(block_1_out)
        out = self.dropout(out)
        out = self.layer_norm_2(out + block_1_out)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=768, vocab_size=8000):
        super().__init__()

        config = BertConfig(vocab_size=vocab_size, hidden_size=d_model)
        self.bert = BertModel(config)
        self.attn = AttentionLayer(config)
        self.bn = nn.LayerNorm(d_model, eps=config.layer_norm_eps)
        self.lin = nn.Linear(d_model, d_model)
        

    def forward(self, x):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
        """
        # x: N x L x d_model
        bert_out = self.bert(x)[0]
        # out: N x L x d_model
        out = self.attn(bert_out)
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

    def predict_activity(self, x):
        return self.last_lin((self.encode(x)))

    def forward(self, x):
        """Run a forward pass that outputs predicted activity and 
        reconstructed input.
        Args:
            x ((N x L) torch.Tensor): an item from the training set
        """
        return self.last_lin((self.encode(x)))