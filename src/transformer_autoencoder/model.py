#######################################################################################################
# Neural network model classes for the Transformer AutoEncoder
#######################################################################################################

from transformers import BertModel, BertConfig

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()

        self.lin_q = nn.Linear(d_model, )
        self.lin_k = nn.Linear(d_model, d_hidden)
        self.lin_v = nn.Linear(d_model, d_hidden)

    def forward(self, x):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
        """
        k = self.lin_k(x)

class TranformerEncoder(nn.Module):
    def __init__(self, d_model):
        super(TransformerEncoder, self).__init__()

        self.bert = BertModel(BertConfig(vocab_size=vocab_size))
        self.attn = AttentionLayer()
        

    def forward(self, x):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
        """
        # x: N x L x d_model
        out = self.bert(x)
        # out: N x L x d_model
        out = self.attn(out)
        # out: N x 1 x d_model
        out = out.squeeze(1)
        # out: N x d_model
        return out
    


class CrossAttentionBlock(nn.Module):
    """A 'Cross Attention' block. This is the part of the TransformerDecoder that 
    differs from the original. Rather than attending separately over the output 
    sequence and the input sequence, it attends over both sequences jointly.
    """
    def __init__(self, d_model, n_heads):
        super(AttentionBlock, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.lin_q = nn.Linear(d_model, d_model)
        self.lin_k = nn.Linear(d_model, d_model)
        self.lin_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x, context=torch.zeros(x.shape[0], 1, self.d_model)):
        """Run a sequence forward through the network
        Args:
            x ((N, L, d_model) torch.Tensor): embedding of the previous tokens
            context ((N, 1, d_model)): context vector from the encoder
        Returns:
            x ((N, L, d_model) torch.Tensor): deeper embedding of the previous tokens
        """
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
        super(TransformerDecoderBlock, self).__init__()

        self.attention = CrossAttentionBlock(d_model, n_heads)
        self.linear = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm()

    def forward(self, x, context=torch.zeros(x.shape[0], 1, self.d_model)):
        """Run a sequence forward through the network
        Args:
            x ((N, L, d_model) torch.Tensor): embedding of the previous tokens
            context ((N, 1, d_model)): context vector from the encoder
        Returns:
            x ((N, L, d_model) torch.Tensor): deeper embedding of the previous tokens
        """
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
    def __init__(self, d_vocab=8000, n_blocks=100, d_model=128, n_heads=8):
        """Constructor for the TransformerDecoder.
        Args:
            d_vocab (int): the number of classes in the input and in the output
            n_blocks (int): the number of self-attention blocks to include
            d_model (int): the hidden dimension of the model
            n_heads (int): the number of different attention heads to use
        """
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model

        # Embedding and positional encoding
        self.embed = nn.Embedding(d_vocab, d_model)
        self.pos_encod = nn.PositionalEncoding()
        # Cross attention layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_blocks)
        ])
        # Classifier layer
        self.final_linear = nn.Linear(d_model, d_vocab)

    def forward(self, x, context=torch.zeros(x.shape[0], 1, self.d_model)):
        """Run a sequence forward through the network
        Args:
            x ((N, L, d_model) torch.Tensor): indices of the previous tokens
            context ((N, 1, d_model)): context vector from the encoder
        Returns:
            out ((N, L, n_classes) torch.Tensor): the predictions for the next tokens
        """
        # Embedding and positional encoding
        out = self.embed(x)
        out += self.pos_encode(out)
        # Cross attention layers
        for block in self.layers:
            out = block(out, context)
        # Classifier layer
        out = self.final_linear(out)
        return out


class TransformerAutoEncoder(nn.Module):
    def __init__(self, d_emb, d_model):
        super(TransformerAutoEncoder, self).__init__()

        self.encoder = TransformerEncoder(d_emb, d_model)
        self.decoder = TransformerDecoder(d_emb, d_model)

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


class TransformerAutoEncoderReactivityPredictor(nn.Module):
    def __init__(self, d_emb, d_model, n_classes, transformer_autoencoder_path='/data/uniparc'):
        super(TransformerAutoEncoderReactivityPredictor, self).__init__()

        if os.path.exists(transformer_autoencoder_path):
            self.autoencoder = 

        self.autoencoder = TransformerAutoEncoder(d_emb, d_model)
        self.last_lin = nn.Linear(d_model, n_classes)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def autoencode(self, x):
        return self.decode(self.encode(x))

    def predict_reactivity(self, x):
        return self.last_lin((self.encode(x)))

    def forward(self, x):
        """Run a forward pass that outputs predicted reactivity and 
        reconstructed input.
        Args:
            x ((N x L) torch.Tensor): an item from the training set
        """
        latent = self.encode(x)        
        return self.last_lin(latent), self.decode(latent)