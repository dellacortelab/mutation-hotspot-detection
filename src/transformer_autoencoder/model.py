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