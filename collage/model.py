# Codon Optimizing Transformer Model
import math
import torch
import torch.nn as nn

from collage.reference_data import CODON_MASK_EMBEDDING
from collage.defaults import HYPERPARAMETERS, TRAINING_PARAMETERS


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(1e4)) / dim_model)  # 1000^(2i/dim_model), neg for reciprocal in division

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0)  # .transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:, :token_embedding.size(1), :])


class ProtEncoder(nn.Module):
    def __init__(self, n_tokens, embed_dim, ff_dim, n_heads, n_layers, dropout, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.prot_embedding = nn.Embedding(n_tokens, embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim,
                                                        n_heads,
                                                        ff_dim,
                                                        dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)

    def forward(self, prot):  # sp):
        pad_mask = (prot == 0).to(prot.device)
        x = self.prot_embedding(prot) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        return x


class CodonDecoder(nn.Module):
    def __init__(self, n_tokens, embed_dim, ff_dim, n_heads, n_layers, dropout, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.codon_embedding = nn.Embedding(n_tokens, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(embed_dim,
                                                        n_heads,
                                                        ff_dim,
                                                        dropout,
                                                        batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, n_layers)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)

    def get_tgt_mask(self, size):
        ones = torch.ones(size, size) == 1
        mask = torch.tril(ones, diagonal=0)
        return torch.where(mask, 0.0, -torch.inf)

    def forward(self, prot, tgt, memory):
        tgt_pad_mask = (tgt == 0).to(tgt.device)
        prot_pad_mask = (prot == 0).to(prot.device)
        tgt_self_mask = self.get_tgt_mask(tgt.size(1)).to(tgt.device)

        x = self.codon_embedding(tgt)
        x = self.pos_encoder(x)
        x = self.transformer_decoder(x, memory, tgt_mask=tgt_self_mask,
                                     tgt_key_padding_mask=tgt_pad_mask,
                                     memory_key_padding_mask=prot_pad_mask)

        return x


class CollageModel(nn.Module):
    def __init__(self, n_input_tokens, n_output_tokens, embed_dim, ff_dim,
                 n_heads, n_encoder_layers, n_decoder_layers, dropout,
                 max_len):
        super().__init__()

        self.protein_encoder = ProtEncoder(n_input_tokens,
                                           embed_dim,
                                           ff_dim,
                                           n_heads,
                                           n_encoder_layers,
                                           dropout,
                                           max_len)
        self.codon_decoder = CodonDecoder(n_output_tokens,
                                          embed_dim,
                                          ff_dim,
                                          n_heads,
                                          n_decoder_layers,
                                          dropout,
                                          max_len)

        self.codon_masker = nn.Embedding.from_pretrained(torch.tensor(CODON_MASK_EMBEDDING))
        self.linear = nn.Linear(embed_dim, n_output_tokens - 1)
        nn.init.constant_(self.linear.bias, -math.log(64.0 - 1))
        self.softmax = nn.LogSoftmax(dim=-1)

    def return_codon_mask(self, protein):
        codon_mask = self.codon_masker(protein)
        return torch.where(codon_mask != 0, 0.0, -torch.inf)

    def forward(self, protein, cds):
        '''
        protein: [batch_size, protein_length]
        cds: [batch_size, codons_length]

        return [batch_size, codons_length+1, 65] (64 possible residues + special start tag)
        '''
        codon_mask = self.return_codon_mask(protein)

        x = self.protein_encoder(protein)
        x = self.codon_decoder(protein, cds, x)
        x = self.linear(x)

        return self.softmax(x + codon_mask[:, :x.size(1), :])


def initialize_collage_model(model_file=None, gpu=False):

    device = 'cuda' if gpu else 'cpu'

    model = CollageModel(HYPERPARAMETERS['n_input_tokens'],
                         HYPERPARAMETERS['n_output_tokens'],
                         HYPERPARAMETERS['embed_dimension'],
                         HYPERPARAMETERS['feedforward_dimension'],
                         HYPERPARAMETERS['n_heads'],
                         HYPERPARAMETERS['n_encoder_layers'],
                         HYPERPARAMETERS['n_decoder_layers'],
                         TRAINING_PARAMETERS['dropout_rate'],
                         HYPERPARAMETERS['max_sequence_length'])
    if model_file:
        model.load_state_dict(torch.load(model_file, map_location=torch.device(device)),
                              strict=True)
    model.eval()
    return model
