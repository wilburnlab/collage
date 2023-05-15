## Codon Optimizing Transformer Model
import math
import torch
import torch.nn as nn

from collage.reference_data import codon_to_residue, codons, residues, int_to_residue


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp( torch.arange(0, dim_model, 2).float() * (-math.log(1e4)) / dim_model ) # 1000^(2i/dim_model), neg for reciprocal in division
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0) #.transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        #return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])
        return self.dropout( token_embedding + self.pos_encoding[ :, :token_embedding.size(1), : ] )


class Prot_Encoder( nn.Module ):
    def __init__( self, n_tokens, model_dim, ff_dim, n_heads, n_layers, dropout, max_len, n_sp ):
        super().__init__()
        self.model_dim = model_dim
        self.prot_embedding = nn.Embedding( n_tokens, model_dim, )
        self.encoder_layer = nn.TransformerEncoderLayer( model_dim,
                                                         n_heads,
                                                         ff_dim,
                                                         dropout,
                                                         batch_first=True, )
        self.transformer_encoder = nn.TransformerEncoder( self.encoder_layer,
                                                          n_layers, )
        self.pos_encoder = PositionalEncoding( model_dim, dropout, max_len, )

        self.species_embed = nn.Embedding( n_sp, model_dim )


    def forward( self, prot, ): # sp, ):
        pad_mask = (prot == 0).to( prot.device )
        x = self.prot_embedding( prot ) * math.sqrt( self.model_dim )
        x = self.pos_encoder( x )
        #x = x + self.species_embed( sp )
        x = self.transformer_encoder( x, src_key_padding_mask = pad_mask, )
        return x

class Codon_Decoder( nn.Module ):
    def __init__( self, n_tokens, model_dim, ff_dim, n_heads, n_layers, dropout, max_len, n_sp ):
        super().__init__()
        self.model_dim = model_dim
        self.codon_embedding = nn.Embedding( n_tokens, model_dim, )
        self.decoder_layer = nn.TransformerDecoderLayer( model_dim,
                                                         n_heads,
                                                         ff_dim,
                                                         dropout,
                                                         batch_first=True, )
        self.transformer_decoder = nn.TransformerDecoder( self.decoder_layer,
                                                          n_layers, )
        self.pos_encoder = PositionalEncoding( model_dim, dropout, max_len, )

        #self.species_embed = nn.Embedding( n_sp, model_dim )

    def get_tgt_mask( self, size, ):
        ones = torch.ones( size, size ) == 1
        mask = torch.tril( ones, diagonal=0, ) #+ torch.triu( ones, diagonal=2 )
        return torch.where( mask, 0.0, -torch.inf  )

    def forward( self, prot, tgt, memory, ): # sp, ):
        tgt_pad_mask = (tgt == 0).to( tgt.device )
        prot_pad_mask = (prot == 0).to( prot.device )
        tgt_self_mask = self.get_tgt_mask( tgt.size(1) ).to( tgt.device )

        x = self.codon_embedding( tgt )
        x = self.pos_encoder( x )
        #x = x + self.species_embed( sp )
        x = self.transformer_decoder( x, memory, tgt_mask = tgt_self_mask,
                                      tgt_key_padding_mask = tgt_pad_mask,
                                      memory_key_padding_mask = prot_pad_mask, )
                                      
        return x

class Codon_Predictor( nn.Module ):
    def __init__( self, n_input_tokens, n_output_tokens, n_sp, model_dim, ff_dim,
                  n_heads, n_encoder_layers, n_decoder_layers, dropout,
                  max_len, ):
        super().__init__()


        self.prot_encoder = Prot_Encoder( n_input_tokens, 
                                          model_dim, 
                                          ff_dim, 
                                          n_heads, 
                                          n_encoder_layers, 
                                          dropout, 
                                          max_len,
                                          n_sp, )
        self.codon_decoder = Codon_Decoder( n_output_tokens, 
                                            model_dim, 
                                            ff_dim, 
                                            n_heads, 
                                            n_decoder_layers, 
                                            dropout, 
                                            max_len,
                                            n_sp, )
        
        codon_mask_embedding = [ [ float( int_to_residue[i] == codon_to_residue[ codons[j] ] )
                                   for j in range( len(codons) - 1 ) ] 
                                 for i in range( len(residues) ) ]
        self.codon_masker = nn.Embedding.from_pretrained( torch.tensor( codon_mask_embedding ) )
        self.linear = nn.Linear(model_dim, n_output_tokens-1)
        self.softmax = nn.LogSoftmax( dim=-1, )

        self.species_embed = nn.Embedding( n_sp, (len(codons)-1) )

    def species_weighted_mask( self, prot, ): #species, prot, ):
        codon_mask = self.codon_masker( prot )
        return torch.where( codon_mask != 0, 0.0, -torch.inf )
        #species_embed = self.species_embed( species )
        #weighted_mask = species_embed * codon_mask
        #weighted_mask[ weighted_mask == 0 ] = -torch.inf
        #return weighted_mask


    def forward( self, prot, cds, ): #sp, ):
        species_mask = self.species_weighted_mask( prot, ) #sp, prot, )

        x = self.prot_encoder( prot, ) #sp, )
        x = self.codon_decoder( prot, cds, x, ) #sp, )
        x = self.linear( x )
        
        return self.softmax( x + species_mask[ :, :x.size(1), : ] )










'''

class Prot_Transformer( nn.Module ):
    def __init__( self, n_tokens, model_dim, ff_dim, n_heads, n_layers, dropout, max_len, ):
        super().__init__()
        self.model_dim = model_dim
        self.prot_embedding = nn.Embedding( n_tokens, model_dim, )
        self.encoder_layer = nn.TransformerEncoderLayer( model_dim,
                                                         n_heads,
                                                         ff_dim,
                                                         dropout,
                                                         batch_first=True, )
        self.transformer_encoder = nn.TransformerEncoder( self.encoder_layer,
                                                          n_layers, )
        self.pos_encoder = PositionalEncoding( model_dim, dropout, max_len, )

    def forward( self, src, ):
        pad_mask = (src == 0).to( src.device )
        x = self.prot_embedding( src ) * math.sqrt( self.model_dim )
        x = self.pos_encoder( x )
        x = self.transformer_encoder( x, src_key_padding_mask = pad_mask, )
        return x


class Codon_Transformer( nn.Module ):
    def __init__( self, n_tokens, model_dim, ff_dim, n_heads, n_layers, dropout, max_len, n_sp ):
        super().__init__()
        self.model_dim = model_dim
        self.codon_embedding = nn.Embedding( n_tokens, model_dim, )
        self.decoder_layer = nn.TransformerDecoderLayer( model_dim,
                                                         n_heads,
                                                         ff_dim,
                                                         dropout,
                                                         batch_first=True, )
        self.transformer_decoder = nn.TransformerDecoder( self.decoder_layer,
                                                          n_layers, )
        self.pos_encoder = PositionalEncoding( model_dim, dropout, max_len, )

        self.species_embed = nn.Embedding( n_sp, model_dim )

    def get_tgt_mask( self, size, ):
        ones = torch.ones( size, size ) == 1
        mask = torch.tril( ones, diagonal=0, ) #+ torch.triu( ones, diagonal=2 )
        return torch.where( mask, 0.0, -torch.inf  )

    def forward( self, src, tgt, memory, sp, ):
        tgt_pad_mask = (tgt == 0).to( tgt.device )
        src_pad_mask = (src == 0).to( src.device )
        tgt_self_mask = self.get_tgt_mask( tgt.size(1) ).to( tgt.device )

        x = self.codon_embedding( tgt )
        x = self.pos_encoder( x )
        x = x + self.species_embed( sp )
        x = self.transformer_decoder( x, memory, tgt_mask = tgt_self_mask,
                                      tgt_key_padding_mask = tgt_pad_mask,
                                      memory_key_padding_mask = src_pad_mask, )
                                      
        return x




class Nuc_Transformer( nn.Module ):
    def __init__( self, n_tokens, model_dim, ff_dim, n_heads, n_layers, dropout, max_len ):
        super().__init__()
        self.model_dim = model_dim
        self.codon_embedding = nn.Embedding( n_tokens, model_dim, )
        self.decoder_layer = nn.TransformerDecoderLayer( model_dim,
                                                         n_heads,
                                                         ff_dim,
                                                         dropout,
                                                         batch_first=True, )
        self.transformer_decoder = nn.TransformerDecoder( self.decoder_layer,
                                                          n_layers, )
        self.pos_encoder = PositionalEncoding( model_dim, dropout, max_len, )

    def get_tgt_mask( self, size, ):
        ones = torch.ones( size, size ) == 1
        mask = torch.tril( ones, diagonal=0, ) + torch.triu( ones, diagonal=2 )
        for i in range(2):
            mask = torch.repeat_interleave( mask, 3, dim=i )
        return torch.where( mask, 0.0, -torch.inf  )

    def forward( self, src, tgt, memory, ):
        tgt_pad_mask = (tgt == 0).to( tgt.device )
        src_pad_mask = (src == 0).to( src.device )
        #print( src.shape, tgt.shape, memory.shape, )
        tgt_self_mask = self.get_tgt_mask( src.size(1) ).to( tgt.device )
        #tgt_self_mask = torch.repeat_interleave( torch.repeat_interleave( self.get_tgt_mask( tgt.size(1) ), dim=1, ) )
        #                                                                  3, dim=1, ),
        #                                         3, dim=0 ).to( tgt.device )
        #tgt_self_mask = torch.repeat_interleave( torch.repeat_interleave( self.get_tgt_mask( src.size(1) ),
        #                                                                  3, dim=1, ),
        #                                         3, dim=0 ).to( tgt.device )

        #print( src.shape, tgt.shape, tgt_self_mask.shape )

        ###
        ### I probably need to patch tgt to be the full sequence and not just tgt[:,:-1]
        ### The code might work as is...consider checking...
        ###

        x = self.codon_embedding( tgt )
        x = self.pos_encoder( x )
        x = self.transformer_decoder( x, memory, tgt_mask = tgt_self_mask,
                                      tgt_key_padding_mask = tgt_pad_mask,
                                      memory_key_padding_mask = src_pad_mask, )
                                      
        return x

class Optimizer_Transformer( nn.Module ):

    def __init__( self, n_input_tokens, n_output_tokens, n_sp, model_dim, ff_dim,
                  n_heads, n_encoder_layers, n_decoder_layers, dropout,
                  max_len, ):
        super().__init__()


        self.prot_transformer = Prot_Transformer( n_input_tokens, 
                                                  model_dim, 
                                                  ff_dim, 
                                                  n_heads, 
                                                  n_encoder_layers, 
                                                  dropout, 
                                                  max_len, )
        self.codon_transformer = Codon_Transformer( n_output_tokens, 
                                                    model_dim, 
                                                    ff_dim, 
                                                    n_heads, 
                                                    n_decoder_layers, 
                                                    dropout, 
                                                    max_len,
                                                    n_sp, )
        
        self.nuc_transformer = Nuc_Transformer( 6, 
                                                model_dim, 
                                                ff_dim, 
                                                n_heads, 
                                                n_decoder_layers, 
                                                dropout, 
                                                max_len*3, )
        
        codon_mask_embedding = [ [ float( int_to_residue[i] == Codon_chart[ codons[j] ] )
                                   for j in range( len(codons) - 1 ) ] 
                                 for i in range( len(residues) ) ]
        self.codon_masker = nn.Embedding.from_pretrained( torch.tensor( codon_mask_embedding ) )
        self.linear = nn.Linear(model_dim, n_output_tokens-1)
        self.softmax = nn.LogSoftmax( dim=-1, )

        self.species_embed = nn.Embedding( n_sp, (len(codons)-1) )

        self.conv = nn.Conv1d( model_dim, n_output_tokens-1, 3, 3, )


    def species_weighted_mask( self, species, src, ):
        species_embed = self.species_embed( species )
        codon_mask = self.codon_masker( src )
        weighted_mask = species_embed * codon_mask
        weighted_mask[ weighted_mask == 0 ] = -torch.inf
        return torch.where( codon_mask != 0, 0.0, -torch.inf )
        #return weighted_mask
        #return torch.where( codon_mask != 0, species_embed, torch.full( codon_mask.shape, -torch.inf, ) )



    def forward( self, src, tgt, sp, nt=False, ):
        species_mask = self.species_weighted_mask( sp, src, )
        #codon_mask = torch.where( self.codon_masker( src, ) != 0, 0.0, -torch.inf )
        #codon_mask = torch.repeat_interleave( codon_mask, 3, dim=1, )

        x = self.prot_transformer( src, )
        if not nt:
            x = self.codon_transformer( src, tgt, x, sp, )
            x = self.linear( x )
        #print( x.shape )
        else:
            x = self.nuc_transformer( src, tgt, x, )
            x = self.conv( x.transpose(1,-1) ).transpose(1,-1)
        #print( x.shape )
        #species_embed = torch.reshape( self.species_embed(sp), self.species_embed_shape )
        #weighted_mask = codon_mask[ :, :x.size(1), : ] * species_embed
        #print( x.shape, species_mask.shape )
        return self.softmax( x + species_mask )


'''



                            


        




