'''
Functions for model evaluation (loss)
'''

# Libraries
import numpy as np
import torch
import torch.nn as nn



class Relative_Codon_Likelihood( nn.Module ):
    def __init__( self, ): #log_codon_freq : np.ndarray, ):
        super().__init__()

        #self.prior = nn.Embedding( 64, 1, )
        #self.prior.weight = nn.Parameter( torch.Tensor( log_codon_freq ),
        #                                  requires_grad=False, )

        self.nllloss = nn.NLLLoss( reduction='none', )

    def forward( self, logprob_by_codon, obs_codons ):
        '''
        logprob_by_codons: tensor of size (batch_size,seq_len,codons)
        obs_codons: tensor of size (batch_size,seq_len)
        '''

        neg_logLs = self.nllloss( logprob_by_codon.transpose( 1, -1, ),
                                  obs_codons ) # Should be size ( B, L )
        #marginal = self.prior( obs_codons ).reshape( neg_logLs.shape )

        neg_posterior = neg_logLs #+ marginal
        posterior_mask = neg_posterior != 0.0
        loss = torch.sum( neg_posterior ) / torch.sum( posterior_mask )
        return loss
        







