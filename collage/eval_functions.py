'''
Functions for model evaluation (loss)
'''
import torch
import torch.nn as nn


class WeightedLikelihood(nn.Module):
    def __init__(self):  # log_codon_freq : np.ndarray):
        super().__init__()

        # self.prior = nn.Embedding( 64, 1)
        # self.prior.weight = nn.Parameter( torch.Tensor( log_codon_freq ),
        #                                  requires_grad=False)

        self.nllloss = nn.NLLLoss(reduction='none')

    def forward(self, logprob_by_codon, obs_codons, weight):
        '''
        logprob_by_codons: tensor of size (batch_size,seq_len,codons)
        obs_codons: tensor of size (batch_size,seq_len)
        '''

        neg_logLs = self.nllloss(logprob_by_codon.transpose(1, -1),
                                 obs_codons)  # Should be size ( B, L )
        mask = neg_logLs != 0.0
        loss = torch.sum(weight * neg_logLs) / torch.sum(weight * mask)

        return loss


def calc_accuracy(targets: torch.Tensor,
                  log_probabilities: torch.Tensor) -> float:
    '''
    For a given set of targets and a probability matrix, calculate the accuracy
    '''

    max_logLs, max_idx_class = log_probabilities.max(dim=-1)  # [ B, n_classes ] -> [ B ]
    n = targets.flatten().size(0)
    accuracy = torch.sum(max_idx_class == targets).item() / n
    return accuracy
