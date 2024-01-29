
import re
import numpy as np
import torch

from collage.utils import translate
from collage.utils import prot_to_coded, orf_to_coded, nt_to_coded, timer
from collage.reference_data import RESIDUE_TO_INT, CODON_TO_INT, NUCLEOTIDE_TO_INT, CODONS

import sys
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
def get_tensor_size(a):
    bytes = a.element_size() * a.nelement()
    return sizeof_fmt(bytes)

import time


# Let's write a test beam generator function

current_species = ['Ecoli_K12']


def gc_100_penalty(seq):
    # Convert the sequence to GC code
    gc = [b in ['C', 'G'] for b in seq]
    slices = [gc[i: i + 100] for i in range(0, len(gc), 100)]
    percents = [np.mean(s) for s in slices]
    #exceptions = [p >= 0.65 for p in percents]
    #return np.any(exceptions)
    penalty = np.square(np.abs(np.max(percents)-0.5)/0.05)
    return penalty


def beam_generator(model, prot, pre_sequence='', gen_size=500, max_seqs=100):
    assert len(
        pre_sequence) % 3 == 0, 'Start sequence length is not a multiple of 3'

    prot_prefix = translate(pre_sequence)
    prot_prefix_len = len(prot_prefix)

    full_prot = translate(pre_sequence) + prot
    full_prot_len = len(full_prot)

    current_seqs = {pre_sequence: 0.0}

    # start_codons = re.findall( '...', start )
    # start_coded = [ 65 ] + [ codon_to_int[ c ] for c in start_codons ]
    # orf_tensor = torch.Tensor( [ start_coded ] ).to( torch.int64 )
    for i in range(prot_prefix_len, full_prot_len):
        if i < gen_size:
            prot_seq = full_prot[: gen_size]
        else:
            prot_seq = full_prot[i - gen_size + 1: i + 1]
        prot_coded = prot_to_coded(prot_seq)
        prot_tensor_0 = torch.tensor([prot_coded]).to(torch.int64)

        coded_seqs = []
        for seq in current_seqs:
            s_codons = re.findall('...', seq)
            if len(s_codons) >= gen_size:
                coded_seqs.append([CODON_TO_INT[c]
                                  for c in s_codons[-gen_size:]])
            else:
                coded_seqs.append([65] + [CODON_TO_INT[c] for c in s_codons])
        orf_tensor = torch.Tensor(coded_seqs).to(torch.int64)

        nt_array = np.array([nt_to_coded(seq) for seq in current_seqs])
        nt_tensor = torch.Tensor(nt_array).to(torch.int64)
        #nt_onehot = torch.nn.functional.one_hot(nt_tensor)
        #print( nt_onehot.shape )


        prot_tensor = prot_tensor_0.repeat(orf_tensor.size(0), 1)

        weights_tensor = torch.ones(prot_tensor.shape)
        #print( prot_tensor.shape, orf_tensor.shape )
        output = model(prot_tensor, orf_tensor)

        logLs = output.cpu().detach().numpy()[:, -1, :]
        candidate_seqs = {}
        for j, seq in enumerate(current_seqs):
            codon_logL = dict([(c, l) for c, l in zip(
                CODONS, logLs[j]) if np.isfinite(l)])
            for c in codon_logL:
                #if gc_100_check(seq + c):
                #    #print('TRIGGER')
                #    continue  # Avoid >65% GC
                new_seq = seq+c
                candidate_seqs[new_seq] = current_seqs[seq] + codon_logL[c] - gc_100_penalty(new_seq)

        candidate_seqs = dict(sorted(candidate_seqs.items(), key=lambda x: x[1], reverse=True))
        current_seqs = dict(list(candidate_seqs.items())[:max_seqs])
        #print(len(current_seqs))

    return current_seqs




## New generator function that _should_ allow for full GPU-based prediction. Currently running into
## memory allocation issues. orf_tensor and prot_tensor are the same size as what we would expect
## from beam_generator, but for some reason, the memory is exploding in the model prediction step
def torch_beam_generator(model, prot, pre_sequence='', gen_size=500, max_seqs=100, device='cuda'):
    assert len(
        pre_sequence) % 3 == 0, 'Start sequence length is not a multiple of 3'

    model.to(device)

    prot_prefix = translate(pre_sequence)
    prot_prefix_len = len(prot_prefix)

    full_prot = translate(pre_sequence) + prot
    full_prot_len = len(full_prot)
    full_prot_coded = prot_to_coded(full_prot)
    full_prot_tensor = torch.tensor([full_prot_coded]).to(torch.int64).to(device)

    pre_orf_coded = [65] + orf_to_coded(pre_sequence)
    orf_tensor = torch.tensor([pre_orf_coded]).to(torch.int64).to(device)
    logLs = torch.zeros((1,1)).to(device)

    for i in range(prot_prefix_len, full_prot_len):
        # Initialize tensors for prediction
        initial_batch_size = orf_tensor.size(0)
        prot_start = max([0,i-gen_size]) # double check I don't need to add a +/- 1 on gen_size
        prot_tensor = full_prot_tensor[:,prot_start:prot_start+gen_size]
        prot_tensor = prot_tensor.repeat(orf_tensor.size(0), 1)
        
        # Calculate logL scores and append to logLs tensor
        print( prot_tensor.shape, orf_tensor.shape )
        logL_output = model(prot_tensor, orf_tensor)[:,-1,:] ## WILL HANG ON THIS STEP IF BATCH SIZE IS TOO LARGE
        n = logL_output.size(1)
        logLs = logLs.repeat_interleave(n,dim=0)
        logLs = torch.cat([logLs, logL_output.reshape((-1,1))], dim=-1)

        # Update candidates on orf_tensor
        orf_tensor = orf_tensor.repeat_interleave(n,dim=0)
        new_residues = torch.arange(n).unsqueeze(-1).repeat((initial_batch_size,1)).to(device)
        orf_tensor = torch.cat([orf_tensor,new_residues], dim=-1)

        # Filter impossible candidates
        candidates = torch.isfinite(logLs.sum(dim=-1))
        orf_tensor = orf_tensor[candidates,:]
        logLs = logLs[candidates,:]

        # Sort by rank
        sum_logLs, rank = torch.sort(logLs.sum(dim=-1), descending=True) # focus on sums after filtering
        orf_tensor = orf_tensor[rank,:]
        logLs = logLs[rank,:]

        # Keep only top sequences
        orf_tensor = orf_tensor[:max_seqs,:]
        logLs = logLs[:max_seqs,:]

        


        print( orf_tensor.shape, logLs.shape )
        print( get_tensor_size(orf_tensor), get_tensor_size(logLs), get_tensor_size(new_residues) )

        #logLs = output.cpu().detach().numpy()[:, -1, :]

        ## logLs needs to be fixed -- we don't want to go to cpu anymore
        ## Next step should be to see what the exact shape of output is
        ## I probably want to add a batch_size variable at the top of the loop
        ## essentially we're going to want to unfold the results matrix and have
        ## that get added into the new scores vector, that will have to get multiplied
        ## by unfolded dimension (i.e. 64 possible codons). The candidate sequences
        ## need to be checked for penalties. So, we'll have new arrays of candidate sequences
        ## plus their associated logLs. If we have a function that maybe converts codons
        ## into 12-slot one hot arrays (3 x 4 one hot values), then we could apply a reshape
        ## to bring them back into a sequence level array. It should be straight forward I think
        ## to write a function to do this; maybe just keep it into this py file since I'm not sure
        ## we would need/want this functionality elsewhere (at least right now)





    return orf_tensor, logLs











def seq_scores_to_seq_dict(seq_scores):
    '''
    Takes a dictionary of form {dna_sequence : negLL}
    and transforms it to {sequence_description : dna_sequence}
    '''
    return {
        f"seq{i}: negLL: {negLL}": seq
        for i, (seq, negLL) in enumerate(seq_scores.items())
    }


def seq_scorer(model, cds, gen_size=500):

    s_codons = re.findall('...', cds)
    prot = translate(cds)
    prot_len = len(prot)

    logLs = []
    best_codons = []
    best_codon_scores = []
    for i in range(prot_len):
        if i < gen_size:
            prot_seq = prot[: gen_size]
        else:
            prot_seq = prot[i - gen_size + 1: i + 1]
        prot_coded = [RESIDUE_TO_INT[r] for r in prot_seq]
        prot_tensor = torch.tensor([prot_coded]).to(torch.int64)

        if i >= gen_size:
            coded_orf = [[CODON_TO_INT[c] for c in s_codons[i - gen_size:i]]]
        else:
            coded_orf = [[65] + [CODON_TO_INT[c] for c in s_codons[:i]]]
        orf_tensor = torch.Tensor(coded_orf).to(torch.int64)

        weights_tensor = torch.ones(prot_tensor.shape)
        output = model(prot_tensor, orf_tensor).cpu().detach().numpy()[0, -1, :]
        logLs.append(output[CODON_TO_INT[s_codons[i]]])
        best_codons.append(np.argmax(output))
        best_codon_scores.append(output[best_codons[-1]])

    return logLs, best_codons, best_codon_scores
