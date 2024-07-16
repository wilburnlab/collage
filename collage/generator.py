
from math import ceil
from collections.abc import Iterable
import re

import numpy as np
import torch

from collage.utils import translate
from collage.utils import prot_to_coded
from collage.reference_data import RESIDUE_TO_INT, CODON_TO_INT, CODONS


def exceeds_gc(seq: Iterable[str], window_size: int = 100, fraction: float = .65):
    '''Checks whether the GC fraction heuristic is violated in a sequence.

    This function returns True if any subsequence of the given size (window_size) within
    the input sequence (seq) has a GC content at or above the specified
    maximum fraction (fraction). 

    Parameters:
    - seq (Iterable[str]): The input sequence of nucleotide bases (strings 'A', 'T', 'G', 'C').
    - window_size (int, optional): The size of the sliding window to check for GC content. 
      Defaults to 100.
    - fraction (float, optional): The fraction of G or C bases within any window at or above which
      will be considered excessive. Defaults to 0.65.

    Returns:
    - bool: True if any subsequence of the specified window_size has a GC fraction greater
      than or equal to fraction, otherwise False.

    Special Case:
    If the sequence length is less than the window_size, the function checks if it is 
    impossible for the sequence to meet the constraint even when padded. For example,
    with a window_size of 6 and a fraction of 0.5, the sequence 'GCGC' would return 
    True because regardless of the additional two bases appended it cannot meet the 
    required fraction for the window size.

    This heuristic is intended to flag sequences that may be challenging to synthesize
    due to high GC content.

    '''
    max_count = ceil(window_size * fraction)
    count = 0

    for i in range(len(seq)):
        count += int(seq[i] in 'GC')
        if i >= window_size:
            count -= int(seq[i - window_size] in 'GC')
        if count >= max_count:
            return True

    return False


def beam_generator(model, prot, pre_sequence='', gen_size=500, max_seqs=100, check_gc=False):
    if len(pre_sequence) % 3 != 0:
        raise ValueError('Start sequence length is not a multiple of 3')

    # The prefix sequence translated to a protein
    prot_prefix = translate(pre_sequence)
    prot_prefix_len = len(prot_prefix)

    # The prefix sequence along with the protein to predict
    full_prot = translate(pre_sequence) + prot
    full_prot_len = len(full_prot)

    # Holds the current candidate sequences as DNA base strings
    # Begins with the inial prefix only
    current_seqs = {pre_sequence: 0.0}

    # For each of the codons to predict (do not predict codons for the pre-sequence)
    for i in range(prot_prefix_len, full_prot_len):
        # Predict no more than gen_size codons at a time
        if i < gen_size:
            prot_seq = full_prot[: gen_size]
        else:
            prot_seq = full_prot[i - gen_size + 1: i + 1]

        # Convert sequence to integer code
        # TODO(auberon): This can just be done once at the beginning
        prot_coded = prot_to_coded(prot_seq)
        prot_tensor_0 = torch.tensor([prot_coded]).to(torch.int64)

        coded_seqs = []
        for seq in current_seqs:
            # Split sequence into groups of 3, discarding any trailing 1 or two bases
            s_codons = re.findall('...', seq)
            # If there are more codons than the gen size, only include the last gen_size of them
            if len(s_codons) >= gen_size:
                coded_seqs.append([CODON_TO_INT[c]
                                  for c in s_codons[-gen_size:]])
            # Otherwise include the whole thing with a start of sequence symbol prepended
            else:
                coded_seqs.append([65] + [CODON_TO_INT[c] for c in s_codons])

        # Create a [num_seq, seq_len] tensor with the current codons for each sequence
        orf_tensor = torch.Tensor(coded_seqs).to(torch.int64)
        # Create a [num_seq, min(gen_size, num_total_residues)] tensor
        # Contains the residues repeated
        prot_tensor = prot_tensor_0.repeat(orf_tensor.size(0), 1)

        # weights_tensor = torch.ones(prot_tensor.shape)
        output = model(prot_tensor, orf_tensor)

        logLs = output.cpu().detach().numpy()[:, -1, :]
        candidate_seqs = {}
        for j, seq in enumerate(current_seqs):
            codon_logL = dict([(c, l) for c, l in zip(
                CODONS, logLs[j]) if np.isfinite(l)])
            for c in codon_logL:
                if check_gc and exceeds_gc(seq + c):
                    continue  # Avoid >65% GC
                candidate_seqs[seq + c] = current_seqs[seq] + codon_logL[c]

        candidate_seqs = dict(sorted(candidate_seqs.items(), key=lambda x: x[1], reverse=True))
        current_seqs = dict(list(candidate_seqs.items())[:max_seqs])

    return current_seqs


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
