import re
import time
import datetime
import math
import numpy as np

from collage.reference_data import NUCLEOTIDES, RESIDUES, CODONS, CODON_TO_RESIDUE, NUCLEOTIDE_TO_INT, CODON_TO_INT, RESIDUE_TO_INT, RESIDUE_TO_CODON_MASK

CODED_CODONS = [CODON_TO_INT[c] for c in CODONS[1:65]]


def identify_alphabet(sequence: str) -> str:
    '''
    Return the likely alphabet of a sequence as DNA, Protein, or Unknown
    '''

    characters = set(sequence)
    dna_check = characters <= set(NUCLEOTIDES)
    protein_check = characters <= set(RESIDUES)

    if dna_check:
        return 'DNA'
    elif protein_check:
        return 'Protein'
    else:
        return 'Unknown'


def translate(sequence: str) -> str:
    '''
    Translate DNA sequence to protein
    '''

    observed_alphabet = identify_alphabet(sequence)
    assert observed_alphabet == 'DNA', 'Attempted to translate non-DNA sequence: ' + sequence

    dna_sequence = sequence.strip().upper()
    if len(dna_sequence) < 3:
        return ''

    obs_codons = re.findall('...', dna_sequence)
    obs_residues = [CODON_TO_RESIDUE.get(c, 'X') for c in obs_codons]
    return ''.join(obs_residues)


def orf_check(prot: str) -> bool:
    '''
    Check if a protein sequence is a perfect start->stop with no ambiguous characters
    '''
    return prot[0] == 'M' and prot[-1] == '.' and prot.count('.') == 1 and prot.count('X') == 0


def len_check(sequence: str,
              min_len: int,
              max_len: int) -> bool:
    '''
    Check if a sequence is within a given min/max len
    '''
    seq_len = len(sequence)
    return seq_len >= min_len and seq_len <= max_len

def nt_to_coded(orf: str) -> list:
    return [NUCLEOTIDE_TO_INT[n] for n in orf]


def orf_to_coded(orf: str,
                 add_start: bool = False) -> list:
    codons = re.findall('...', orf)
    coded = [CODON_TO_INT[c] for c in codons]
    if add_start:
        coded = [65] + coded
    return coded


def prot_to_coded(prot: str) -> list:
    return [RESIDUE_TO_INT[r] for r in prot]


def codon_counts_in_library(sequence_dict: dict) -> np.ndarray:
    '''
    Compute the log relative frequency of each codon
    '''

    codon_counts = dict([(c, 0) for c in CODONS[1:65]])
    for dna_sequence in sequence_dict.values():
        codons_in_gene = re.findall('...', dna_sequence)
        for c in codons_in_gene:
            codon_counts[c] += 1
    codon_array = np.array(list(codon_counts.values()))
    # codon_dict =  dict( zip( coded_codons, list( codon_array ) ) )
    return codon_array


def calc_codon_weights(sequence_dict: dict,
                       beta_factor: float = 4.0,
                       aa_normalize: bool = False) -> dict:
    codon_array = codon_counts_in_library(sequence_dict)
    if aa_normalize:
        norm_codon_array = np.zeros(codon_array.shape)
        for r in RESIDUE_TO_CODON_MASK:
            residue_counts = codon_array * RESIDUE_TO_CODON_MASK[r]
            norm_counts = residue_counts / np.sum(residue_counts)
            norm_codon_array += norm_counts
        # codon_array =
        # NEED TO INCORPORATE SOME TYPE OF NORMALIZER HERE THAT ACCOUNTS FOR THE # OF EACH AA THAT'S PRESENT
        # I SHOULD BE ABLE TO DO THIS FAIRLY SIMPLY USING THE RESIDUE_TO_CODON_MASK FEATURE TO GET THE RELATIVE
        # ABUNDANCE OF THE AA
    beta = 1 - math.pow(10.0, -beta_factor)
    effective_n = (1 - np.power(beta, codon_array)) / (1 - beta)
    weight_array = np.mean(effective_n) / effective_n
    weight_array = np.ones(weight_array.shape)
    weight_dict = dict(zip(CODED_CODONS, list(weight_array)))
    return weight_dict


def codedorf_to_weights(orf_coded: list,
                        weight_dict: dict) -> list:
    return [weight_dict[c] for c in orf_coded]


def calc_null_codon_logL(sequence_dict: dict) -> dict:
    codon_array = codon_counts_in_library(sequence_dict)
    norm_codon_array = np.zeros(codon_array.shape)
    for r in RESIDUE_TO_CODON_MASK:
        residue_counts = codon_array * RESIDUE_TO_CODON_MASK[r]
        norm_counts = residue_counts / np.sum(residue_counts)
        norm_codon_array += norm_counts
    norm_codon_dict = dict(zip(CODED_CODONS, list(np.log(norm_codon_array))))
    return norm_codon_dict


def dna_dictionary_to_records(dna_dict: dict,
                              min_aa: int = 30,
                              max_aa: int = 2000) -> list:
    '''
    Process DNA dictionary into a list of dictionaries with processed data
    '''

    codon_weight_dict = calc_codon_weights(dna_dict)
    norm_codon_dict = calc_null_codon_logL(dna_dict)

    records = []
    for seq_name in dna_dict:
        record = {'Ensembl_ID': seq_name,
                  'ORF': dna_dict[seq_name], }
        record['NT_coded'] = nt_to_coded(record['ORF'])
        record['ORF_coded'] = orf_to_coded(record['ORF'], True)
        prot = translate(record['ORF'])
        if orf_check(prot) and len_check(prot, min_aa, max_aa):
            record['Translation'] = prot
            record['Length'] = len(prot)
            record['Translation_coded'] = prot_to_coded(prot)
            record['Codon_weights'] = codedorf_to_weights(record['ORF_coded'][1:], norm_codon_dict)  # Remove start
            records.append(record)
    return records
    # df = pd.DataFrame.from_records( records )
    # return df


def timer(start):
    return str(datetime.timedelta(seconds=round(time.time() - start)))


# def actg_check( seq ):
#    return set(seq) == set( ['A','T','C','G'] )


'''
def dna_to_prot_dict( dna_dict, min_aa = 30, max_aa = 2000):
    prot_dict = { }
    for seq_name in dna_dict:
        prot = translate( dna_dict[seq_name] )
        if orf_check( prot ) and prot_size_check( prot, min_aa, max_aa ):
            prot_dict[ seq_name ] = prot
    return prot_dict


def seqdict_to_df( dna_dict, min_aa = 30, max_aa = 2000):
    rows = [ ]
    for seq_name in dna_dict:
        prot = translate( dna_dict[seq_name] )
        if orf_check( prot ) and prot_size_check( prot, min_aa, max_aa ):
            rows.append( [ seq_name, dna_dict[seq_name], prot, len(prot) ] )

    df = pd.DataFrame( rows, columns=[ 'Ensembl_ID', 'ORF', 'Translation', 'Translation_length', ])
    return df

def orf_to_coded( orf ):
    codons = re.findall( '...', orf )
    coded = [ codon_to_int[c] for c in codons ]
    return coded

def prot_to_coded( prot ):
    return [ residue_to_int[r] for r in prot ]

def nt_to_coded( orf ):
    return [ nt_to_int[n] for n in orf ]
'''
