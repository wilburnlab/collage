'''
tensorize.py
'''


# Libraries
import re, random
import numpy as np

import torch

from collage.src.reference_data import codons, residues, codon_to_residue, codon_to_int

# Global variables
residue_to_codon_mask = dict( [ ( r, np.asarray( [ codon_to_residue[c] == r for c in codons[1:65] ], 
                                                 'float32', ) ) 
                                for r in residues[1:] ] )

coded_codons = [ codon_to_int[ c ] for c in codons[1:65] ]





def coded_codon_to_log_freq( sequence_dict: dict ) -> dict:
    '''
    Compute the log relative frequency of each codon
    '''

    codon_counts = dict( [ (c,0) for c in codons[1:65] ] )
    for dna_sequence in sequence_dict.values():
        codons_in_gene = re.findall( '...', dna_sequence )
        for c in codons_in_gene: codon_counts[c] += 1
    codon_array = np.array( list( codon_counts.values() ) )
    norm_codon_array = np.zeros( codon_array.shape )
    for r in residue_to_codon_mask:
        residue_counts = codon_array * residue_to_codon_mask[r]
        norm_counts = residue_counts / np.sum(residue_counts)
        norm_codon_array += norm_counts
        
    log_norm_dict = dict( zip( coded_codons, list( np.log( norm_codon_array ) ) ) )
    return log_norm_dict


def split_train_test_data( processed_data: list,
                           test_fraction : float,
                           random_seed : int, ) -> dict:
    '''
    Split processed DNA into train and test sets
    '''

    rng = random.Random( random_seed )
    data_len = len( processed_data )
    shuffled_data = rng.sample( processed_data, k = data_len, )
    split_idx = int( round( test_fraction * data_len ) )
    data_by_mode = { 'train' : shuffled_data[ split_idx : ],
                     'test' : shuffled_data[ : split_idx ], }
    return data_by_mode

'''
def records_to_tensors( records: list,
                        start_idx: int,
                        end_idx: int,
                        max_len: int, ) -> tuple:

    assert end_idx - start_idx <= max_len, 'End index exceeds max length in sequence slice'

    padded_orfs = [ [65] + r['ORF_coded'] + [0]*( max_len - r['Length'] )
                    for r in records ]
    orf_tensor = torch.Tensor( padded_orfs ).to( torch.int64 )

    padded_proteins = [ r['Translation'] + [0]*( max_len - r['Length'] ) 
                        for r in records ]
    protein_tensor = torch.Tensor( padded_proteins ).to( torch.int64 )

    return ( protein_tensor, orf_tensor, )
'''

def record_slice_to_arrays( record: dict,
                            start_idx: int,
                            end_idx: int,
                            max_len: int, ) -> tuple:
    '''
    Reformat a list of processed sequence records (dicts) into sliced and padded arrays
    '''
    
    seq_len = end_idx - start_idx
    assert seq_len <= max_len, 'End index exceeds max length in sequence slice'

    padded_orf = [65] + record['ORF_coded'][ start_idx : end_idx ] + [0]*( max_len - seq_len )
    padded_protein = record['Translation_coded'][ start_idx : end_idx ] + [0]*( max_len - seq_len )
    return ( padded_protein, padded_orf )


def record_to_segment_tensors( record : list,
                               segment_len : int = 20, 
                               randomize_start : bool = True, ) -> list:
    '''
    Split a sequence into segmented regions as tensors for CoLLAGE training
    '''

    n_segments = int( record['Length'] / segment_len )
    if n_segments == 0:
        # Shorter than seq_len
        return [ record_slice_to_arrays( record, 0, record['Length'],  segment_len ) ]
    else:
        n_extra_residues = record['Length'] - n_segments*segment_len
        start_idx = random.randint( 0, int( n_extra_residues / 2.0 ) ) if randomize_start else 0
        array_sets = [ record_slice_to_arrays( record, 
                                               start_idx + i*segment_len, 
                                               start_idx + (i+1)*segment_len,
                                               segment_len, ) 
                       for i in range( n_segments ) ]
        return array_sets

def records_to_batches( records: list,
                        segment_len : int,
                        batch_size : int,
                        randomize_order : bool,
                        randomize_start : bool,
                        by_gene : bool, ) -> list:
    '''
    Records to batches for training
    '''

    array_sets = [ x for record in records 
                   for x in record_to_segment_tensors( record, 
                                                       segment_len, 
                                                       randomize_start, ) ]
    
    
    if randomize_order:
        random.shuffle( array_sets )

    batches = [ ]
    for i in range( 0, len( array_sets ), batch_size ):
        batch = array_sets[ i : i + batch_size ]


        prot_tensor = torch.Tensor( [ x[0] for x in array_sets[ i : i + batch_size ] ] ).to( torch.int64 )
        orf_tensor = torch.Tensor( [ x[1] for x in array_sets[ i : i + batch_size ] ] ).to( torch.int64 )
        #logL_tensor = torch.Tensor( [ x[3] for x in entries[ i : i + batch_size ] ] )
        #genes = [ x[4] for x in entries[ i : i+batch_size] ]
        batches.append( ( prot_tensor, orf_tensor, ) ) #sp_tensor, logL_tensor, genes ) )
    return batches



