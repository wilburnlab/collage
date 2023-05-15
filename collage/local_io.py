'''
Local I/O Functions
'''

# Libraries
import re
import numpy as np, pandas as pd
from collage.utils import identify_alphabet
from collage.reference_data import nucleotides, residues, codons




def read_fasta( file_name : str, 
                first_word : bool ) -> dict: 
    '''
    Return dict with keys = names, values = sequences
    '''

    # Read in sequences
    seq_dict = {}
    for line in open( file_name, 'r' ):
        if line[0] == '>': # New seq
            name = line[1:].rstrip()
            if first_word:
                name = name.split(' ')[0] # Uniprot style
            seq_dict[ name ] = ''
        else:
            seq_dict[ name ] += line.rstrip()

    # Ensure there are sequences in the file
    assert len( seq_dict ) > 0, 'No sequences found in FASTA file: ' + file_name

    # Infer if alphabet is nucleotide or protein
    observed_alphabets = set( [ identify_alphabet( s ) for s in seq_dict.values() ] )
    assert 'Unknown' not in observed_alphabets, 'Unknown characters in FASTA file: ' + file_name
    assert len( observed_alphabets ) == 1, 'Both DNA and Protein sequences in FASTA file: ' + file_name
    alphabet = list( observed_alphabets )[0]

    # Remove sequences with degenerate nucleotides
    #if alphabet == 'DNA':
    #    seq_dict = dict( [ x for x in seq_dict.items() if set( x[1] ) <= set( nucleotides ) ] )
    
    # Return seq dict
    return seq_dict



def write_fasta( seq_dict: dict, 
                 file_name: str,
                 append: bool = True ) -> None:
    '''
    Write a sequence dictionary to FASTA file, defaults to append rather than overwrite.
    In append mode, will create a new file first if it doesn't already exist
    '''

    if append == False: # overwrite mode
        write_mode = 'w'
    else: # append mode
        write_mode = 'a' if os.path.isfile( file_name ) else 'w'

    file_out = open( file_name, write_mode, )

    for name, seq in seq_dict.items():
        file_out.write( '>' + name + '\n' + seq + '\n' )

    file_out.close()
    return None

