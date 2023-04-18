import re, time, datetime
import numpy as np, pandas as pd
#from reference_data import Codon_to_AA, codon_to_int, residue_to_int, nt_to_int

from collage.src.reference_data import nucleotides, residues, codon_to_residue, codon_to_int, residue_to_int


def identify_alphabet( sequence : str ) -> str:
    '''
    Return the likely alphabet of a sequence as DNA, Protein, or Unknown
    '''

    characters = set( sequence )
    dna_check = characters  <= set( nucleotides )
    protein_check = characters <= set( residues )

    if dna_check:
        return 'DNA'
    elif protein_check:
        return 'Protein'
    else:
        return 'Unknown'


def translate( sequence: str ) -> str:
    '''
    Translate DNA sequence to protein
    '''

    observed_alphabet = identify_alphabet( sequence )
    assert observed_alphabet == 'DNA', 'Attempted to translate non-DNA sequence: ' + sequence

    dna_sequence = sequence.strip().upper()
    if len( dna_sequence ) < 3: return ''

    obs_codons = re.findall( '...', dna_sequence )
    obs_residues = [ codon_to_residue.get( c, 'X' ) for c in obs_codons ]
    return ''.join( obs_residues )


def orf_check( prot: str ) -> bool:
    '''
    Check if a protein sequence is a perfect start->stop with no ambiguous characters
    '''
    return prot[0] == 'M' and prot[-1] == '.' and prot.count('.') == 1 and prot.count('X') == 0


def len_check( sequence: str, 
                min_len: int, 
                max_len: int, ) -> bool:
    '''
    Check if a sequence is within a given min/max len
    '''
    seq_len = len( sequence )
    return seq_len >= min_len and seq_len <= max_len


def orf_to_coded( orf ):
    codons = re.findall( '...', orf )
    coded = [ codon_to_int[c] for c in codons ]
    return coded


def prot_to_coded( prot ):
    return [ residue_to_int[r] for r in prot ]


def dna_dictionary_to_records( dna_dict: dict, 
                               min_aa: int = 30, 
                               max_aa: int = 2000, ) -> list:
    '''
    Process DNA dictionary into a list of dictionaries with processed data
    '''

    records = [ ]
    for seq_name in dna_dict:
        record = { 'Ensembl_ID' : seq_name,
                   'ORF' : dna_dict[ seq_name ], }
        record[ 'ORF_coded' ] = orf_to_coded( record[ 'ORF' ] )
        prot = translate( record[ 'ORF' ] )
        if orf_check( prot ) and len_check( prot, min_aa, max_aa ):
            record[ 'Translation' ] = prot
            record[ 'Length' ] = len( prot )
            record[ 'Translation_coded' ] = prot_to_coded( prot )
            records.append( record )
    return records
    #df = pd.DataFrame.from_records( records )
    #return df



def timer( start, ):
    return str( datetime.timedelta( seconds=round( time.time() - start ) ) )




#def actg_check( seq ):
#    return set(seq) == set( ['A','T','C','G'] )






'''
def dna_to_prot_dict( dna_dict, min_aa = 30, max_aa = 2000, ):
    prot_dict = { }
    for seq_name in dna_dict:
        prot = translate( dna_dict[seq_name] )
        if orf_check( prot ) and prot_size_check( prot, min_aa, max_aa ):
            prot_dict[ seq_name ] = prot
    return prot_dict


def seqdict_to_df( dna_dict, min_aa = 30, max_aa = 2000, ):
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

