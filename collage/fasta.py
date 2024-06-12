'''
Local I/O Functions
'''
import gzip
from pathlib import Path
from typing import TextIO

from collage.utils import identify_alphabet


class FileContentsError(RuntimeError):
    pass


def read_fasta(file_name: str | Path,
               first_word: bool,
               override_alphabet_check: bool = False) -> dict:
    '''
    Return dict with keys = names, values = sequences
    '''

    file_name = Path(file_name)
    file_opener = gzip.open if file_name.suffix == '.gz' else open
    with file_opener(file_name, 'rt') as fasta:
        seq_dict = parse_fasta(fasta, first_word)

    # Ensure there are sequences in the file
    if not seq_dict:
        raise FileContentsError(f'No sequences found in FASTA file: "{file_name}"')

    validate_seq_dict(seq_dict, override_alphabet_check)

    return seq_dict


def validate_seq_dict(seq_dict, override_alphabet_check: bool = False):
    # Infer if alphabet is nucleotide or protein
    if not override_alphabet_check:
        observed_alphabets = set([identify_alphabet(s)
                                 for s in seq_dict.values()])

        if 'Unknown' in observed_alphabets:
            raise FileContentsError('Unknown characters in FASTA file')

        if len(observed_alphabets) != 1:
            raise FileContentsError('Both DNA and Protein sequences in FASTA file')

        alphabet = list(observed_alphabets)[0]

    # TODO(auberon): Check with Damien whether this can be
    # Remove sequences with degenerate nucleotides
    # if alphabet == 'DNA':
    #    seq_dict = dict( [ x for x in seq_dict.items() if set( x[1] ) <= set( nucleotides ) ] )


def parse_fasta(file_data: TextIO, first_word: bool):
    seq_dict = {}

    for line in file_data:
        if line[0] == '>':  # New seq
            name = line[1:].rstrip()
            if first_word:
                name = name.split(' ')[0]  # Uniprot style
            seq_dict[name] = ''
        else:
            seq_dict[name] += line.rstrip()

    return seq_dict


def to_fasta(seq_dict: dict) -> str:
    '''
    Converts a sequence dictionary to a string in the FASTA format.
    Lines are separated with UNIX-stye line endings and a trailing newline
    is included.
    '''
    return '\n'.join(f'>{name}\n{seq}' for name, seq in seq_dict.items()) + '\n'


def write_fasta(seq_dict: dict,
                file_name: Path | str,
                append: bool = True) -> None:
    '''
    Write a sequence dictionary to FASTA file, defaults to append rather than overwrite.
    In append mode, will create a new file first if it doesn't already exist
    '''

    write_mode = 'a+' if append else 'w'

    with open(file_name, write_mode) as f:
        f.write(to_fasta(seq_dict))
