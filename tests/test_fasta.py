from pathlib import Path
import pytest
from typing import Dict
from collage.fasta import read_fasta

from pytest_mock import MockFixture


@pytest.fixture
def tiny_fasta_file_contents(mocker: MockFixture):
    '''A mock FASTA file with 3 artificial DNA sequences'''
    return mocker.mock_open(read_data='''>sp|P00968|CARB_ECOLI
GATTACA
>sp|P02929|TONB_ECOLI
TAGACAT
>sp|P04982|RBSD_ECOLI
GAGAGAG
''')


@pytest.fixture
def tiny_fasta_seq_dict() -> Dict[str, str]:
    '''
    A sequence dictionary that corresponds to `tiny_fasta_file_contents`
    Keys are the names, values are the sequences.
    If `tiny_fasta_file_contents` is changed, this fixture should be too.
    '''
    return {
        'sp|P00968|CARB_ECOLI': 'GATTACA',
        'sp|P02929|TONB_ECOLI': 'TAGACAT',
        'sp|P04982|RBSD_ECOLI': 'GAGAGAG'
    }


def test_read_fasta_creates_valid_seq_dict_for_unzipped_tiny_data(
        tiny_fasta_file_contents,
        tiny_fasta_seq_dict: Dict[str, str],
        mocker: MockFixture):

    mocker.patch('builtins.open', tiny_fasta_file_contents)
    actual = read_fasta('tiny.fasta', True)
    assert tiny_fasta_seq_dict == actual


def test_read_fasta_creates_valid_seq_dict_for_zipped_tiny_data(
        tiny_fasta_file_contents,
        tiny_fasta_seq_dict: Dict[str, str],
        mocker: MockFixture):

    mocker.patch('gzip.open', tiny_fasta_file_contents)
    actual = read_fasta('tiny.fasta.gz', True)
    assert tiny_fasta_seq_dict == actual


def test_read_fasta_works_with_path_object(
        tiny_fasta_file_contents,
        tiny_fasta_seq_dict: Dict[str, str],
        mocker: MockFixture):

    mocker.patch('gzip.open', tiny_fasta_file_contents)
    actual = read_fasta(Path('tiny.fast.gz'), True)
    assert tiny_fasta_seq_dict == actual
