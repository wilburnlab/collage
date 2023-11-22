from pathlib import Path
import pytest
from typing import Dict
from collage.fasta import FileContentsError, read_fasta, to_fasta

from pytest_mock import MockFixture

# TODO(auberon): Better understand first_word and override_alphabet_check in read_fasta to write tests for them
# TODO(auberon): Test write_fasta


@pytest.fixture
def tiny_fasta_data():
    '''A mock FASTA data with 3 artificial DNA sequences'''
    return '''>sp|P00968|CARB_ECOLI
GATTACA
>sp|P02929|TONB_ECOLI
TAGACAT
>sp|P04982|RBSD_ECOLI
GAGAGAG
'''


@pytest.fixture
def tiny_fasta_file(mocker: MockFixture, tiny_fasta_data):
    '''A mock FASTA file with 3 artificial DNA sequences'''
    return mocker.mock_open(read_data=tiny_fasta_data)


@pytest.fixture
def tiny_seq_dict() -> Dict[str, str]:
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
        tiny_fasta_file,
        tiny_seq_dict: Dict[str, str],
        mocker: MockFixture):

    mocker.patch('builtins.open', tiny_fasta_file)
    actual = read_fasta('tiny.fasta', True)
    assert tiny_seq_dict == actual


def test_read_fasta_creates_valid_seq_dict_for_zipped_tiny_data(
        tiny_fasta_file,
        tiny_seq_dict: Dict[str, str],
        mocker: MockFixture):

    mocker.patch('gzip.open', tiny_fasta_file)
    actual = read_fasta('tiny.fasta.gz', True)
    assert tiny_seq_dict == actual


def test_read_fasta_works_with_path_object(
        tiny_fasta_file,
        tiny_seq_dict: Dict[str, str],
        mocker: MockFixture):

    mocker.patch('gzip.open', tiny_fasta_file)
    actual = read_fasta(Path('tiny.fast.gz'), True)
    assert tiny_seq_dict == actual


def test_read_empty_fasta_raises_FileContentsError(mocker: MockFixture):
    empty_file = mocker.mock_open(read_data="")
    mocker.patch('builtins.open', empty_file)

    with pytest.raises(FileContentsError) as e:
        read_fasta('empty.fasta', True)

    assert 'empty.fasta' in str(e.value)


def test_to_fasta_creates_correct_fasta_data_from_seq_dict(tiny_fasta_data, tiny_seq_dict):
    assert to_fasta(tiny_seq_dict) == tiny_fasta_data
