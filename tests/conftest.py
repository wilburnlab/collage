from pathlib import Path
import pytest


@pytest.fixture
def test_data_path() -> Path:
    '''Path of the directory that holds the test data files'''
    test_data_dir = Path(__file__).parent
    return test_data_dir / 'data'


@pytest.fixture
def unzipped_tiny_fasta_path(test_data_path: Path) -> Path:
    '''Path to a small fasta file with 10 ecoli sequences'''
    return test_data_path / 'tiny.fasta'


@pytest.fixture
def zipped_tiny_fasta_path(test_data_path: Path) -> Path:
    '''Path to a gzipped small fasta file with 10 ecoli sequences'''
    return test_data_path / 'tiny.fasta.gz'
