from pathlib import Path
import pytest


@pytest.fixture
def test_data_path():
    test_data_dir = Path(__file__).parent
    return test_data_dir


@pytest.fixture
def unzipped_tiny_fasta_path(test_data_path):
    return test_data_path / 'tiny.fasta'


@pytest.fixture
def zipped_tiny_fasta_path(test_data_path):
    return test_data_path / 'tiny.fasta.gz'
