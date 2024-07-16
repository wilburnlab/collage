from pathlib import Path
from unittest.mock import MagicMock
from typing import Dict

import pytest
import torch

from collage.model import CollageModel
from collage.reference_data import CODON_MASK_EMBEDDING, INT_TO_RESIDUE


@pytest.fixture
def test_data_path() -> Path:
    '''Path of the directory that holds the test data files'''
    test_data_dir = Path(__file__).parent
    return test_data_dir / 'data'


@pytest.fixture
def ecoli10_path(test_data_path: Path) -> Path:
    '''Path to a small fasta file with 10 ecoli sequences'''
    return test_data_path / 'ecoli10.fasta'


@pytest.fixture
def ecoli_model_path(test_data_path: Path) -> Path:
    return test_data_path / 'Ecoli.pt'


@pytest.fixture
def mock_collage_model() -> MagicMock:
    return MagicMock(spec=CollageModel)


@pytest.fixture
def mock_lls() -> Dict[int, torch.FloatTensor]:
    '''Returns a dictionary of amino acid strings to mock log likelihoods.

    Each value is a tensor of length 66 (64 possible residues + 2 special tags).
    The corresponding index for each impossible codon will be -inf, and each possible
    codon will have an arbitrary (but consistent) LL.'''

    mock_data = {}
    for residue_int, mask in enumerate(CODON_MASK_EMBEDDING):
        # Create weights for each possible codon
        # The first possible codon will have a weight of -1, the second -2...
        # Impossible codons will have a weight of -inf
        weights = []
        curr_weight = -1
        for possible in mask:
            if possible:
                weights.append(curr_weight)
                curr_weight -= 1
            else:
                weights.append(float('-inf'))

        mock_data[INT_TO_RESIDUE[residue_int]] = torch.tensor(weights)

    return mock_data
