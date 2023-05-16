from pathlib import Path

from collage.runner import create_model
from collage.constants import VALIDATION_FRACTION, RANDOM_SEED


def test_end_to_end_training(tmp_path):
    '''
    Simple test to check that training the model does not crash.
    Does not check any correctness or peformance.
    Tests on `tests/data/tiny.fasta`, a small file with 10 ecoli sequences.
    Only tests on CPU.
    '''
    # Temp location to store output
    prefix = tmp_path / 'test'

    # Build the path to the input fasta
    # The input is a small fasta with 10 ecoli sequences
    testing_dir = Path(__file__).parent
    fasta_path = testing_dir / 'data' / 'tiny.fasta'

    # Do a quick train of the model on the CPU with 10 epochs.
    create_model(str(fasta_path),
                 str(prefix),
                 VALIDATION_FRACTION,
                 RANDOM_SEED,
                 False,
                 10)
    
    # Check that the model state was saved to disk
    expected_pt_file = prefix.with_suffix('.pt')
    assert expected_pt_file.exists()
