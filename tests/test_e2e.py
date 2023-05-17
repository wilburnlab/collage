from collage.runner import create_model
from collage.defaults import RANDOM_SEED, VALIDATION_FRACTION


def test_end_to_end_training(tmp_path, ecoli10_path):
    '''
    Simple test to check that training the model does not crash.
    Does not check any correctness or peformance.
    Tests on a small file with 10 ecoli sequences.
    Only tests on CPU.
    '''
    # Temp location to store output
    prefix = tmp_path / 'test'

    # Do a quick train of the model on the CPU with 10 epochs.
    create_model(str(ecoli10_path),
                 str(prefix),
                 VALIDATION_FRACTION,
                 RANDOM_SEED,
                 False,
                 10)

    # Check that the model state was saved to disk
    expected_pt_file = prefix.with_suffix('.pt')
    assert expected_pt_file.exists()
