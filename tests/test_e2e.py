import torch

from collage.defaults import RANDOM_SEED, VALIDATION_FRACTION, HYPERPARAMETERS, TRAINING_PARAMETERS
from collage.fasta import read_fasta, write_fasta
from collage.generator import beam_generator, seq_scores_to_seq_dict
from collage.model import CollageModel
from collage.runner import create_model


def test_end_to_end_training_and_generation(tmp_path, ecoli10_path):
    '''
    Simple test to check that training the model and generating a sequence does not crash.
    Does not check any correctness or peformance.
    Tests on a small file with 10 ecoli sequences.
    Only tests on CPU.
    '''
    ######## Step 1: TEST TRAINING ########

    # Temp location to store output
    prefix = tmp_path / 'test'
    output_file = tmp_path / 'output.fasta'

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

    ######## Step 2: TEST INFERENCE ########

    state_dict = torch.load(expected_pt_file, map_location=torch.device('cpu'))
    model = CollageModel(HYPERPARAMETERS['n_input_tokens'],
                         HYPERPARAMETERS['n_output_tokens'],
                         HYPERPARAMETERS['embed_dimension'],
                         HYPERPARAMETERS['feedforward_dimension'],
                         HYPERPARAMETERS['n_heads'],
                         HYPERPARAMETERS['n_encoder_layers'],
                         HYPERPARAMETERS['n_decoder_layers'],
                         TRAINING_PARAMETERS['dropout_rate'],
                         HYPERPARAMETERS['max_sequence_length'])

    model.load_state_dict(state_dict)

    protein = '.PASTA'

    seq_scores = beam_generator(model, protein)
    seq_dict = seq_scores_to_seq_dict(seq_scores)

    write_fasta(seq_dict, output_file)
    read_seqs = read_fasta(output_file, False)

    assert seq_dict == read_seqs


def test_end_to_end_pretrained_generation(tmp_path, ecoli_model_path):
    '''
    Tests sequence generation against a fully trained model.
    Currently used to validate that model does not deviate
    from the pretrained organism models. May be removed once 
    there is better unit testing, as this should be largely covered
    by the fuller e2e test.
    '''

    output_file = tmp_path / 'output.fasta'

    state_dict = torch.load(ecoli_model_path, map_location=torch.device('cpu'))
    model = CollageModel(HYPERPARAMETERS['n_input_tokens'],
                         HYPERPARAMETERS['n_output_tokens'],
                         HYPERPARAMETERS['embed_dimension'],
                         HYPERPARAMETERS['feedforward_dimension'],
                         HYPERPARAMETERS['n_heads'],
                         HYPERPARAMETERS['n_encoder_layers'],
                         HYPERPARAMETERS['n_decoder_layers'],
                         TRAINING_PARAMETERS['dropout_rate'],
                         HYPERPARAMETERS['max_sequence_length'])
    model.load_state_dict(state_dict)

    protein = '.PASTA'

    seq_scores = beam_generator(model, protein)
    seq_dict = seq_scores_to_seq_dict(seq_scores)

    write_fasta(seq_dict, output_file)
    read_seqs = read_fasta(output_file, False)

    assert seq_dict == read_seqs
