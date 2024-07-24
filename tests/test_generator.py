import torch

from collage.generator import exceeds_gc, beam_generator


def test_beam_generator_one_residue_protein(mock_collage_model, mock_lls):
    '''
    Test that the correct sequences are generated for a "protein" that only has a single
    residue: P.
    '''

    # Mock likelihoods across all 65 tokens for P
    P = mock_lls['P']

    # Mock response for the first call to the model
    # Dimensions (1, 1, 65) (1 candidate sequence with 1 residue being predicted across 65 possible tokens)
    lls = torch.tensor([
                        [P]
                       ])

    # Set only a single mock response because the model should only be called once for a sequence length of 1
    mock_collage_model.side_effect = [lls]

    res = beam_generator(mock_collage_model, 'P', '', gen_size=5, max_seqs=5)

    # The valid sequences should be the valid codons for 'P'
    assert set(res.keys()) == set(['CCT', 'CCC', 'CCA', 'CCG'])

def test_beam_generator_two_residue_protein(mock_collage_model, mock_lls):
    '''
    Test that the correct sequences are generated for a "protein" that has two
    residues: PD
    '''

    # Mock likelihoods across all 65 tokens for P and D residues
    P, D = mock_lls['P'], mock_lls['D']

    # Mock response for the first call to the model (predicting P codon)
    # Dimensions (1, 1, 65) (1 candidate sequence with 1 residue being predicted across 65 possible tokens)
    lls1 = torch.tensor([[P]])

    # Mock response for the second call to the model (predicting P and D codon)
    # Dimensions (4, 2, 65) (4 candidate sequence with 2 residues being predicted across 65 possible tokens)
    # There are 4 candidate sequences because there are 4 possible codons for P
    lls2 = torch.tensor([[P, D],
                         [P, D],
                         [P, D],
                         [P, D]])

    # Mock responses for each call to the model
    mock_collage_model.side_effect = [lls1, lls2]

    res = beam_generator(mock_collage_model, 'PD', '', gen_size=10, max_seqs=10)

    # The valid sequences should be the valid codons for 'PD'
    assert set(res.keys()) == set(['CCTGAC', 'CCCGAC', 'CCAGAC', 'CCGGAC',
                                   'CCTGAT', 'CCCGAT', 'CCAGAT', 'CCGGAT'])
    
    # TODO(auberon): Verify weights are properly calculated and worst candidates are evicted


def test_exceeds_gc_not_triggered_on_sequence_with_no_gc():
    seq = 'AT' * 300
    assert not exceeds_gc(seq)


def test_exceeds_gc_triggered_on_sequence_with_only_gc():
    seq = 'GC' * 300
    assert exceeds_gc(seq)


def test_exceeds_gc_triggered_if_impossible_to_extend_to_valid_sequence():
    # With the given sequence it's impossible to get below a fraction of .75
    # for a window size of 4. This exceeds the max_fraction of .66
    seq = 'GCC'
    assert exceeds_gc(seq, window_size=4, fraction=.66)


def test_exceeds_gc_not_triggered_if_possible_to_extend_to_valid_sequence():
    # With the given sequence it is possible to get as low of a fraction of .3
    # for a window size of 10. This does not exceed the max_fraction of .66
    seq = 'GCC'
    assert not exceeds_gc(seq, window_size=10, fraction=.66)


def test_exceeds_gc_not_triggered_with_valid_sequence_matching_window_size():
    seq = 'TAGCAT'
    assert not exceeds_gc(seq, window_size=6, fraction=.5)


def test_exceeds_gc_triggered_with_excessive_sequence_matching_window_size():
    seq = 'CCGCAT'
    assert exceeds_gc(seq, window_size=6, fraction=.5)


def test_exceeds_gc_triggered_with_final_window_violating():
    seq = 'AAATGGG'
    assert exceeds_gc(seq, window_size=3, fraction=.8)


def test_exceeds_gc_triggered_with_middle_window_violating():
    seq = 'AAGGGTA'
    assert exceeds_gc(seq, window_size=3, fraction=.8)


def test_exceeds_gc_triggered_with_first_window_violating():
    seq = 'GGGTAAA'
    assert exceeds_gc(seq, window_size=3, fraction=.8)
