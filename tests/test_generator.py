from collage.generator import exceeds_gc


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
