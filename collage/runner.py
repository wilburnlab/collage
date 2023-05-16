import time
from collage.local_io import read_fasta
from collage.training_code import train_collage

from collage.utils import dna_dictionary_to_records, timer


def create_model(input_fasta: str,
                 prefix: str,
                 validation_fraction: float,
                 random_seed: int,
                 gpu: bool,
                 epochs: int):
    '''
    Parses an input fasta, trains a CollageModel, and saves it to the filesystem.
    '''

    start_time = time.time()
    sequences = read_fasta(input_fasta, first_word=True)
    n_seq = len(sequences)
    print(str(n_seq) + ' sequences loaded from ' +
          input_fasta + ' (' + timer(start_time) + ')')
    processed_sequences = dna_dictionary_to_records(sequences)
    n_proc = len(processed_sequences)
    print(str(n_proc) + ' processed for training (' + timer(start_time) + ')')

    train_collage(output_name=prefix,
                  training_data=processed_sequences,
                  test_frac=validation_fraction,
                  random_seed=random_seed,
                  start_time=start_time,
                  device='cuda' if gpu else 'cpu',
                  epochs=epochs)
