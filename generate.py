'''
Generate CoLLAGE Script
'''
import argparse
from pathlib import Path
import sys

from collage.fasta import read_fasta
from collage.generator import beam_generator
from collage.model import initialize_collage_model


def parse_args(args: list):
    '''
    Read in arguments
    '''

    parser = argparse.ArgumentParser(usage='generate.py [optional arguments] input_path weights_path output_path',
                                     description='Use a trained CoLLAGE model to predict a sequence for a given protein.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_path',
                        type=str,
                        help='Path to input FASTA file')
    parser.add_argument('weights_path',
                        type=str,
                        help='Path to model weights (.pt file)')
    parser.add_argument('output_path',
                        type=str,
                        help='Location to store output')
    parser.add_argument('--beam_size',
                        type=int,
                        default=100,
                        help='Number of beams to use in beam search')
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use GPU (CUDA) for CoLLAGE training')

    return parser.parse_args(args)

# TODO(auberon) Move this inside package?


def predict_and_save(input_path, weights_path, output_path, beam_size, gpu):
    '''
    Uses given model weights to predict the first protein specified in a given FASTA
    and saves the results to a given file location.
    '''
    input_path, weights_path = Path(input_path), Path(weights_path)
    for path in (input_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(f"File '{input_path}' not found")

    proteins = read_fasta(input_path, True)

    # Currently only makes predicitions for a single protein.
    # TODO(auberon): Make predicitions for multiple proteins?
    first_protein = list(proteins.values())[0]
    model = initialize_collage_model(weights_path, gpu)
    predictions = beam_generator(model, first_protein, max_seqs=beam_size)

    with open(output_path, 'w') as file:
        file.write(str(predictions))

    print(predictions)


def main():
    args = parse_args(sys.argv[1:])
    predict_and_save(**vars(args))


if __name__ == "__main__":
    main()
