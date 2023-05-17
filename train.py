'''
Train CoLLAGE Script
'''
import sys
import argparse

import collage.defaults as constants
from collage.runner import create_model
from collage.defaults import TRAINING_PARAMETERS
import collage.defaults


def parse_args(args: list):
    '''
    Read in arguments
    '''

    parser = argparse.ArgumentParser(usage='Train_CoLLAGE.py [optional arguments] input.fasta output_prefix',
                                     description='Train a CoLLAGE codon likelihood model from a provided FASTA '
                                     'input of coding sequences.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_fasta',
                        type=str,
                        help='Path to input FASTA file')
    parser.add_argument('prefix',
                        type=str,
                        help='Model/log file prefix')
    parser.add_argument('--validation_fraction',
                        type=float,
                        help='Fraction of data to withhold for model evaluation' +
                        '\n\tDefault: ' +
                        format(collage.defaults.VALIDATION_FRACTION, '.2f'),
                        default=collage.defaults.VALIDATION_FRACTION)
    parser.add_argument('--random_seed',
                        type=int,
                        help='Random seed for splitting training vs evaluation data' +
                        '\n\tDefault: ' + str(constants.RANDOM_SEED),
                        default=constants.RANDOM_SEED)
    parser.add_argument('--gpu',
                        action='store_true',
                        help='Use GPU (CUDA) for CoLLAGE training')
    parser.add_argument('--epochs',
                        type=int,
                        help='Number of epochs to train for' +
                             f'\n\tDefault: {TRAINING_PARAMETERS["n_epochs"]}',
                        default=TRAINING_PARAMETERS['n_epochs'])

    return parser.parse_args(args)


def main():
    '''
    Train CoLLAGE
    '''
    args = parse_args(sys.argv[1:])
    create_model(**vars(args))


if __name__ == "__main__":
    main()
    sys.exit()
