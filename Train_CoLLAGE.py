'''
Train CoLLAGE Script
'''

# Libraries
import sys, time, argparse, time, concurrent

import collage.constants as constants
from collage.local_io import read_fasta
from collage.utils import dna_dictionary_to_records, timer
from collage.training_code import train_collage


def parse_args( args: list ):
    '''
    Read in arguments
    '''

    parser = argparse.ArgumentParser( usage = 'Train_CoLLAGE.py [optional arguments] input.fasta output_prefix',
                                      description = 'Train a CoLLAGE codon likelihood model from a provided FASTA ' 
                                                    'input of coding sequences.',
                                      formatter_class = argparse.RawTextHelpFormatter, )
    parser.add_argument( 'input_fasta',
                         type = str,
                         help = 'Path to input FASTA file')
    parser.add_argument( 'prefix',
                         type = str,
                         help = 'Model/log file prefix')
    parser.add_argument( '--validation_fraction',
                         type = float,
                         help = 'Fraction of data to withhold for model evaluation' +
                                '\n\tDefault: ' + format( constants.validation_fraction,'.2f' ),
                         default = constants.validation_fraction, )
    parser.add_argument( '--random_seed',
                         type = int,
                         help = 'Random seed for splitting training vs evaluation data' +
                                '\n\tDefault: ' + str(constants.random_seed),
                         default = constants.random_seed, )
    parser.add_argument( '--gpu',
                         action = 'store_true',
                         help = 'Use GPU (CUDA) for CoLLAGE training', )

    return parser.parse_args(args)


def main():
    '''
    Train CoLLAGE
    '''
    start_time = time.time()
    args = parse_args( sys.argv[1:] )

    sequences = read_fasta( args.input_fasta, first_word=True, )
    n_seq = len( sequences )
    print( str( n_seq ) + ' sequences loaded from ' + args.input_fasta + ' (' + timer(start_time) + ')' )
    processed_sequences = dna_dictionary_to_records( sequences )
    n_proc = len( processed_sequences )
    print( str( n_proc ) + ' processed for training (' + timer(start_time) + ')' )

    train_collage( output_name = args.prefix, 
                   training_data = processed_sequences, 
                   test_frac = args.validation_fraction, 
                   random_seed = args.random_seed, 
                   start_time = start_time, 
                   device = 'cuda' if args.gpu else 'cpu', )


if __name__ == "__main__":
    main()
    sys.exit()


