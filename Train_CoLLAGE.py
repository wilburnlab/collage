'''
Train CoLLAGE Script
'''

# Libraries
import sys, time, argparse, time, concurrent

import torch

import collage.constants as constants
from collage.local_io import read_fasta
from collage.utils import dna_dictionary_to_records, timer
from collage.tensorize import split_train_test_data
from collage.training_code import train_loop
from collage.eval_functions import Relative_Codon_Likelihood

from collage.model import Codon_Predictor










def parse_args( args: list ):
    '''
    Read in arguments
    '''

    parser = argparse.ArgumentParser( usage = 'Generate_Library.py [optional arguments] input.fasta output.dlib',
                                      description = 'Generate EncyclopeDIA-compatible library from FASTA input'
                                                    '\n\nModifications should be supplied as X=M where X is a '
                                                    'single letter amino acid code and M is a specified '
                                                    'modification. Currently supported modifications:'
                                                    '\n\tC=Cam       Carbamidomethylation (Cys)'
                                                    '\n\tM=Ox        Oxidation (Met)'
                                                    '\n\tS=Phospho   Phosphorylation (Ser)'
                                                    '\n\tT=Phospho   Phosphorylation (Thr)'
                                                    '\n\tY=Phospho   Phosphorylation (Tyr)'
                                                    '\n\tK=Ub        Ubiquitination (Lys)',
                                     formatter_class = argparse.RawTextHelpFormatter, )
    parser.add_argument( 'input_fasta',
                         type = str,
                         help = 'Path to input FASTA file')
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

    return parser.parse_args(args)




def main():
    '''
    Train CoLLAGE
    '''


    start_time = time.time()
    args = parse_args( sys.argv[1:] )

    sequences = read_fasta( args.input_fasta, first_word=True, )
    n_seq = len( sequences )
    print( str( n_seq ) + ' sequences loaded from ' + args.input_fasta + '(' + timer(start_time) + ')' )
    processed_sequences = dna_dictionary_to_records( sequences )
    n_proc = len( processed_sequences )
    print( str( n_proc ) + ' processed for training (' + timer(start_time) + ')' )
    data_by_mode = split_train_test_data( processed_sequences, 
                                          args.validation_fraction, 
                                          args.random_seed )

    ## CLEAN THIS UP LATER!
    
    output_file = 'test'
    #scheduler = Scheduler( opt, 1e-3, 512, int(1e5),  )
    loss_fx = Relative_Codon_Likelihood(  )

    model = Codon_Predictor( n_input_tokens = 22, 
                         n_output_tokens = 66,
                         n_sp = 1, #len( species ),
                         model_dim = 64, 
                         ff_dim = 256,
                         n_heads = 4,
                         n_encoder_layers = 1, 
                         n_decoder_layers = 2, 
                         dropout = 0.2,
                         max_len = 500, )
    opt = torch.optim.SGD( model.parameters(), lr=1e-1, ) ## CLEAN THIS UP LATER
    train_loop( model = model, 
                optimizer = opt, 
                loss_fx = loss_fx, 
                data_by_mode = data_by_mode, 
                output_name = output_file, 
                start_time = start_time, 
                epochs = 250, 
                epochs_to_2x_batch = 50,
                device = 'cuda', 
                train_batch_size = 4, 
                initial_seq_len = 30, 
                scheduler = None, )

    #train_loop( model, opt, loss_fn, data_by_mode, output_file, start_time, epochs=250, device='cpu', train_batch_size=4, seq_len=30, scheduler=None, ) # train_seq_len=32, )





if __name__ == "__main__":
    main()
    sys.exit()


