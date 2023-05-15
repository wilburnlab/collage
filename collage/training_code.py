'''
Description
'''


# Libraries
import random, math
import numpy as np
import torch

from collage.utils import timer
from collage.tensorize import records_to_batches



def train_loop( model, 
                optimizer, 
                loss_fx, 
                data_by_mode : dict, 
                output_name : str, 
                start_time : float, 
                epochs : int = 20, 
                epochs_to_2x_batch : int = 50,
                device : str ='cpu', 
                train_batch_size : int = 6, 
                initial_seq_len : int = 30, 
                scheduler = None, ):
    
    # Create log file
    log_file = open( output_name+'.log', 'w' )
    log_file.close()

    model.to( device )
    loss_fx.to( device )
    
    best_epoch = -1
    best_score = 1e8

    for epoch in range( epochs ):
        print( '----------\nEpoch ' + str(epoch+1) + ' of ' + str(epochs) + '\n----------')

        log_file = open( output_name+'.log', 'a' )
        log_file.write( 'Epoch ' + str(epoch+1) + '\n' ) 
        log_file.close()

        for mode in data_by_mode:
            if mode == 'train':
                model.train()
            else:
                model.eval()

            total_loss = 0

            seq_len = int( initial_seq_len * math.pow( 2.0, epoch / epochs_to_2x_batch ) )
            seq_len = min( [ seq_len, 500, ] )

            batches = records_to_batches( data_by_mode[ mode ],
                                          segment_len = seq_len,
                                          batch_size = train_batch_size if mode == 'train' else 32,
                                          randomize_order = mode == 'train',
                                          randomize_start = mode == 'train',
                                          by_gene = False, )

            total_n = 0
            counter = 0
            for i, batch in enumerate( batches ):
                prot, cds = [ x.to( device ) for x in batch ] #, sp, null
                n = prot.size(0)
                pred = model( prot, cds[:,:-1], ) #sp )
                loss = loss_fx( pred, cds[ :, 1: ] )
                
                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                total_loss += loss.detach().item() * n
                total_n += n
                counter += n
            score = total_loss / total_n
            log_file = open( output_name+'.log', 'a' )
            log_file.write( '\t' + mode.capitalize() + ' score: ' + format( score, '.5f' ) + '\n'  ) 
            log_file.close()
            print( mode.capitalize() + format( score, '.4f' ).rjust(8) )
            if mode == 'test':
                if score < best_score:
                    print( 'New best weights! Copying and saving model to\n\t' + output_name + '.pt' )
                    best_epoch = epoch
                    best_score = score
                    torch.save( model.state_dict(), output_name + '.pt' )
                    epochs_wo_improv = 0
                else:
                    print( 'Did not improve, best performance was epoch ' + 
                           str(best_epoch) + ' (' + format(best_score,'.4f') + ')' )
        print( 'Runtime: ' + timer( start_time ) + '\n' )
    return score


