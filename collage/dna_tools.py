

import os, regex, concurrent, pickle, time
import numpy as np

import torch.nn as nn

from collage.utils import translate, orf_to_coded, prot_to_coded
from collage.tensorize import record_slice_to_arrays, tensorize_batch


def reverse_complement( sequence: str ) -> str:
    temp_seq = sequence.upper()
    temp_seq = temp_seq.replace('A','t').replace('T','a').replace('C','g').replace('G','c')
    #temp_seq = temp_seq.replace('K','m').replace('M','k').replace('R','y').replace('Y','r')
    #temp_seq = temp_seq.replace('B','v').replace('V','b').replace('H','d').replace('D','h')
    return temp_seq.upper()[::-1]


def orf_searcher( dna_sequence: str, 
                  min_length: int = 30, 
                  both_strands: bool = False,
                  longest_only: bool = True,
                  sequence_name: str = 'Sequence',
                  dir = None, ) -> list:
    '''
    Generate a list of records describing potential ORFs in a given DNA sequence
    '''

    #print( sequence_name )

    ORF_search = regex.compile('ATG(?:...){2,}?(?:TAG|TAA|TGA|TRA|TAR)')

    records = [ ]
    for strand in range(both_strands+1):
        if strand == 0: # For strand
            sequence = dna_sequence
            strand_label = '+'
        else:           # Rev strand
            sequence = reverse_complement( dna_sequence )
            strand_label = '-'
            
        #if sequence == 303030303030:
        #    print( sequence_name )
        if type( sequence ) != type(''):
            print( sequence, type(sequence), )
        orfs = ORF_search.findall( sequence, overlapped = True, )

        for orf in set( orfs ):
            
            protein = translate( orf )
            if protein.count( '.' ) != 1: continue # Limit to 1 stop codon
            if len( protein ) <= min_length: continue # Ensure # of AAs >= min length

            if strand_label == '+':
                start_pos = dna_sequence.find( orf ) + 1
            else:
                start_pos = dna_sequence.find( reverse_complement( orf ) ) + 1
            orf_length = len( orf )
            end_pos = start_pos + orf_length - 1

            record = { 'Name' : sequence_name,
                       'ORF' : orf,
                       'ORF_coded': orf_to_coded( orf, True, ),
                       'Translation' : protein,
                       'Translation_coded' : prot_to_coded( protein ),
                       'Start_position' : start_pos,
                       'End_position' : end_pos,
                       'Strand' : strand_label,
                       'Length' : len( protein ),
                       'n' : 1,
                       'Codon_weights' : [1.0]*orf_length } # Temporary patch on Codon_weights, will likely delete later
            records.append( record )
            
    # Prune records of internal ORFs
    if longest_only:
        longest_orfs = dict( [ ( ( r[ 'Strand' ], r[ 'End_position' ] ), { 'Length' : 0 } ) for r in records ] )
        for record in records:
            key = ( record[ 'Strand' ], record[ 'End_position' ] )
            #print( record[ 'Length' ], longest_orfs[ key ] )
            if record[ 'Length' ] > longest_orfs[ key ][ 'Length' ]:
                longest_orfs[ key ] = record
        records = list( longest_orfs.values() )

    if not dir:
        return records
    else:
        # Dump the records to the provided directory
        out_filename = sequence_name.replace('/','_').replace('\\','_') + '.pkl'
        out_path = os.path.join( dir, out_filename )
        pickle.dump( records, open( out_path, 'wb', ) )
        return None


def library_to_orf_records( sequence_library: dict,
                            unique_orf_only: bool = False,
                            unique_prot_only: bool = False,
                            min_length: int = 30,
                            both_strands: bool = False,
                            longest_only: bool = True,
                            executor = None, ) -> list:
    ''' 
    Process a DNA library to ORF records
    '''
    
    if unique_prot_only:
        unique_orf_only = True # Force unique ORF identifier before doing unique protein search
    
    if not executor:
        orf_records = [ ]
        for name in sequence_library:
            orf_records += orf_searcher( dna_sequence = sequence_library[ name ],
                                         min_length = min_length,
                                         both_strands = both_strands, 
                                         longest_only = longest_only,
                                         sequence_name = name, )
    else:
        #library_elements = list( sequence_library.items() )
        #orf_records = [ ]
        #library_items = list( sequence_library.items() )
        #for i in range( 0, len( library_items ), executor_batch_size, ):
        #    print( i )
        #    batch = list( library_items[ i : i + executor_batch_size ] )
        #temp_dir = 'tmp' + str( round(time.time()) )
        #os.mkdir( temp_dir )
        futures = [ executor.submit( orf_searcher, 
                                     sequence,
                                     min_length,
                                     both_strands,
                                     longest_only,
                                     name, ) for name, sequence in sequence_library.items() ]
        concurrent.futures.wait( futures )
        #print( 'ORFs processed' )
        #record_files = os.listdir( temp_dir )
        #record_sets = [ pickle.load( open( os.path.join( temp_dir, f ), 'rb', ) ) for f in record_files ]
        #print( 'Files read' )
        #orf_records = [ x for x in record_sets ]
        #orf_records = [ x for f in record_files 
        #                for x in pickle.load( open( os.path.join( temp_dir, f ), 'rb', ) ) ]

        #shutil.rmtree( temp_dir )
            
            #results = [ f.result() for f in futures ]
            #orf_records = [ x for x in results ]
            #print( results )
        orf_records = [ x for f in futures for x in f.result() ]
        
    if unique_orf_only:
        unique_orfs = dict( [ ( r[ 'ORF' ], [ ] ) for r in orf_records ] )
        for r in orf_records:
            unique_orfs[ r['ORF'] ].append( r )
        new_records = [ ]
        for records in unique_orfs.values():
            #print( records )
            total_n = np.sum( [ r[ 'n' ] for r in records ] )
            n_per_strand = dict( [ ( s, np.sum( [ r['Strand'] == s for r in records ] ) ) for s in [ '+', '-' ] ] )
            modal_strand = '-' if n_per_strand[ '-' ] > n_per_strand[ '+' ] else '+'
            #print( modal_strand )
            starts = [ r[ 'Start_position' ] for r in records if r[ 'Strand' ] == modal_strand ]
            start_to_n = dict( [ ( s, starts.count( s ) ) for s in sorted( set( starts ) ) ] )
            start_to_n = dict( sorted( start_to_n.items(), key=lambda x:x[1], reverse=True, ) )
            modal_start = list( start_to_n )[ 0 ]
            #print( modal_start )
            modal_records = [ r for r in records if r[ 'Strand' ] == modal_strand and r[ 'Start_position' ] == modal_start ]
            modal_record = dict( modal_records[ 0 ] )
            modal_record[ 'n' ] = total_n
            new_records.append( modal_record )
        orf_records = new_records
        
    return orf_records
            

        
## Draft batch function for prediciton (which is a bit different from training)





def library_scorer( model: nn.Module,
                    orf_records: list,
                    max_context: int = 500,
                    max_sites_per_batch: int = 2000,
                    device: str = 'cpu', ) -> list:
    '''
    Update ORF records to include CoLLAGE site scores
    '''
    nllloss = nn.NLLLoss( reduction='none', )
    
    # Sort records based on length, longest first, work on copy
    orf_records = sorted( orf_records, key=lambda x:x['Length'], reverse=True, )
    
    # Loop over records to create batches with orf indices
    batches = [ ]
    running_batch = [ ] 
    n_records = len( orf_records )
    for i, record in enumerate( orf_records ):
        orf_records[ i ][ 'CoLLAGE_scores' ] = [ ]
        if record[ 'Length' ] <= max_context:
            # Add elements to running_batch with appropriate padding
            if len( running_batch ) == 0:
                # Adding the first entry, and since records are sorted, 
                # this will always be the longest for the batch
                seq_len_for_batch = record[ 'Length' ]
                n_sequences_per_batch = int( max_sites_per_batch / seq_len_for_batch )
                last_idx_in_batch = min( [ n_records-1, i + n_sequences_per_batch - 1 ] )
            array = record_slice_to_arrays( record, 0, record['Length'], seq_len_for_batch, i )
            running_batch.append( array )
            if i == last_idx_in_batch:
                batches.append( tensorize_batch( running_batch ) )
                running_batch = [ ]
            
        else:
            # Sequence will need to be sequentially processed
            # Number of predictions = Length - max_content + 1
            # E.g. 501 residues with max_content = 500 will take 2 predictions
            # Don't bother adding to running_batch
            
            
            n_predictions = record[ 'Length' ] - max_context + 1
            
            sliced_arrays = [ record_slice_to_arrays( record, s, s+max_context, max_context, i ) 
                              for s in range( n_predictions ) ]
            
            n_slices_per_batch = int( max_sites_per_batch / max_context )
            
            for k in range( 0, len( sliced_arrays ), n_slices_per_batch ):
                batch = sliced_arrays[ k : k + n_slices_per_batch ]
                batches.append( tensorize_batch( batch ) )
                
    # Run CoLLAGE model on the batches
    model.eval()    
    for batch in batches:
        prot, cds, weight, gene = [ x.to( device ) for x in batch ]
        targets = cds[ :, 1: ]
        pred = model( prot, cds[:,:-1], ).transpose( 1, -1, )
        logLs = -nllloss( pred, targets, ).detach().numpy()
        
        orf_idxs = gene.detach().flatten().numpy()
        for orf_idx, logL in zip( orf_idxs, logLs ):
            if len( orf_records[ orf_idx ][ 'CoLLAGE_scores' ] ) == 0:
                orf_records[ orf_idx ][ 'CoLLAGE_scores' ] = list( logL )
            else:
                orf_records[ orf_idx ][ 'CoLLAGE_scores' ].append( logL[-1] )
        
    
    return orf_records
                
            
            
            
            
            
            
            
            
    