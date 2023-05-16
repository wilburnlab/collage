
import re
import numpy as np
import torch

from collage.utils import translate
from collage.utils import prot_to_coded
from collage.reference_data import RESIDUE_TO_INT, CODON_TO_INT, CODONS



## Let's write a test beam generator function

current_species = ['Ecoli_K12']

def gc_100_check( seq ):
    # Convert the sequence to GC code
    gc = [ b in ['C','G'] for b in seq ]
    slices = [ gc[ i : i+100 ] for i in range( 0, len(gc), 100, ) ]
    percents = [ np.mean( s ) for s in slices ]
    exceptions = [ p >= 0.65 for p in percents ]
    return np.any( exceptions )
    


def beam_generator( model, prot, pre_sequence='', gen_size=500, max_seqs = 100, ):
    assert len( pre_sequence )%3 == 0, 'Start sequence length is not a multiple of 3'
    
    prot_prefix = translate( pre_sequence )
    prot_prefix_len = len( prot_prefix )
    
    full_prot = translate( pre_sequence ) + prot
    full_prot_len = len( full_prot )
    
    current_seqs = { pre_sequence : 0.0 }
    
    #start_codons = re.findall( '...', start )
    #start_coded = [ 65 ] + [ codon_to_int[ c ] for c in start_codons ]
    #orf_tensor = torch.Tensor( [ start_coded ] ).to( torch.int64 )
    
    for i in range( prot_prefix_len, full_prot_len, ):
        if i < gen_size:
            prot_seq = full_prot[ : gen_size ]
        else:
            prot_seq = full_prot[ i - gen_size + 1 : i + 1 ]
        prot_coded =  prot_to_coded( prot_seq )
        prot_tensor_0 = torch.tensor( [ prot_coded ] ).to( torch.int64 )
        
        coded_seqs = [ ]
        for seq in current_seqs:
            s_codons = re.findall( '...', seq )
            if len( s_codons ) >= gen_size:
                coded_seqs.append( [ CODON_TO_INT[ c ] for c in s_codons[ -gen_size: ] ] )
            else:
                coded_seqs.append( [ 65 ] + [ CODON_TO_INT[ c ] for c in s_codons ] )
        orf_tensor = torch.Tensor( coded_seqs ).to( torch.int64 )
        prot_tensor = prot_tensor_0.repeat( orf_tensor.size(0), 1, )
        
        weights_tensor = torch.ones( prot_tensor.shape )
        output = model( prot_tensor, orf_tensor, )
        
        logLs = output.cpu().detach().numpy()[ :, -1, : ]
        candidate_seqs = { }
        for j, seq in enumerate( current_seqs ):
            codon_logL = dict( [ ( c, l ) for c, l in zip( CODONS, logLs[j] ) if np.isfinite( l ) ] )
            for c in codon_logL:
                if gc_100_check( seq + c ): continue # Avoid >65% GC
                candidate_seqs[ seq + c ] = current_seqs[ seq ] + codon_logL[ c ]
        
        candidate_seqs = dict( sorted( candidate_seqs.items(), key=lambda x:x[1], reverse=True, ) )
        current_seqs = dict( list(candidate_seqs.items())[:max_seqs] )
    
    return current_seqs
    
    
def seq_scorer( model, cds, gen_size=500, ):
    
    s_codons = re.findall( '...', cds )
    prot = translate( cds )
    prot_len = len( prot )
    
    logLs = [ ]
    best_codons = [ ]
    best_codon_scores = [ ]
    for i in range( prot_len ):
        if i < gen_size:
            prot_seq = prot[ : gen_size ]
        else:
            prot_seq = prot[ i - gen_size + 1 : i+1 ]
        prot_coded =  [ RESIDUE_TO_INT[ r ] for r in prot_seq ]
        prot_tensor = torch.tensor( [ prot_coded ] ).to( torch.int64 )
        
        if i >= gen_size:
            coded_orf = [ [ CODON_TO_INT[ c ] for c in s_codons[ i-gen_size:i ] ] ]
        else:
            coded_orf = [ [ 65 ] + [ CODON_TO_INT[ c ] for c in s_codons[:i] ] ]
        orf_tensor = torch.Tensor( coded_orf ).to( torch.int64 )

        weights_tensor = torch.ones( prot_tensor.shape )
        output = model( prot_tensor, orf_tensor, ).cpu().detach().numpy()[ 0, -1, : ]
        logLs.append( output[ CODON_TO_INT[ s_codons[i] ] ] )
        best_codons.append( np.argmax( output ) )
        best_codon_scores.append( output[ best_codons[-1] ] )
        
    return logLs, best_codons, best_codon_scores