'''
Reference data
'''

nucleotides = [ '-', 'A', 'T', 'C', 'G', ]

codon_to_residue = { '---':'-',
                     'TTT':'F', 'TCT':'S', 'TAT':'Y', 'TGT':'C',
                     'TTC':'F', 'TCC':'S', 'TAC':'Y', 'TGC':'C',
                     'TTA':'L', 'TCA':'S', 'TAA':'.', 'TGA':'.',
                     'TTG':'L', 'TCG':'S', 'TAG':'.', 'TGG':'W',
                     'CTT':'L', 'CCT':'P', 'CAT':'H', 'CGT':'R',
                     'CTC':'L', 'CCC':'P', 'CAC':'H', 'CGC':'R',
                     'CTA':'L', 'CCA':'P', 'CAA':'Q', 'CGA':'R',
                     'CTG':'L', 'CCG':'P', 'CAG':'Q', 'CGG':'R',
                     'ATT':'I', 'ACT':'T', 'AAT':'N', 'AGT':'S',
                     'ATC':'I', 'ACC':'T', 'AAC':'N', 'AGC':'S',
                     'ATA':'I', 'ACA':'T', 'AAA':'K', 'AGA':'R',
                     'ATG':'M', 'ACG':'T', 'AAG':'K', 'AGG':'R',
                     'GTT':'V', 'GCT':'A', 'GAT':'D', 'GGT':'G',
                     'GTC':'V', 'GCC':'A', 'GAC':'D', 'GGC':'G',
                     'GTA':'V', 'GCA':'A', 'GAA':'E', 'GGA':'G',
                     'GTG':'V', 'GCG':'A', 'GAG':'E', 'GGG':'G',
                     '***':'*',
                   }

codons = list( codon_to_residue )
codon_to_int = dict( zip( codons, range( len(codons) ) ) )
int_to_codon = dict( [ x[::-1] for x in codon_to_int.items() ] )


residues = [ '-',
             'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 
             '.', ]
residue_to_int = dict( zip( residues, range( len(residues) ) ) )
int_to_residue = dict( [ x[::-1] for x in residue_to_int.items() ] )


