import random

import torch


def split_train_test_data(processed_data: list,
                          test_fraction: float,
                          random_seed: int) -> dict:
    '''
    Split processed DNA into train and test sets
    '''

    rng = random.Random(random_seed)
    data_len = len(processed_data)
    shuffled_data = rng.sample(processed_data, k=data_len)
    split_idx = int(round(test_fraction * data_len))
    data_by_mode = {'train': shuffled_data[split_idx:],
                    'test': shuffled_data[: split_idx]}
    return data_by_mode


'''
def records_to_tensors( records: list,
                        start_idx: int,
                        end_idx: int,
                        max_len: int, ) -> tuple:

    assert end_idx - start_idx <= max_len, 'End index exceeds max length in sequence slice'

    padded_orfs = [ [65] + r['ORF_coded'] + [0]*( max_len - r['Length'] )
                    for r in records ]
    orf_tensor = torch.Tensor( padded_orfs ).to( torch.int64 )

    padded_proteins = [ r['Translation'] + [0]*( max_len - r['Length'] ) 
                        for r in records ]
    protein_tensor = torch.Tensor( padded_proteins ).to( torch.int64 )

    return ( protein_tensor, orf_tensor, )
'''


def record_slice_to_arrays(record: dict,
                           start_idx: int,
                           end_idx: int,
                           max_len: int,
                           gene_id: int) -> tuple:
    '''
    Reformat a list of processed sequence records (dicts) into sliced and padded arrays
    '''

    seq_len = end_idx - start_idx
    assert seq_len <= max_len, 'End index exceeds max length in sequence slice'

    padded_orf = record['ORF_coded'][start_idx: end_idx + 1] + [0] * (max_len - seq_len)
    padded_protein = record['Translation_coded'][start_idx: end_idx] + [0] * (max_len - seq_len)
    padded_weight = record['Codon_weights'][start_idx: end_idx] + [0] * (max_len - seq_len)
    gene_array = [gene_id]
    return (padded_protein, padded_orf, padded_weight, gene_array)


def record_to_segment_tensors(record: dict,
                              gene_id: int = 0,
                              segment_len: int = 20,
                              randomize_start: bool = True) -> list:
    '''
    Split a sequence into segmented regions as tensors for CoLLAGE training
    '''

    n_segments = int(record['Length'] / segment_len)
    if n_segments == 0:
        # Shorter than seq_len
        return [record_slice_to_arrays(record, 0, record['Length'], segment_len, gene_id)]
    else:
        n_extra_residues = record['Length'] - n_segments * segment_len
        start_idx = random.randint(0, int(n_extra_residues / 2.0)) if randomize_start else 0
        array_sets = [record_slice_to_arrays(record,
                                             start_idx + i * segment_len,
                                             start_idx + (i + 1) * segment_len,
                                             segment_len,
                                             gene_id)
                      for i in range(n_segments)]
        return array_sets


def tensorize_batch(batch: list) -> tuple:
    '''
    Convert arrays into tensors with proper typing
    '''
    prot_tensor = torch.Tensor([x[0] for x in batch]).to(torch.int64)
    orf_tensor = torch.Tensor([x[1] for x in batch]).to(torch.int64)
    weight_tensor = torch.Tensor([x[2] for x in batch]).to(torch.float32)
    gene_tensor = torch.Tensor([x[3] for x in batch]).to(torch.int64)
    return (prot_tensor, orf_tensor, weight_tensor, gene_tensor)


def records_to_batches(records: list,
                       segment_len: int,
                       batch_size: int,
                       randomize_order: bool,
                       randomize_start: bool,
                       by_gene: bool, ) -> list:
    '''
    Records to batches for training
    '''

    array_sets = [x for g, record in enumerate(records)
                  for x in record_to_segment_tensors(record,
                                                     g,
                                                     segment_len,
                                                     randomize_start)]

    if randomize_order:
        random.shuffle(array_sets)

    batches = []
    for i in range(0, len(array_sets), batch_size):
        batch = array_sets[i: i + batch_size]
        batches.append(tensorize_batch(batch))

    return batches
