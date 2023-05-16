import math
import torch

from collage.utils import timer
from collage.tensorize import records_to_batches
from collage.eval_functions import WeightedLikelihood, calc_accuracy
from collage.tensorize import split_train_test_data
from collage.model import initialize_collage_model
from collage.settings import TRAINING_PARAMETERS


def train_loop(model,
               optimizer,
               loss_fx,
               data_by_mode: dict,
               output_name: str,
               start_time: float,
               epochs: int = 20,
               epochs_to_2x_length: int = 50,
               device: str = 'cpu',
               train_batch_size: int = 6,
               initial_seq_len: int = 30,
               scheduler=None):

    # Create log file
    log_file = open(output_name + '.log', 'w')
    log_file.close()

    model.to(device)
    loss_fx.to(device)

    best_epoch = -1
    best_score = 1e8

    for epoch in range(epochs):
        print('----------\nEpoch ' + str(epoch + 1) + ' of ' + str(epochs) + '\n----------')

        log_file = open(output_name + '.log', 'a')
        log_file.write('Epoch ' + str(epoch + 1) + '\n')
        log_file.close()

        for mode in data_by_mode:
            if mode == 'train':
                model.train()
            else:
                model.eval()

            total_loss = 0
            total_accuracy = 0

            seq_len = int(initial_seq_len * math.pow(2.0,
                          epoch / epochs_to_2x_length))
            seq_len = min([seq_len, 500, ])

            batches = records_to_batches(data_by_mode[mode],
                                         segment_len=seq_len,
                                         batch_size=train_batch_size if mode == 'train' else 32,
                                         randomize_order=mode == 'train',
                                         randomize_start=mode == 'train',
                                         by_gene=False)

            total_n = 0
            counter = 0
            for i, batch in enumerate(batches):
                prot, cds, weight, gene = [
                    x.to(device) for x in batch]  # , sp, null
                targets = cds[:, 1:]
                n = prot.flatten().size(0)
                pred = model(prot, cds[:, :-1], )  # sp )
                loss = loss_fx(pred, targets, weight)
                accuracy = calc_accuracy(targets, pred, )

                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step()

                total_loss += loss.detach().item() * n
                total_accuracy += accuracy * n
                total_n += n
                counter += n
            score = total_loss / total_n
            mode_accuracy = total_accuracy / total_n
            log_file = open(output_name + '.log', 'a')
            log_file.write('\t' + mode.capitalize() +
                           ' score: ' + format(score, '.5f') + '\n')
            log_file.close()
            print(mode.capitalize() + format(score, '.4f').rjust(8) +
                  ' (' + format(mode_accuracy, '.2%') + ')')
            if mode == 'test':
                if score < best_score:
                    print('New best weights! Copying and saving model to\n\t' + output_name + '.pt')
                    best_epoch = epoch
                    best_score = score
                    torch.save(model.state_dict(), output_name + '.pt')
                    epochs_wo_improv = 0
                else:
                    print('Did not improve, best performance was epoch ' +
                          str(best_epoch + 1) + ' (' + format(best_score, '.4f') + ')')
        print('Runtime: ' + timer(start_time) + '\n')
    return score


def train_collage(output_name, training_data, test_frac, random_seed, start_time,
                  device=None, start_model=None, epochs=TRAINING_PARAMETERS['n_epochs']):

    data_by_mode = split_train_test_data(training_data,
                                         test_frac,
                                         random_seed)
    loss_fx = WeightedLikelihood()
    model = initialize_collage_model(start_model,
                                     gpu=device == 'cuda')
    optimizer = TRAINING_PARAMETERS['optimizer'](model.parameters(),
                                                 lr=TRAINING_PARAMETERS['learning_rate'])

    train_loop(model=model,
               optimizer=optimizer,
               loss_fx=loss_fx,
               data_by_mode=data_by_mode,
               output_name=output_name,
               start_time=start_time,
               epochs=epochs,
               epochs_to_2x_length=TRAINING_PARAMETERS['epochs_to_2x_length'],
               device=device,
               train_batch_size=TRAINING_PARAMETERS['train_batch_size'],
               initial_seq_len=TRAINING_PARAMETERS['initial_sequence_length'],
               scheduler=None)
