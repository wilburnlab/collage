import torch


HYPERPARAMETERS = { 'n_input_tokens' : 22,
                    'n_output_tokens' : 66,
                    'embed_dimension' : 64,
                    'feedforward_dimension' : 256,
                    'n_heads' : 4,
                    'n_encoder_layers' : 1,
                    'n_decoder_layers' : 2,
                    'max_sequence_length' : 500, }

TRAINING_PARAMETERS = { 'n_epochs' : 250,
                        'learning_rate' : 1e-1,
                        'dropout_rate' : 0.20,
                        'train_batch_size' : 4,
                        'initial_sequence_length' : 30,
                        'epochs_to_2x_length' : 50,
                        'optimizer' : torch.optim.SGD, }