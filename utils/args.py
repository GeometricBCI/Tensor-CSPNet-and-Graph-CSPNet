'''
####################################################################################################################
Discription: 

Class of argparse.
####################################################################################################################
'''

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg_name', default = 'Graph_CSPNet', help = 'name of model')
    parser.add_argument('--mlp', default = False, help = 'whether the classifier is a multiple layer perception or not')

    parser.add_argument('--no-cuda', action = 'store_true', default=False, help = 'disables CUDA training')
    parser.add_argument('--initial_lr', type = float, default = 1e-3, help = "initial_lr")
    parser.add_argument('--decay', type = float, default = 1, help= "decay rate for adjust_learning")

    parser.add_argument('--start_No', type=int, default = 1, help = 'testing starts on subject #')
    parser.add_argument('--end_No', type=int, default = 9, help = 'testing ends on subject #')
    parser.add_argument('--epochs', type=int, default = 100, help = 'number of epochs to train')
    parser.add_argument('--patience', type=int, default = 15, help = 'patience for early stopping')

    parser.add_argument('--train_batch_size', type = int, default = 29, help = 'batch size in each epoch for trainning')
    parser.add_argument('--test_batch_size', type = int, default = 29, help = 'batch size in each epoch for testing')
    parser.add_argument('--valid_batch_size', type = int, default = 29, help = 'batch size in each epoch for validation')

    parser.add_argument('--seed', type = int, default = 1, metavar='S', help = 'random seed (default: 1)')
    parser.add_argument('--log_interval', type = int, default = 1, help = 'how many batches to wait before logging training status')
    parser.add_argument('--save-model', action = 'store_true', default=False, help = 'for saving the current model')

    parser.add_argument('--folder_name', default = 'results')
    parser.add_argument('--weights_folder_path', default = 'model_paras/')

    args = parser.parse_args(args=[])

    return args
