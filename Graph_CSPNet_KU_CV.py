'''
#####################################################################################################################
Author     : Ce Ju, Nanyang Technological University. 
Date       : 1st, Sep., 2022
---------------------------------------------------------------------------------------------------------------------
Discription: Trainning file of Graph-CSPNet for cross-validation scenario on KU. 
#######################################################################################################################
'''

import time
import pandas as pd
import numpy as np



#import torch and sklearn
from torch.autograd import Variable
import torch.nn.functional as F
import torch as th
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from sklearn.model_selection import StratifiedShuffleSplit


#import util folder
from utils.model import Graph_CSPNet_Basic
#from utils.functional import MixOptimizer
from utils.early_stopping import EarlyStopping
from utils.load_data import load_KU, dataloader_in_main
from utils.args import args_parser
import utils.geoopt as geoopt



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    optimizer.lr = args.initial_lr * (args.decay ** (epoch // 100))


def main(args, train, test, train_y, test_y, graph_matrix, adjacency_matrix, sub, total_sub, kf_iter, validation):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    if validation:
        index_split = StratifiedShuffleSplit(n_splits=1, test_size=0.01, random_state=42)
        for train_index, valid_index in index_split.split(train,train_y):
            train_sampler = SubsetRandomSampler(train_index)
            valid_sampler = SubsetRandomSampler(valid_index)
            
        train       = Variable(torch.from_numpy(train)).double()
        test        = Variable(torch.from_numpy(test)).double()
        train_y     = Variable(torch.LongTensor(train_y))
        test_y      = Variable(torch.LongTensor(test_y))
          
        train_dataset = dataloader_in_main(train, train_y)
        test_dataset  = dataloader_in_main(test, test_y)

        train_kwargs = {'batch_size': args.train_batch_size}
        if use_cuda:
              cuda_kwargs ={'num_workers': 100,
                            'sampler': train_sampler,
                              'pin_memory': True,
                              'shuffle': True     
              }
              train_kwargs.update(cuda_kwargs)
              
        valid_kwargs = {'batch_size': args.valid_batch_size}
        if use_cuda:
              cuda_kwargs ={'num_workers': 10,
                            'sampler':valid_sampler,
                              'pin_memory': True,
                              'shuffle': True     
              }
              valid_kwargs.update(cuda_kwargs)

        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
              cuda_kwargs ={'num_workers': 1,
                              'pin_memory': True,
                              'shuffle': True      
              }
              test_kwargs.update(cuda_kwargs)

        train_loader  = torch.utils.data.DataLoader(dataset= train_dataset, **train_kwargs)
        valid_loader  = torch.utils.data.DataLoader(dataset= train_dataset, **valid_kwargs)
        test_loader   = torch.utils.data.DataLoader(dataset= test_dataset,  **test_kwargs)
    else:
        train       = Variable(torch.from_numpy(train)).double()
        test        = Variable(torch.from_numpy(test)).double()
        train_y     = Variable(torch.LongTensor(train_y))
        test_y      = Variable(torch.LongTensor(test_y))

        train_dataset = dataloader_in_main(train, train_y)
        test_dataset  = dataloader_in_main(test, test_y)

        train_kwargs  = {'batch_size': args.train_batch_size}

        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                            'pin_memory': True,
                            'shuffle': True		
        }
            train_kwargs.update(cuda_kwargs)

        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs  = {'num_workers': 1,
                            'pin_memory': True,
                            'shuffle': True      
            }
            test_kwargs.update(cuda_kwargs)

        train_loader  = torch.utils.data.DataLoader(dataset= train_dataset, **train_kwargs)
        test_loader   = torch.utils.data.DataLoader(dataset= test_dataset,  **test_kwargs)

    

    model = Graph_CSPNet_Basic(channel_num = train.shape[1],
        P = Variable(torch.from_numpy(graph_matrix)).double(),  
        mlp = args.mlp,
        ).to(device)

    optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=args.initial_lr)

    early_stopping = EarlyStopping(
        alg_name = args.alg_name, 
        path_w   = args.weights_folder_path + args.alg_name + '_checkpoint.pt', 
        patience = args.patience, 
        verbose  = True, 
        )

    print('#####Start Trainning######')

    for epoch in range(1, args.epochs+1):

        adjust_learning_rate(optimizer, epoch)

        model.train()

        train_correct = 0
    
        for batch_idx, (batch_train, batch_train_y) in enumerate(train_loader):

            optimizer.zero_grad()

            logits = model(batch_train.to(device))
            output = F.log_softmax(logits, dim = -1)
            loss   = F.nll_loss(output, batch_train_y.to(device))


            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('----#------#-----#-----#-----#-----#-----#-----')
                pred    = output.data.max(1, keepdim=True)[1]
                train_correct += pred.eq(batch_train_y.to(device).data.view_as(pred)).long().cpu().sum()
                torch.save(model.state_dict(), args.weights_folder_path + args.alg_name+'_model.pth')
                torch.save(optimizer.state_dict(), args.weights_folder_path+'optimizer.pth')


                print('['+args.alg_name+': Sub No.{}/{} Fold {}/10, Epoch {}/{}, Completed {:.0f}%]:\nTrainning loss {:.10f} Acc.: {:.4f}'.format(\
                        sub, total_sub, kf_iter+1, epoch, args.epochs, 100. * (1+batch_idx) / len(train_loader), loss.cpu().detach().numpy(),\
                        train_correct.item()/len(train_loader.dataset)))
                    

        if validation:
            #Validate the Model
            valid_losses  = []
            valid_loss    =  0
            valid_correct =  0

            model.eval()

            for batch_idx, (batch_valid, batch_valid_y) in enumerate(valid_loader):

                logits         = model(batch_valid.to(device))
                output         = F.log_softmax(logits, dim = -1)
                valid_loss    += F.nll_loss(output, batch_valid_y.to(device))
                valid_losses.append(valid_loss.item())
                
                
                pred           = output.data.max(1, keepdim=True)[1]
                valid_correct += pred.eq(batch_valid_y.to(device).data.view_as(pred)).long().cpu().sum()

            print('Validate loss: {:.10f} Acc: {:.4f}'.format(sum(valid_losses), valid_correct.item()/len(valid_loader.dataset)))
            
            early_stopping(np.average(valid_losses), model)
            
            if early_stopping.early_stop:
              print("Early Stopping!")
              break
        else:
            pass
        

    #Testing
    print('###############################################################')
    print('START TESTING')
    print('###############################################################')

    
    model.eval()

    test_loss    = 0
    test_correct = 0

    with torch.no_grad():
        for batch_idx, (batch_test, batch_test_y) in enumerate(test_loader):

            logits        = model(batch_test.to(device))
            output        = F.log_softmax(logits, dim = -1)
            test_loss    += F.nll_loss(output, batch_test_y.to(device))
            
            test_pred     = output.data.max(1, keepdim=True)[1]
            test_correct += test_pred.eq(batch_test_y.to(device).data.view_as(test_pred)).long().cpu().sum()

            print('-----------------------------------')
            print('Testing Batch {}:'.format(batch_idx))
            print('  Pred Label:', test_pred.view(1, test_pred.shape[0]).cpu().numpy()[0])
            print('Ground Truth:', batch_test_y.numpy())


    return test_correct.item()/len(test_loader.dataset), test_loss.item()/len(test_loader.dataset)


if __name__ == '__main__':

    args   = args_parser()

    alg_df = pd.DataFrame(columns=['R1', 'R2', 'R3','R4', 'R5', 'R6', 'R7', 'R8','R9', 'R10','Avg'])

    print('############Start Task#################')
    
    for sub in range(args.start_No, args.end_No + 1):

        KU_dataset = load_KU(['dataset/KU_sess1_sub'+str(sub)], 
            'index/KU_sess1_sub'+str(sub), 
            alg_name = args.alg_name,
            scenario = 'CV'
            )

        alg_record = []

        start      = time.time()

        for kf_iter in range(0, 10):

            x_train_stack, x_test_stack, y_train, y_test = KU_dataset.generate_training_test_set_CV(kf_iter)

            graph_M, adj_M = KU_dataset.LGT_graph_matrix_fn()
            print('###Graph Generated!###')

            acc, loss = main(
                args       = args, 
                train      = x_train_stack, 
                test       = x_test_stack, 
                train_y    = y_train, 
                test_y     = y_test,
                graph_matrix = graph_M,
                adjacency_matrix = adj_M,
                sub        = sub, 
                total_sub  = args.end_No - args.start_No + 1,
                kf_iter    = kf_iter, 
                validation = True,
                )
            

            print('##############################################################')

            print(args.alg_name + ' Testing Loss.: {:4f} Acc: {:4f}'.format(loss, acc))

            alg_record.append(acc)

        end = time.time()

        alg_record.append(np.mean(alg_record))
        
        alg_df.loc[sub] = alg_record
 
        alg_df.to_csv(args.folder_name + '/' \
        + time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()) \
        + args.alg_name \
        +'_Sub(' \
        + str(args.start_No) \
        +'-' \
        +str(args.end_No) \
        +')' \
        +'_' \
        + str(args.epochs)\
        + '.csv'\
        , index = False)
    

