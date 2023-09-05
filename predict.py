import os
from run import init_model
import argparse
import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from train import evaluate
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from prepare_ds import transforms


def predict():
    # region Load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='..' + os.sep + 'dataset')

    parser.add_argument('--save_model_path',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='..' + os.sep + 'output')
    
    parser.add_argument('--mode',
                        type=str,
                        help='base|maml|protonet',
                        default='base')

    parser.add_argument('--checkpoint_path',
                        type=str,
                        help='root where to store checkpoit to init weight for model',
                        default='..' + os.sep + 'output')
    parser.add_argument('--batch_size',
                    type=int,
                    help='input for the batch size initializations',
                    default= 1024)
    args = parser.parse_args()
    print(f'args: {args}')
    
    # endregion

    # region get model
    model = init_model(
        checkpoint_path = args.checkpoint_path,
        num_classes= 4,
        mode = args.mode,
    )
    # endregion

    # region get criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # endregion

    # region get loader
    predictset = ImageFolder(root=args.dataset_root, transform=transforms)
    predictloader = DataLoader(predictset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=2
                            )
    # endregion

    # region test
    model.eval()
    test_acc, _ = evaluate(
        model,
        predictloader,
        criterion,
        device
    )
        
    with open(f'{args.save_model_path}/test.log', 'w', encoding='utf-8') as log:
        log.write(f'Accuracy: {test_acc}')
    print(f'Accuracy: {test_acc}')


    # endregion

if __name__ == '__main__':
    predict()
