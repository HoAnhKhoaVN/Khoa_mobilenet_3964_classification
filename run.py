from prepare_ds import load_dataset
from parser_util import get_parser
from torch import nn
import torch
import numpy as np
import torch.optim as optim
from mobilenet import MyMobileNetV2
from torch.utils.tensorboard import SummaryWriter
from train import train, evaluate
from os import makedirs, path
from mobilenetv2 import MobileNetV2
from my_efficientnet_b3 import MyEfficientnetB3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_model(
    checkpoint_path: str,
    num_classes: int,
    mode: str = 'base'
):
    # region
    if mode.lower() == 'maml':
        model = MobileNetV2(num_classes = num_classes).to(device)
    if mode.lower() == 'han_nom_cls':
       model = MyEfficientnetB3(num_classes=4).to(device)
    else:
        model = MyMobileNetV2(num_classes = num_classes).to(device)
    # endregion

    # region Load checkpoints if available
    if path.exists(checkpoint_path):
        print(f'loading pretrained model from {checkpoint_path}')
        model.load_state_dict(
            torch.load(
                path.join(checkpoint_path)
            )
        )
    # endregion
    return model

def main():
    # region Load arguments
    parser = get_parser()
    args = parser.parse_args()
    print(f'args: {args}')
    
    # endregion

    # region settings to reproduce
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    # endregion

    # region write tensorboard arguments
    makedirs(name = args.save_model_path, exist_ok= True)
    writer = SummaryWriter(
        log_dir=args.save_model_path
    )
    # endregion

    # region Load data
    (
        trainloader,
        valloader,
        testloader,
        classes
    ) = load_dataset(args)
    # endregion

    # region get model
    model = init_model(
        checkpoint_path = args.checkpoint_path,
        num_classes= len(classes),
        mode = args.mode,
    )
    print(f'Model architecture:')
    # print(model)
    # endregion

    # region get criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    optimizer = optim.Adadelta(
        filtered_parameters,
        lr=args.learning_rate,
        rho = args.rho, 
        eps = args.eps
        )
    # endregion

    # region training
    if args.train:
        train(
            num_epoch = args.epochs,
            trainloader = trainloader,
            optimizer = optimizer,
            model = model,
            device = device,
            criterion = criterion,
            save_model_path = args.save_model_path,
            valloader = valloader,
            writer = writer,
            start_iter = args.startIter,
            val_iter = args.valIter,
            max_iter = args.maxIter,
        )
    # endregion

    # region test
    if args.test:
        model.eval()
        test_acc, _ = evaluate(
            model,
            testloader,
            criterion,
            device
        )
        
        with open(f'{args.save_model_path}/test.log', 'w', encoding='utf-8') as log:
            log.write(f'Accuracy: {test_acc}')
        print(f'Accuracy: {test_acc}')


    # endregion

if __name__ == '__main__':
    main()


