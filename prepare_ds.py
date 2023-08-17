import os
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from typing import Text, List, Tuple
import torchvision

def load_dataset(
    args
)-> Tuple:
    DS_PATH = args.root
    # region Detete empty folder in train, test and valid
    PATH = os.path.join(DS_PATH, 'train')
    small_fds =[fd for fd in tqdm(os.listdir(PATH)) if len(os.listdir(os.path.join(PATH, fd))) < 247]
    for role in ['train', 'test', 'val']:
        for fd in tqdm(small_fds, desc = f'Detele folder {role}'):
            path = os.path.join(DS_PATH, role, fd)
            os.system(command = f'rm -r {path}')

    # endregion

    # region Handle empty valid
    for fd in tqdm(os.listdir(f'{DS_PATH}/val'), desc = "Handle empty valid: "):
        val_path = os.path.join(DS_PATH, 'val', fd)
        if len(os.listdir(val_path)) ==0: # Empty
            train_path = os.path.join(DS_PATH, 'train', fd)
            top3_train = os.listdir(train_path)[:3]


            for fn in top3_train: # Chuyển 3 ảnh từ tập train sang làm valid
                src = os.path.join(train_path, fn)
                tgt = os.path.join(val_path, fn)
                os.system(command = f'mv {src} {tgt}')
    # endregion


    # region Dataset and Dataloader
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((32, 32), antialias= False),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    trainset = torchvision.datasets.ImageFolder(root=f'{DS_PATH}/train', transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)


    valset = torchvision.datasets.ImageFolder(root=f'{DS_PATH}/val', transform=transform)
    valloader = DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root=f'{DS_PATH}/test', transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    classes = trainset.classes


    # endregion

    # region write log for dataset
    log = open(f'{args.save_model_path}/log_dataset.txt', 'a')

    log.write(f'''
    ================ DATASET ======================
    Number of train: {len(trainset.samples)}
    Number of valid: {len(valset.samples)}
    Number of test: {len(testset.samples)}
    Length of class: {len(classes)}
    Top 10 class: {classes[:10]}
    ''')
    log.write('-' * 80 + '\n')
    log.close()
    # endregion

    return (
        trainloader,
        valloader,
        testloader,
        classes
    )




