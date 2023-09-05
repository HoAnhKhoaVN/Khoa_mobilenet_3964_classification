import os
from run import init_model
import argparse
import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from prepare_ds import transforms
from PIL import Image
import torch.nn.functional as F


IMG_SIZE = 224
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE), antialias= False),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
class RawDataset(Dataset):

    def __init__(self, root):
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        # self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        try:
            img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            img = transform(img)
        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


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

    # region prepare data
    predictset = RawDataset(root = args.dataset_root)
    predictloader = DataLoader(predictset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=2
                            )
    # endregion

    # region test
    
    with open(f'{args.save_model_path}/log_predict_result.txt', 'w') as log:
        model.eval()
        with torch.no_grad():
            for image_tensors, image_path_list in predictloader:
                images = image_tensors.to(device)
                # print(f'Size images: {images.size()}')
                preds = model(images)
                # print(f'Size predictions: {preds.size()}')
                preds = F.softmax(preds, dim=1)
                preds_prob, preds = preds.max(1)

                # print(f'Predictions: {preds}')
                # print(f'Size predictions probability: {preds_prob}')
                for img_name, pred, pred_max_prob in zip(image_path_list, preds, preds_prob):
                    str_val = str(pred.item())
                    log.write(f'{img_name:25s}\t{str_val:25s}\t{pred_max_prob}\n')
    # endregion

if __name__ == '__main__':
    predict()
