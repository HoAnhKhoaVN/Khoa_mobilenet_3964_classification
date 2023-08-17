import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from torch import nn
from utils import Averager


def evaluate(
    model: nn.Module,
    evaluation_loader: DataLoader,
    criterion,
    device,
):
    n_correct = 0
    length_of_data = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        images = image_tensors.to(device)
        labels = labels.to(device)


        outputs = model(images)

        cost = criterion(outputs, labels)
        valid_loss_avg.add(cost)

        _, preds = outputs.max(1)

        n_correct +=torch.sum(preds==labels).cpu().detach().numpy().tolist()
    accuracy = n_correct / float(length_of_data) * 100
    return (
        accuracy,
        valid_loss_avg.val()
    )

def train(
    num_epoch: int,
    trainloader,
    optimizer,
    model,
    device,
    criterion,
    save_model_path,
    valloader,
    writer,
    start_iter,
    val_iter,
    max_iter
):
    loss_avg = Averager()
    iteration = start_iter
    best_accuracy = -1


    for _ in tqdm(range(num_epoch), desc= 'Epoch progress:'):  # loop over the dataset multiple times
        for inputs, labels in trainloader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.to(device)
            outputs = model(inputs)
            labels = labels.to(device)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_avg.add(loss)

            # validation part
            if (iteration + 1) % val_iter == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0'
                with open(f'{save_model_path}/log_train.txt', 'a') as log:
                    model.eval()
                    with torch.no_grad():
                        current_accuracy, valid_loss = evaluate(model, valloader)
                    model.train()

                    # training loss and validation loss
                    log.write(f'[{iteration+1}/{max_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}\n')


                    # region write to tensorboard
                    writer.add_scalar('training loss',
                                    loss_avg.val(),
                                    iteration+1)

                    writer.add_scalar('valid loss',
                                    valid_loss,
                                    iteration+1)

                    writer.add_scalar('Acc valid',
                        current_accuracy,
                        iteration+1)
                    # endregion

                    loss_avg.reset()

                    log.write(f'{"Current_accuracy":17s}: {current_accuracy:0.3f}\n')

                    # keep best accuracy model (on valid dataset)
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(model.state_dict(), f'{save_model_path}/best_accuracy.pth')
                    log.write(f'{"Best_accuracy":17s}: {best_accuracy:0.3f}\n')


            # save model per 1000 iter.
            if (iteration + 1) % 1000 == 0:
                torch.save(
                    model.state_dict(),
                    f'{save_model_path}/iter_{iteration+1}.pth'
                )



            if (iteration + 1) == max_iter:
                with open(f'{save_model_path}/log_train.txt', 'a') as log:
                    log.write('End the training\n')
                print('End the training')
                sys.exit()
            iteration += 1
