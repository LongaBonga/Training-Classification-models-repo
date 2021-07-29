from tqdm import tqdm
import time
import torch
from help_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib, add_to_writer
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ExponentialLR

def train_func(args, model, criterion, optimizer, train_dataloader, test_dataloader, save_path, model_name,device, writer, NUM_EPOCH=40):

    cnt = 0
    scaler = GradScaler()
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    for epoch in tqdm(range(NUM_EPOCH)):
        model.train()

        epoch_start_time = time.time()
        for imgs, labels in train_dataloader:
            cnt += 1
            train_loss = 0.
            train_size = 0
            train_pred = 0.

            imgs = imgs.to(device)
            labels = labels.to(device)

            loss, y_pred =  gradient_step(args, model, optimizer, criterion, scaler, imgs, labels)

            train_loss += loss.item()
            train_size += y_pred.size(0)

            train_pred += (y_pred.argmax(1) == labels).sum()

            add_to_writer(writer, train_loss, train_pred, train_size, cnt, 'Train')

        epoch_time = time.time() - epoch_start_time
        print_at_master(
                "\nFinished Epoch #{}, Training Rate: {:.1f} [img/sec]".format(epoch, len(train_dataloader) *
                                                                       args.batch_size / epoch_time * max(num_distrib(),
                                                                                                          1)))

        val_loss, val_acr = val_func(args, model, criterion, optimizer, test_dataloader, device, writer, epoch)

        scheduler.step()

        print_at_master(f'Train loss: {train_loss / train_size}')
        print_at_master(f'Val loss: {val_loss}')
        print_at_master(f'Train acc: {train_pred / train_size * 100}')
        print_at_master(f'Val acc: {val_acr * 100}')




        if (epoch + 1) % 10 == 0:
            save_last_model_path = save_path + model_name + '_last_model_state_dict.pth'
            torch.save(model.state_dict(), save_last_model_path)



def val_func(args, model, criterion, optimizer, test_dataloader, device, writer, epoch = 0):
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            val_loss = 0.
            val_size = 0
            val_pred = 0.

            imgs = imgs.to(device)
            labels = labels.to(device)

            pred = model(imgs)
            loss = criterion(pred, labels)

            val_loss += loss.item()
            val_size += pred.size(0)

            val_pred += (pred.argmax(1) == labels).sum()

            if num_distrib() > 1:
                val_pred = reduce_tensor(val_pred, num_distrib())
                val_loss = reduce_tensor(val_loss, num_distrib())
                torch.cuda.synchronize()

            if args.mode == 'train':
                add_to_writer(writer, val_loss, val_pred, val_size, epoch, 'Val')
            
    return val_loss / val_size, val_pred / val_size


def gradient_step(args, model, optimizer, criterion, scaler, imgs, labels):
    if args.fp16:
        with autocast():
            y_pred = model(imgs)
            loss = criterion(y_pred, labels)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss, y_pred

    else:
        y_pred = model(imgs)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
        return loss, y_pred