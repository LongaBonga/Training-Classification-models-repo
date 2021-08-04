from tqdm import tqdm
import time
import torch
from help_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib, add_to_writer, is_master
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_func(args, model, criterion, optimizer, train_dataloader, test_dataloader, save_path, model_name,device, writer):

    cnt = 0
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # scheduler = OneCycleLR(optimizer, max_lr = 0.001, pct_start=0.2, anneal_strategy='cos',
    #         cycle_momentum=True, base_momentum=0.85,
    #         max_momentum=0.95, div_factor=args.scheduler_coef, total_steps = args.num_epoch,
    #         final_div_factor=10000.0)

    for epoch in tqdm(range(args.num_epoch)):
        model.train()

        epoch_start_time = time.time()
        for imgs, labels in train_dataloader:
            cnt += 1
            train_loss = 0.
            train_size = 0
            train_pred = 0.

            imgs = imgs.to(device)
            labels = labels.to(device)

            loss, y_pred =  gradient_step(args, model, optimizer, criterion, scaler, imgs, labels, device)

            train_loss += loss.item()
            train_size += y_pred.size(0)

            train_pred += (y_pred.argmax(1) == labels).sum()

            add_to_writer(writer, train_loss, train_pred, train_size, cnt, 'Train')

        epoch_time = time.time() - epoch_start_time
        print_at_master(
                "\nFinished Epoch #{}, Training Rate: {:.1f} [img/sec]".format(epoch, len(train_dataloader) *
                                                                       args.batch_size / epoch_time * max(num_distrib(),
                                                                                                          1)))

        val_acr = val_func(args, model, criterion, optimizer, test_dataloader, device, writer, epoch)

        if args.scheduler:
            scheduler.step(loss)

        print_at_master(f'Train loss: {train_loss / train_size}')
        print_at_master(f'Train acc: {train_pred / train_size * 100}')
        print_at_master(f'Val acc: {val_acr * 100}')


        if is_master() and ((epoch + 1) % 10) == 0:
            save_last_model_path = save_path + model_name + '_last_model_state_dict.pth'
            torch.save(model.state_dict(), save_last_model_path)



def val_func(args, model, criterion, optimizer, test_dataloader, device, writer, epoch = 0):
    model.eval()
    with torch.no_grad():
        val_size = 0
        val_pred = 0.
        for imgs, labels in test_dataloader:
            
            

            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs)

            val_size += pred.size(0)
            val_pred += (pred.argmax(1) == labels).sum()

            if num_distrib() > 1:
                val_pred = reduce_tensor(val_pred, num_distrib())
                torch.cuda.synchronize()

        if args.mode == 'train':
            add_to_writer(writer, None, val_pred, val_size, epoch, 'Val')
            
    return val_pred / val_size


def gradient_step(args, model, optimizer, criterion, scaler, imgs, labels, device):
    if args.fp16:
        with autocast():
            y_pred = model(imgs)
            loss = criterion(y_pred, labels) + L1(model, args.l1_coef, device)

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

def L1(model, coef, device):
    l1_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l1_reg += torch.sum(torch.abs(param))
    return l1_reg * coef