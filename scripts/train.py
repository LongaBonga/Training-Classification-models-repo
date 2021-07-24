from tqdm import tqdm

def train_func(model, criterion, optimizer, train_dataloader, test_dataloader, device, save_path, model_name, writer, NUM_EPOCH=15):

    for epoch in tqdm(range(NUM_EPOCH)):
        model.train()

        for imgs, labels in train_dataloader:

            train_loss = 0.
            train_size = 0
            train_pred = 0.

            optimizer.zero_grad()

            imgs = imgs
            labels = labels

            y_pred = model(imgs)

            loss = criterion(y_pred, labels)
            loss.backward()

            train_loss += loss.item()
            train_size += y_pred.size(0)

            train_pred += (y_pred.argmax(1) == labels).sum()

            optimizer.step()

            writer.add_scalar('Train loss:', (train_loss / train_size), epoch)
            writer.add_scalar('Train acc:', (train_pred / train_size) * 100, epoch)

        val_loss, val_pred, val_size = val_func(model, criterion, optimizer, test_dataloader, device)


        print('Train loss:', (train_loss / train_size))
        print('Val loss:', (val_loss / val_size))
        print('Train acc:', (train_pred / train_size)*100)
        print('Val acc:', (val_pred / val_size)*100)




        if (epoch + 1) % 10 == 0:
            save_last_model_path = cfg.output_dir + model_name + ' last_model_state_dict.pth'
            torch.save(model.state_dict(), save_last_model_path)



def val_func(model, criterion, optimizer, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            val_loss = 0.
            val_size = 0
            val_pred = 0.

            imgs = imgs
            labels = labels

            pred = model(imgs)
            loss = criterion(pred, labels)

            val_loss += loss.item()
            val_size += pred.size(0)

            val_pred += (pred.argmax(1) == labels).sum()

            writer.add_scalar('Val loss:', (val_loss / val_size), epoch)
            writer.add_scalar('Val acc:', (val_pred / val_size) * 100, epoch)

    return val_loss, val_pred, val_size
