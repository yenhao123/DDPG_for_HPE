import torch
import torch.nn as nn

from model.dnn_arch import DNN


# Check device
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def train(args, train_loader, val_loader):
    device = get_device()
    print(f"DEVICE: {device}")

    # Create model, and define loss function and optimizer
    model = DNN(args.n_counters).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # The path where checkpoint saved
    param_path = args.param_dir / "model.ckpt"

    # Start training
    best_acc = 0.0
    acc_record = {"train": [], "val": []}
    loss_record = {"train": [], "val": []}

    for epoch in range(args.n_epoch):
        train_acc, val_acc = 0.0, 0.0
        train_loss, val_loss = 0.0, 0.0

        # Training
        model.train() # Set the model to training mode
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            max_outputs, train_pred = torch.max(outputs, 1) # Get the index of the class with the highest probability
            batch_loss.backward()
            optimizer.step()

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # Validation
        if len(val_set) > 0:
            model.eval() # Set the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)
                    _, val_pred = torch.max(outputs, 1) # Get the index of the class with the highest probability

                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                    val_loss += batch_loss.item()

                print("[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}".format(
                    epoch + 1, NUM_EPOCH,
                    train_acc/len(train_set), train_loss/len(train_loader),
                    val_acc/len(val_set), val_loss/len(val_loader)
                ))

                # If the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), param_path)
                    print("Saving model with acc {:.3f}".format(best_acc/len(val_set)))

                acc_record["train"].append(train_acc/len(train_set))
                acc_record["val"].append(val_acc/len(val_set))
                loss_record["train"].append(train_loss/len(train_loader))
                loss_record["val"].append(val_loss/len(val_loader))

        else:
            print("[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}".format(
                epoch + 1, NUM_EPOCH, train_acc/len(train_set), train_loss/len(train_loader)
            ))

    # If not validating, save the last epoch
    if len(val_set) == 0:
        torch.save(model.state_dict(), param_path)
        print("Saving model at last epoch")