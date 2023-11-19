import torch
import torch.nn as nn
from torcheval.metrics import R2Score

from model.dnn_arch import DNN


# Check device
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def train(args, train_loader, val_loader):
    device = get_device()
    print(f"DEVICE: {device}")

    # Create model, and define loss function and optimizer
    model = DNN(args.n_counters).to(device)
    l1loss = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_metric, val_metric = R2Score(), R2Score()

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
            preds = model(inputs)
            batch_loss = l1loss(preds, labels)
            batch_loss.backward()
            optimizer.step()

            train_metric.update(preds, labels)
            train_loss += batch_loss.item()

        # Validation
        model.eval() # Set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs)
                batch_loss = l1loss(preds, labels)
                
                val_metric.update(preds, labels)
                val_loss += batch_loss.item()

            train_acc = train_metric.compute()
            val_acc = val_metric.compute()
            print("[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}".format(
                epoch + 1, args.n_epoch,
                train_acc, train_loss/len(train_loader),
                val_acc, val_loss/len(val_loader)
            ))

            # If the model improves, save a checkpoint at this epoch
            if val_acc < best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), param_path)
                print("Saving model with acc {:.3f}".format(best_acc/len(val_loader)))

            acc_record["train"].append(train_acc/len(train_loader))
            acc_record["val"].append(val_acc/len(val_loader))
            loss_record["train"].append(train_loss/len(train_loader))
            loss_record["val"].append(val_loss/len(val_loader))

    return acc_record, loss_record