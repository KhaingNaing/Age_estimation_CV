import torch 
from tqdm import tqdm
from config import config

def train_one_epoch(model, train_loader, optimizer, loss_fn, metric):
    model.train()
    train_loss = 0
    train_metric = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(config["device"]), labels.to(config["device"])

        optimizer.zero_grad()

        outputs = model(images)
        outputs = outputs.squeeze()

        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_metric += metric(outputs, labels)

    return model, train_loss / len(train_loader), train_metric / len(train_loader)

def validate_one_epoch(model, valid_loader, loss_fn, metric):
    model.eval()
    valid_loss = 0
    valid_metric = 0

    metric.reset()

    with torch.no_grad():
        for images, labels in tqdm(valid_loader):
            images, labels = images.to(config["device"]), labels.to(config["device"])

            outputs = model(images)
            outputs = outputs.squeeze()

            loss = loss_fn(outputs, labels)

            valid_loss += loss.item()
            valid_metric += metric(outputs, labels)

    return valid_loss / len(valid_loader), valid_metric / len(valid_loader)