import torch
from torch.utils.data import DataLoader


@torch.enable_grad()
def train_step(model, train_data, loss_fn, optimizer, writer, epoch, device):
    """
    Performs a single training step returning the current avarage loss and accuracy.
    """
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in train_data:
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)
    accuracy = total_correct / total_samples

    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/accuracy", accuracy, epoch)

    return avg_loss, accuracy


@torch.no_grad()
def val_step(model, val_data, loss_fn, scheduler, writer, epoch, device):
    """
    Evaluates the model on the validation data without updating weights and returning the current avarage loss and accuracy.
    """
    
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0

    for inputs, targets in val_data:
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()
    
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / len(val_data)
    accuracy = total_correct / total_samples

    writer.add_scalar("validation/loss", avg_loss, epoch)
    writer.add_scalar("validation/accuracy", accuracy, epoch)

    scheduler.step()

    return avg_loss, accuracy


@torch.no_grad()
def t_step(model, test_data, device,):
    """
    Evaluates the model on the test data without updating weights and returning the current accuracy.
    """

    model.eval()
    total_correct = 0
    total_samples = 0

    for inputs, targets in test_data:
        inputs = inputs.to(device).float()
        targets = targets.to(device).long()

        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy
