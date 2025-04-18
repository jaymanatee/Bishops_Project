import torch
from torch.utils.data import DataLoader
try:
    from utils import alert_generation
except:
    from src.utils import alert_generation

@torch.enable_grad()
def train_step(model, train_data, loss_fn, optimizer, writer, epoch, device, accuracy_fn):
    """
    Performs a single training step returning the current avarage loss and accuracy.
    """
    
    model.train()
    total_loss = 0.0
    total_ner_accuracy = 0
    total_sa_accuracy = 0
    total_batches = 0

    for _, tweet, caption, ner, sa in train_data:
        tweet = tweet.to(device).long()
        caption = caption.to(device).long()
        ner = ner.to(device).long()
        sa = sa.to(device).long()

        inputs = torch.cat((tweet, caption), dim=1)

        optimizer.zero_grad()
        predicted_ner, predicted_sa = model(inputs)
        loss = loss_fn(predicted_ner, ner, predicted_sa, sa)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        ner_accuracy, sa_accuracy = accuracy_fn(predicted_ner, ner, predicted_sa, sa)
        total_ner_accuracy += ner_accuracy.item()
        total_sa_accuracy += sa_accuracy.item()
        total_batches += 1

    ner_accuracy = total_ner_accuracy / total_batches
    sa_accuracy = total_sa_accuracy / total_batches

    writer.add_scalar("train/ner_accuracy", ner_accuracy, epoch)
    writer.add_scalar("train/sa_accuracy", sa_accuracy, epoch)

    return ner_accuracy, sa_accuracy


@torch.no_grad()
def val_step(model, val_data, writer, epoch, device, accuracy_fn):
    """
    Evaluates the model on the validation data without updating weights and returning the current avarage loss and accuracy.
    """
    
    model.eval()
    total_ner_accuracy = 0
    total_sa_accuracy = 0
    total_batches = 0

    for _, tweet, caption, ner, sa in val_data:
        tweet = tweet.to(device).long()
        caption = caption.to(device).long()
        ner = ner.to(device).long()
        sa = sa.to(device).long()

        inputs = torch.cat((tweet, caption), dim=1)

        predicted_ner, predicted_sa = model(inputs)
        ner_accuracy, sa_accuracy = accuracy_fn(predicted_ner, ner, predicted_sa, sa)
        total_ner_accuracy += ner_accuracy.item()
        total_sa_accuracy += sa_accuracy.item()
        total_batches += 1

    ner_accuracy = total_ner_accuracy / total_batches
    sa_accuracy = total_sa_accuracy / total_batches

    writer.add_scalar("validation/ner_accuracy", ner_accuracy, epoch)
    writer.add_scalar("validation/sa_accuracy", sa_accuracy, epoch)

    return ner_accuracy, sa_accuracy


@torch.no_grad()
def t_step(model, test_data, accuracy_fn, device):
    """ 
    Evaluates the model on the test data without updating weights and returning the current accuracy.
    """

    model.eval()
    total_ner_accuracy = 0
    total_sa_accuracy = 0
    total_batches = 0

    for _, tweet, caption, ner, sa in test_data:
        tweet = tweet.to(device).long()
        caption = caption.to(device).long()
        ner = ner.to(device).long()
        sa = sa.to(device).long()

        inputs = torch.cat((tweet, caption), dim=1)

        predicted_ner, predicted_sa = model(inputs)
        ner_accuracy, sa_accuracy = accuracy_fn(predicted_ner, ner, predicted_sa, sa)
        total_ner_accuracy += ner_accuracy.item()
        total_sa_accuracy += sa_accuracy.item()
        total_batches += 1

    ner_accuracy = total_ner_accuracy / total_batches
    sa_accuracy = total_sa_accuracy / total_batches

    return ner_accuracy, sa_accuracy