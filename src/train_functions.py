import torch
from torch.utils.data import DataLoader
from utils import alert_generation


@torch.enable_grad()
def train_step(model, train_data, loss_fn, optimizer, writer, epoch, device):
    """
    Performs a single training step returning the current avarage loss and accuracy.
    """
    
    model.train()
    total_loss = 0.0

    for _, tweet, caption, ner, sa in train_data:
        tweet = tweet.to(device).long()
        caption = caption.to(device).long()
        ner = ner.to(device).long()
        sa = sa.to(device).long()

        inputs = torch.cat((tweet, caption), dim=1)

        optimizer.zero_grad()
        predicted_ner, predicted_sa = model(inputs)
        alert_generation(predicted_ner, predicted_sa)
        loss = loss_fn(predicted_ner, ner, predicted_sa, sa)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)

    writer.add_scalar("train/loss", avg_loss, epoch)

    return avg_loss


@torch.no_grad()
def val_step(model, val_data, loss_fn, writer, epoch, device):
    """
    Evaluates the model on the validation data without updating weights and returning the current avarage loss and accuracy.
    """
    
    model.eval()
    total_loss = 0.0

    for _, tweet, caption, ner, sa in val_data:
        tweet = tweet.to(device).long()
        caption = caption.to(device).long()
        ner = ner.to(device).long()
        sa = sa.to(device).long()

        inputs = torch.cat((tweet, caption), dim=1)

        predicted_ner, predicted_sa = model(inputs)
        loss = loss_fn(predicted_ner, ner, predicted_sa, sa)
        total_loss += loss.item()
    
    avg_loss = total_loss / len(val_data)

    writer.add_scalar("validation/loss", avg_loss, epoch)

    return avg_loss


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
    print(f"Test NER Accuracy: {ner_accuracy:.4f}")
    print(f"Test SA Accuracy: {sa_accuracy:.4f}")
