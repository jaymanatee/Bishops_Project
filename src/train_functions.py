import torch
from tqdm.auto import tqdm


def idx2tag(idx):
    """
    Converts an index to its corresponding Named Entity Recognition (NER) tag string.

    Args:
        idx (int): Index to convert.

    Returns:
        str: Corresponding NER tag.
    """
    idx2tag = {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-LOC",
        4: "I-LOC",
        5: "B-ORG",
        6: "I-ORG",
        7: "B-OTHER",
        8: "I-OTHER",
        9: "B-MISC",
        10: "I-MISC",
        11: "PAD",
    }
    return idx2tag.get(idx)


@torch.enable_grad()
def train_step(
    model, train_data, loss_fn, optimizer, writer, epoch, device, accuracy_fn
):
    """
    Performs a single training epoch over the training dataset.

    Args:
        model (nn.Module): The model to train.
        train_data (DataLoader): DataLoader for training data.
        loss_fn (function): Custom loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        writer (SummaryWriter): TensorBoard writer for logging.
        epoch (int): Current epoch number.
        device (torch.device): Device to run computations on (CPU/GPU).
        accuracy_fn (function): Function to compute accuracy for NER and sentiment.

    Returns:
        tuple: Average NER accuracy and sentiment analysis accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_ner_accuracy = 0
    total_sa_accuracy = 0
    total_batches = 0

    for _, tweet, caption, ner, sa in tqdm(train_data):
        tweet = tweet.to(device).long()
        caption = caption.to(device).long()
        ner = ner.to(device).long()
        sa = sa.to(device).long()

        inputs = torch.cat((tweet, caption), dim=1)

        optimizer.zero_grad()
        predicted_ner, predicted_ner_decoded, predicted_sa = model(inputs)
        loss = loss_fn(predicted_ner, ner, predicted_sa, sa, last_epochs=epoch > 19)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        ner_accuracy, sa_accuracy = accuracy_fn(
            predicted_ner_decoded, ner, predicted_sa, sa
        )
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
    Performs a validation step without updating model weights.

    Args:
        model (nn.Module): The model to evaluate.
        val_data (DataLoader): DataLoader for validation data.
        writer (SummaryWriter): TensorBoard writer for logging.
        epoch (int): Current epoch number.
        device (torch.device): Device to run computations on (CPU/GPU).
        accuracy_fn (function): Function to compute accuracy for NER and sentiment.

    Returns:
        tuple: Average NER accuracy and sentiment analysis accuracy for validation data.
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

        _, predicted_ner_decoded, predicted_sa = model(inputs)
        ner_accuracy, sa_accuracy = accuracy_fn(
            predicted_ner_decoded, ner, predicted_sa, sa
        )
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
    Performs a test step on the test dataset and returns predictions along with metrics.

    Args:
        model (nn.Module): The trained model to evaluate.
        test_data (DataLoader): DataLoader for test data.
        accuracy_fn (function): Function to compute accuracy for NER and sentiment.
        device (torch.device): Device to run computations on (CPU/GPU).

    Returns:
        tuple: Average NER accuracy, sentiment accuracy, and predictions as list of tuples.
    """
    model.eval()
    total_ner_accuracy = 0
    total_sa_accuracy = 0
    total_batches = 0

    all_data = []

    for _, tweet, caption, ner, sa in test_data:
        tweet = tweet.to(device).long()
        caption = caption.to(device).long()
        ner = ner.to(device).long()
        sa = sa.to(device).long()

        inputs = torch.cat((tweet, caption), dim=1)

        _, predicted_ner_decoded, predicted_sa = model(inputs)

        ner_accuracy, sa_accuracy = accuracy_fn(
            predicted_ner_decoded, ner, predicted_sa, sa
        )
        total_ner_accuracy += ner_accuracy.item()
        total_sa_accuracy += sa_accuracy.item()
        total_batches += 1

        ner_pred_list = predicted_ner_decoded.tolist()
        sa_pred_list = (torch.sigmoid(predicted_sa) > 0.5).int().tolist()

        for t, c, ner_p, sa_p in zip(tweet, caption, ner_pred_list, sa_pred_list):
            ner_tags = [idx2tag(idx) for idx in ner_p]
            all_data.append((t, c, ner_tags, sa_p))

    ner_accuracy = total_ner_accuracy / total_batches
    sa_accuracy = total_sa_accuracy / total_batches

    return ner_accuracy, sa_accuracy, all_data
