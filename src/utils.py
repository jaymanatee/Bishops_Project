import torch
import numpy as np
import os
import random


@torch.no_grad()
def parameters_to_double(model: torch.nn.Module) -> None:
    """
    Converts all model parameters to double precision (torch.float64).

    Args:
        model: A PyTorch model.
    """
    for param in model.parameters():
        param.data = param.data.double()


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    Saves the model's state_dict to the 'models' directory.
    Creates the directory if it does not exist.

    Args:
        model: A PyTorch model.
        name: File name without extension (e.g., 'my_model' -> 'my_model.pt').
    """
    if not os.path.isdir("models"):
        os.makedirs("models")

    # Save the model's state_dict
    torch.save(model.state_dict(), f"models/{name}.pt")

    print(f"Model saved as models/{name}.pt")
    return None


def load_model(
    model_class, name: str, hidden_dim=16, ner_output_dim=12
) -> torch.nn.Module:
    """
    Loads a model from a saved state_dict in the 'models' folder.

    Args:
        model_class: The model class (e.g., MyModel).
        name: The filename of the model to load (without '.pt').
        hidden_dim: Hidden layer size for the model.
        ner_output_dim: Number of output labels for the NER task.

    Returns:
        A model instance with weights loaded.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_class(hidden_dim=hidden_dim, ner_output_dim=ner_output_dim).to(device)

    # Load the model to the correct device
    state_dict = torch.load(f"models/{name}.pt", map_location=device)
    model.load_state_dict(state_dict)

    print(f"Model '{name}' loaded successfully on {device} ")
    return model


def set_seed(seed: int) -> None:
    """
    Sets all relevant seeds to ensure deterministic behavior.

    Args:
        seed: An integer value to fix randomness.
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


class MultiTaskLoss(torch.nn.Module):
    """
    Multi-task loss function for NER and Sentiment Analysis (SA).
    Combines cross-entropy loss for NER, binary cross-entropy for SA,
    and optionally CRF loss for the final epochs.
    """

    def __init__(self, model, device, ner_weight=1, sa_weight=1.0, crf_weight=0.01):
        super(MultiTaskLoss, self).__init__()
        self.model = model

        # Adjust class weights for NER to address class imbalance
        weights = 10 * torch.tensor(
            [
                0,
                1 / 0.002,
                1 / 0.002,
                1 / 0.002,
                1 / 0.002,
                1 / 0.002,
                1 / 0.002,
                1 / 0.002,
                1 / 0.002,
                1 / 0.002,
                1 / 0.002,
                0,
            ]
        )
        weights = weights.to(device)

        self.ner_loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        self.sa_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.ner_weight = ner_weight
        self.sa_weight = sa_weight
        self.crf_weight = crf_weight
        self.device = device

    def forward(self, ner_logits, ner_labels, sa_logits, sa_labels, last_epochs=False):
        """
        Computes the combined multi-task loss.

        Args:
            ner_logits: Tensor of shape [batch_size, seq_len, num_ner_tags]
            ner_labels: Tensor of shape [batch_size, seq_len]
            sa_logits: Tensor of shape [batch_size] (binary sentiment prediction)
            sa_labels: Tensor of shape [batch_size]
            last_epochs: If True, includes CRF loss.

        Returns:
            Total combined loss.
        """
        total_loss = 0

        if last_epochs:
            mask = (ner_labels != 11).to(torch.bool).to(self.device)
            crf_loss = -self.model.crf(
                ner_logits, ner_labels, mask=mask, reduction="mean"
            )
            total_loss += self.crf_weight * crf_loss

        ner_loss = self.ner_loss_fn(
            ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1)
        )
        sa_loss = self.sa_loss_fn(sa_logits.view(-1, 1), sa_labels.float().view(-1, 1))

        total_loss += self.ner_weight * ner_loss + self.sa_weight * sa_loss

        return total_loss


class MultiTaskAccuracy(torch.nn.Module):
    """
    Computes accuracy for both NER and Sentiment Analysis tasks.
    """

    def __init__(self, device, ner_threshold=0.5, sa_threshold=0.5):
        super(MultiTaskAccuracy, self).__init__()
        self.ner_threshold = ner_threshold
        self.sa_threshold = sa_threshold
        self.device = device

    def forward(self, ner_predictions, ner_labels, sa_logits, sa_labels):
        """
        Computes accuracy for each task separately.

        Args:
            ner_predictions: Tensor [batch_size, seq_len] (predicted NER tags)
            ner_labels: Tensor [batch_size, seq_len] (ground truth NER tags)
            sa_logits: Tensor [batch_size] (logits for SA)
            sa_labels: Tensor [batch_size] (binary sentiment labels)

        Returns:
            Tuple: (NER accuracy, SA accuracy)
        """
        # NER accuracy
        non_padding_mask = ner_labels != 11
        ner_correct = (ner_predictions == ner_labels) & non_padding_mask
        ner_accuracy = ner_correct.sum().float() / non_padding_mask.sum().float()

        # SA accuracy
        sa_predictions = (torch.sigmoid(sa_logits) > self.sa_threshold).float()
        sa_correct = (sa_predictions == sa_labels).float()
        sa_accuracy = sa_correct.sum() / sa_labels.size(0)

        return ner_accuracy, sa_accuracy
