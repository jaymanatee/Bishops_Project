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
    model_class, name: str, hidden_dim=128, ner_output_dim=12
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

    def __init__(
        self, model, device, loss_weights, ner_weight=1, sa_weight=1.0, crf_weight=0.01
    ):
        super(MultiTaskLoss, self).__init__()
        self.model = model

        # Adjust class weights for NER to address class imbalance
        weights = loss_weights
        weights = weights.to(device)

        self.ner_loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        self.sa_loss_fn = torch.nn.BCEWithLogitsLoss()
        self.ner_weight = ner_weight
        self.sa_weight = sa_weight
        self.crf_weight = crf_weight
        self.device = device

    def forward(self, ner_logits, ner_labels, sa_logits, sa_labels):
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
        mask = (ner_labels != 11).to(torch.bool).to(self.device)
        crf_loss = -self.model.crf(
            ner_logits.detach(), ner_labels, mask=mask, reduction="mean"
        )

        ner_loss = self.ner_loss_fn(
            ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1)
        )
        sa_loss = self.sa_loss_fn(sa_logits.view(-1, 1), sa_labels.float().view(-1, 1))

        total_loss = (
            self.ner_weight * ner_loss
            + self.sa_weight * sa_loss
            + self.crf_weight * crf_loss
        )
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

    def forward(
        self, ner_logits, ner_predictions_decoded, ner_labels, sa_logits, sa_labels
    ):
        """
        Inputs:
        - ner_predictions: [batch_size, seq_len]
        - ner_labels: [batch_size, seq_len]
        - sa_logits:  [batch_size]  (logits for binary classification)
        - sa_labels:  [batch_size] or [batch_size] (0 or 1 labels)
        """

        # --- NER Accuracy ---
        # Find the predicted NER labels by taking the argmax along the last dimension (num_ner_tags)
        # Do a counter to check the number of appearances of each label
        ner_logits = ner_logits.to(self.device)
        ner_predictions_decoded = ner_predictions_decoded.to(self.device)

        ner_predictions_raw = torch.argmax(ner_logits, dim=-1)
        # Flatten the NER logits and labels for accuracy calculation (ignoring padding)
        non_padding_mask = ner_labels != 11  # assuming padding labels are -11
        non_padding_mask_wo = (ner_labels != 11) & (ner_labels != 0)

        ner_correct_raw = (ner_predictions_raw == ner_labels) & non_padding_mask
        ner_correct_wo_raw = (ner_predictions_raw == ner_labels) & non_padding_mask_wo
        ner_accuracy_raw = (
            ner_correct_raw.sum().float() / non_padding_mask.sum().float()
        )
        ner_accuracy_wo_raw = (
            ner_correct_wo_raw.sum().float() / non_padding_mask_wo.sum().float()
        )

        ner_correct_decoded = (ner_predictions_decoded == ner_labels) & non_padding_mask
        ner_correct_wo_decoded = (
            ner_predictions_decoded == ner_labels
        ) & non_padding_mask_wo
        ner_accuracy_decoded = (
            ner_correct_decoded.sum().float() / non_padding_mask.sum().float()
        )
        ner_accuracy_wo_decoded = (
            ner_correct_wo_decoded.sum().float() / non_padding_mask_wo.sum().float()
        )

        # --- SA Accuracy ---
        # Convert logits to predictions (0 or 1)

        sa_predictions = (
            torch.sigmoid(sa_logits) > self.sa_threshold
        ).float()  # Apply sigmoid and threshold

        # Compare with the true labels (0 or 1)
        sa_correct = (sa_predictions == sa_labels).float()
        sa_accuracy = sa_correct.sum() / sa_labels.size(0)

        # Return both accuracies as a tuple
        return (
            ner_accuracy_raw,
            ner_accuracy_wo_raw,
            ner_accuracy_decoded,
            ner_accuracy_wo_decoded,
            sa_accuracy,
        )
