import torch
import numpy as np
from torch.jit import RecursiveScriptModule
import os
import random

@torch.no_grad()
def parameters_to_double(model: torch.nn.Module) -> None:
    """
    This function transforms the model parameters to double.

    Args:
        model: pytorch model.
    """

    for param in model.parameters():
        param.data = param.data.double()


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model's state_dict in the 'models' folder.
    It should create the 'models' folder if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """
    
    # Create the 'models' folder if it doesn't exist
    if not os.path.isdir("models"):
        os.makedirs("models")
    
    # Save the model's state_dict
    torch.save(model.state_dict(), f"models/{name}.pt")
    
    print(f"Model saved as models/{name}.pt")
    return None



def load_model(model_class: torch.nn.Module, name: str, hidden_dim=16, ner_output_dim=11) -> torch.nn.Module:
    """
    This function is to load a model's state_dict from the 'models' folder.

    Args:
        model_class: The class of the model (e.g., MyModel).
        name: Name of the model to load (without the extension, e.g., name.pt).
        hidden_dim: Default value for hidden_dim.
        ner_output_dim: Default value for ner_output_dim.

    Returns:
        A model instance with the loaded state_dict.
    """
    
    # Create a model instance with the default arguments
    model = model_class(hidden_dim=hidden_dim, ner_output_dim=ner_output_dim)
    
    # Load the model's state_dict
    model.load_state_dict(torch.load(f"models/{name}.pt"))
    
    # Set the model to evaluation mode (important for inference)
    
    print(f"Model {name} loaded successfully! :-)))) almu esta feliz")
    
    return model

def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
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
    def __init__(self, ner_weight=1.0, sa_weight=1.0):
        super(MultiTaskLoss, self).__init__()

        self.ner_loss_fn = torch.nn.CrossEntropyLoss()
        self.sa_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.ner_weight = ner_weight
        self.sa_weight = sa_weight

    def forward(self, ner_logits, ner_labels, sa_logits, sa_labels):
        """
        Inputs:
        - ner_logits: [batch_size, seq_len, num_ner_tags]
        - ner_labels: [batch_size, seq_len]
        - sa_logits:  [batch_size]  (logits for binary classification)
        - sa_labels:  [batch_size] or [batch_size] (0 or 1 labels)
        """

        # Flatten NER predictions and labels for loss
        ner_loss = self.ner_loss_fn(
            ner_logits.view(-1, ner_logits.size(-1)),
            ner_labels.view(-1)
        )

        # Ensure labels are float for BCEWithLogitsLoss
        sa_labels = sa_labels.float().view(-1, 1)  # ensure shape is [batch_size, 1]
        sa_loss = self.sa_loss_fn(sa_logits, sa_labels)

        total_loss = self.ner_weight * ner_loss + self.sa_weight * sa_loss
        return total_loss
    

import torch

class MultiTaskAccuracy(torch.nn.Module):
    def __init__(self, ner_threshold=0.5):
        super(MultiTaskAccuracy, self).__init__()

        self.ner_threshold = ner_threshold  # Threshold for NER (e.g., 0.5)

    def forward(self, ner_logits, ner_labels, sa_logits, sa_labels):
        """
        Inputs:
        - ner_logits: [batch_size, seq_len, num_ner_tags] (raw logits)
        - ner_labels: [batch_size, seq_len]
        - sa_logits:  [batch_size]  (logits for binary classification)
        - sa_labels:  [batch_size] or [batch_size] (0 or 1 labels)
        """

        # --- NER Accuracy ---
        # Find the predicted NER labels by taking the argmax along the last dimension (num_ner_tags)
        ner_predictions = torch.argmax(ner_logits, dim=-1)  # [batch_size, seq_len]

        # Flatten the NER logits and labels for accuracy calculation (ignoring padding)
        non_padding_mask = ner_labels != -100  # assuming padding labels are -100
        ner_correct = (ner_predictions == ner_labels) & non_padding_mask

        ner_accuracy = ner_correct.sum().float() / non_padding_mask.sum().float()

        # --- SA Accuracy ---
        # Convert logits to predictions (0 or 1)
        sa_predictions = (torch.sigmoid(sa_logits) > self.ner_threshold).float()  # Apply sigmoid and threshold

        # Compare with the true labels (0 or 1)
        sa_correct = (sa_predictions == sa_labels).float()

        sa_accuracy = sa_correct.sum() / sa_labels.size(0)

        # Return both accuracies as a tuple
        return ner_accuracy, sa_accuracy
