import torch
from torch.jit import RecursiveScriptModule
from typing import Final
from src.data import load_data
from src.utils import set_seed, load_model
from src.train_functions import t_step

DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    
    batch_size: int = 32

    print("1. Loading data...")
    _, _, test_data, = load_data(DATA_PATH, batch_size)
    
    print("2. Loading trained model...")
    model: RecursiveScriptModule = load_model("best_model").to(device)

    print("3. Evaluating model on test data...")
    model.eval()
    with torch.no_grad():
        test_accuracy = t_step(model=model, test_data=test_data, device=device)

    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
