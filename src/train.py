import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Final
from src.data import load_data
from src.models import MyModel
from src.utils import set_seed, save_model, MultiTaskLoss, MultiTaskAccuracy
from src.train_functions import train_step, val_step
from tqdm.auto import tqdm


DATA_PATH: Final[str] = "data/csv"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main():
    epochs = 20
    lr = 1e-3
    batch_size = 32
    hidden_size = 16
    ner_output_dim = 12

    print("1. Loading data...")
    (train_data, val_data, _, loss_weights) = load_data(DATA_PATH, batch_size)

    print("2. Defining model and log writer...")
    name = f"model_lr_{lr}_hs_{hidden_size}_batch_{batch_size}_epochs_{epochs}"
    writer = SummaryWriter(f"runs/{name}")

    print("3. Creating model...")
    model = (
        MyModel(hidden_dim=hidden_size, ner_output_dim=ner_output_dim)
        .to(device)
        .float()
    )

    print("4. Defining loss and optimizer...")
    loss_fn = MultiTaskLoss(model, device=device, loss_weights=loss_weights)
    accuracy_fn = MultiTaskAccuracy(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("5. Training and evaluating model...")
    for epoch in tqdm(range(epochs)):
        _, _ = train_step(
            model, train_data, loss_fn, optimizer, writer, epoch, device, accuracy_fn
        )
        _, _ = val_step(model, val_data, writer, epoch, device, accuracy_fn)

    print("6. Saving model...")
    save_model(model, name)
    writer.close()


if __name__ == "__main__":
    main()
