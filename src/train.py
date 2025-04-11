import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Final
from src.data import load_data
from src.models import NerSAModel
from src.utils import set_seed, save_model
from src.train_functions import train_step, val_step


DATA_PATH: Final[str] = "data/csv"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main():

    epochs = 100
    lr = 1e-3
    batch_size = 32
    hidden_size = 64
    num_layers = 1
    weight_decay = 0

    print("1. Loading data...")
    train_data, val_data, _, = load_data(DATA_PATH, batch_size)

    print("2. Defining model and log writer...")
    name = f"model_lr_{lr}_hs_{hidden_size}_batch_{batch_size}_epochs_{epochs}"
    writer = SummaryWriter(f"runs/{name}")

    print("3. Creating model...")
    model = NerSAModel(hidden_size=hidden_size, num_layers=num_layers).to(
        device
    ).float()

    print("4. Defining loss, optimizer and scheduler...")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("5. Training and evaluating model...")
    for epoch in range(epochs):
        
        train_loss, train_acc = train_step(model, train_data, loss_fn, optimizer, device)
        val_loss, val_acc = val_step(model, val_data, loss_fn, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("validation/loss", val_loss, epoch)
        writer.add_scalar("validation/accuracy", val_acc, epoch)

        scheduler.step()

    print("6. Saving model...")
    save_model(model, name)
    writer.close()


if __name__ == "__main__":
    main()
