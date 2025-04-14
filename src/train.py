import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Final
from src.data import load_data
from src.models import MyModel
from src.utils import set_seed, save_model, MultiTaskLoss
from src.train_functions import train_step, val_step
from tqdm.auto import tqdm


DATA_PATH: Final[str] = "data/csv"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main():

    epochs = 1
    lr = 1e-3
    batch_size = 32
    hidden_size = 16
    ner_output_dim = 11
    weight_decay = 0

    print("1. Loading data...")
    train_data, val_data, _, = load_data(DATA_PATH, batch_size)

    print("2. Defining model and log writer...")
    name = f"model_lr_{lr}_hs_{hidden_size}_batch_{batch_size}_epochs_{epochs}"
    writer = SummaryWriter(f"runs/{name}")

    print("3. Creating model...")
    model = MyModel(hidden_dim=hidden_size, ner_output_dim=ner_output_dim).to(
        device
    ).float()

    print("4. Defining loss, optimizer and scheduler...")
    loss_fn = MultiTaskLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    
    print("5. Training and evaluating model...")
    for epoch in tqdm(range(epochs)):
        
        train_loss = train_step(model, train_data, loss_fn, optimizer, writer, epoch, device)
        val_loss = val_step(model, val_data, loss_fn, writer, epoch, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("validation/loss", val_loss, epoch)


    print("6. Saving model...")
    save_model(model, name)
    writer.close()


if __name__ == "__main__":
    main()
