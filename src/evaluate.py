import torch
from typing import Final
from src.data import load_data
from src.utils import set_seed, load_model, MultiTaskAccuracy
from src.train_functions import t_step
from src.models import MyModel, MyDeepSeek


DATA_PATH: Final[str] = "data/csv"
NUM_CLASSES: Final[int] = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    batch_size: int = 32

    print("1. Loading data...")
    (
        _,
        _,
        test_data,
        _,
    ) = load_data(DATA_PATH, batch_size)

    print("2. Loading trained model...")
    model = load_model(MyModel, "crf_128").to(device)

    print("3. Evaluating model on test data...")
    model.eval()
    accuracy = MultiTaskAccuracy(device=device)
    with torch.no_grad():
        (
            ner_accuracy_raw,
            ner_accuracy_wo_raw,
            ner_accuracy_decoded,
            ner_accuracy_wo_decoded,
            sa_accuracy,
            alert_inputs,
        ) = t_step(
            model=model, test_data=test_data, accuracy_fn=accuracy, device=device
        )

    print(f"NER Accuracy (without crf): {ner_accuracy_raw:.4f}")
    print(f"NER Accuracy (wtihout crf) without O: {ner_accuracy_wo_raw:.4f}")
    print(f"NER Accuracy (with crf): {ner_accuracy_decoded:.4f}")
    print(f"NER Accuracy (with crf) without O: {ner_accuracy_wo_decoded:.4f}")
    print(f"SA Accuracy: {sa_accuracy:.4f}")

    # Generation of alerts
    deep_model = MyDeepSeek(inputs=alert_inputs, show_last=True)
    alerts = deep_model.generate_alerts(num_alerts=10)

    for alert in alerts:
        print(alert)


if __name__ == "__main__":
    main()
