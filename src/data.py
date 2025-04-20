import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import warnings

# Suppress warnings from pandas and other libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class TweetDataset(Dataset):
    """
    Custom PyTorch dataset for loading and tokenizing tweet-caption-annotated data.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        tokenizer (BertTokenizer): Tokenizer from HuggingFace Transformers to convert text to token IDs.
    """

    def __init__(self, file_path: str, tokenizer: BertTokenizer) -> None:
        # Load the dataset using pandas
        self.dataset = pd.read_csv(file_path, sep=";")
        self.tokenizer = tokenizer

        # Tokenize the tweet and caption columns
        self.dataset["tweet"] = self.dataset["tweet"].apply(self.tokenize_text)
        self.dataset["caption"] = self.dataset["caption"].apply(self.tokenize_text)

        # Convert NER tag strings into integer IDs using tag2idx
        tokenize_ner = lambda ner: torch.tensor([tag2idx(tag) for tag in ner.split()])
        self.dataset["ner"] = self.dataset["ner"].apply(tokenize_ner)

        # Convert sentiment analysis labels to binary: POSITIVE → 1, else 0
        tokenize_sa = (
            lambda label: torch.tensor(1) if label == "POSITIVE" else torch.tensor(0)
        )
        self.dataset["sa"] = self.dataset["sa"].apply(tokenize_sa)

    def tokenize_text(self, text: str) -> torch.Tensor:
        """
        Tokenizes input text and converts it to a list of token IDs using the BERT tokenizer.

        Args:
            text (str): The raw input text.

        Returns:
            torch.Tensor: A tensor of token IDs.
        """
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset.iloc[index]
        return row["id"], row["tweet"], row["caption"], row["ner"], row["sa"]

    def get_weights(self, num_classes):
        frequencies = [0 for _ in range(num_classes)]
        total = 0
        for ner in self.dataset["ner"].values:
            for label in ner:
                frequencies[label.item()] += num_classes
            total += len(ner)
        weights = total / torch.tensor(frequencies)
        weights[-1] = 0.00
        return weights


def tag2idx(tag):
    """
    Converts a NER tag string to its corresponding index.

    Args:
        tag (str): Named Entity Recognition (NER) tag.

    Returns:
        int: The corresponding index of the tag.
    """

    tag2idx = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-LOC": 3,
        "I-LOC": 4,
        "B-ORG": 5,
        "I-ORG": 6,
        "B-OTHER": 7,
        "I-OTHER": 8,
        "B-MISC": 9,
        "I-MISC": 10,
        "PAD": 11,
    }
    return tag2idx.get(tag)


def idx2tag(idx):
    """
    Converts a tag index back to its corresponding NER string.

    Args:
        idx (int): Index of the NER tag.

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


def collate_fn(batch):
    """
    Collate function to pad the tweet, caption, and NER tag sequences to the same length.

    Ensures that the padded tweet + caption length matches the NER tag length.
    Appends the combined tweet and caption tensor to NER list temporarily to compute correct length.

    Args:
        batch (list): List of samples returned by Dataset.__getitem__.

    Returns:
        Tuple: Batched IDs, tweets, captions, NERs, sentiment labels.
    """
    ids = [item[0] for item in batch]
    tweets = [torch.tensor(item[1]) for item in batch]
    captions = [torch.tensor(item[2]) for item in batch]
    ners = [torch.tensor(item[3]) for item in batch]
    sentiment_labels = torch.tensor([item[4] for item in batch])

    tweet_padded = pad_sequence(tweets, padding_value=0, batch_first=True)
    caption_padded = pad_sequence(captions, padding_value=0, batch_first=True)

    # Append combined tokenized tweet + caption for NER padding length matching
    ners.append(torch.cat((tweet_padded[0], caption_padded[0])))
    ner_padded = pad_sequence(ners, padding_value=11, batch_first=True)
    ner_padded = ner_padded[:-1]  # Remove the temporary sequence added above

    return ids, tweet_padded, caption_padded, ner_padded, sentiment_labels


def load_data(path: str, batch_size=64) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads training, validation, and test sets from the given path and returns DataLoaders.

    Args:
        path (str): Directory path containing 'train.csv', 'valid.csv', and 'test.csv'.
        batch_size (int, optional): Number of samples per batch. Defaults to 64.

    Returns:
        tuple: DataLoaders for train, validation, test datasets and weights tensor for CE Loss
               computed with tag frequency.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = TweetDataset(os.path.join(path, "train.csv"), tokenizer)
    val_dataset = TweetDataset(os.path.join(path, "valid.csv"), tokenizer)
    test_dataset = TweetDataset(os.path.join(path, "test.csv"), tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    loss_weights = train_dataset.get_weights(12)

    return train_dataloader, val_dataloader, test_dataloader, loss_weights
