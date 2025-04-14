import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class TweetDataset(Dataset):
    """
    Custom PyTorch dataset to load the data from CSV files and tokenize text columns using BERT.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        tokenizer (BertTokenizer): The BERT tokenizer to use for tokenizing text data.
    """

    def __init__(self, file_path: str, tokenizer: BertTokenizer) -> None:
        
        # Load the dataset
        self.dataset = pd.read_csv(file_path, sep=";")
        self.tokenizer = tokenizer

        # Tokenize string input columns
        self.dataset['tweet'] = self.dataset['tweet'].apply(self.tokenize_text)
        self.dataset['caption'] = self.dataset['caption'].apply(self.tokenize_text)
        tag2idx = {
            'O': 0,
            'B-PER': 1,
            'I-PER': 2,
            'B-LOC': 3,
            'I-LOC': 4,
            'B-ORG': 5,
            'I-ORG': 6,
            'B-OTHER': 7,
            'I-OTHER': 8,
            'B-MISC': 9,
            'I-MISC': 10
        }
        tokenize_ner = lambda ner: torch.tensor([tag2idx[token] for token in ner.split()])
        self.dataset['ner'] = self.dataset['ner'].apply(tokenize_ner)
        tokenize_sa = lambda label: torch.tensor(1) if label=="POSITIVE" else torch.tensor(0)
        self.dataset['sa'] = self.dataset['sa'].apply(tokenize_sa)
        
    def tokenize_text(self, text: str) -> torch.Tensor:
        """
        Tokenizes the input text and converts it to token IDs using the BERT tokenizer.
        """
        
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset.iloc[index]
        return row["id"], row["tweet"], row["caption"], row["ner"], row["sa"]


def collate_fn(batch):
    """
    Custom collate function for padding the sequences in the batch and ensuring
    that the combined length of `tweet` + `caption` matches the length of `ner`.
    """
    
    ids = [item[0] for item in batch]
    tweets = [torch.tensor(item[1]) for item in batch]
    captions = [torch.tensor(item[2]) for item in batch]
    ners = [torch.tensor(item[3]) for item in batch]
    sentiment_labels = torch.tensor([item[4] for item in batch])
    
    tweet_padded = pad_sequence(tweets, padding_value=0, batch_first=True)
    caption_padded = pad_sequence(captions, padding_value=0, batch_first=True)
    ners.append(torch.cat((tweet_padded[0], caption_padded[0])))
    ner_padded = pad_sequence(ners, padding_value=0, batch_first=True)
    ner_padded = ner_padded[:-1]
    
    return ids, tweet_padded, caption_padded, ner_padded, sentiment_labels

def load_data(path: str, batch_size=64) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads the train, validation, and test data from the given CSV files and returns
    the respective DataLoaders.

    Args:
        path (str): Path to the directory containing the CSV files.
        batch_size (int, optional): Batch size for DataLoader. Default is 64.

    Returns:
        tuple: A tuple containing the train, validation, and test DataLoaders.
    """
    
    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets for train, validation, and test sets
    train_dataset = TweetDataset(os.path.join(path, "train.csv"), tokenizer)
    val_dataset = TweetDataset(os.path.join(path, "valid.csv"), tokenizer)
    test_dataset = TweetDataset(os.path.join(path, "test.csv"), tokenizer)

    # Create the DataLoader instances
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

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":

    embedder = BertModel.from_pretrained('bert-base-uncased', ignore_mismatched_sizes=True)
    train_dataloader, val_dataloader, test_dataloader = load_data("data/csv/")
    for item in train_dataloader:
        print(item[3][0])
        break