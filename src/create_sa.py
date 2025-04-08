import os
import torch
import imghdr
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM

class TextDataset(Dataset):

    def __init__(self, path_txt, path_csv): 
        self.captions = pd.read_csv(path_csv, sep=";")

    def __len__(self):
        return len(self.captions)
    

if __name__ == "__main__":

    data = TextDataset("", "captionsImages.csv")
    print(len(data))