import os
import torch
import imghdr
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



class ImageDataset(Dataset):
    def __init__(self, root_dirs):
        """
        Args:
            root_dirs (list): List of directories where to search for images.
        """
        self.image_paths = []
        self.image_names = []
        for root_dir in root_dirs:
            for file in os.listdir(root_dir):
                file_path = os.path.join(root_dir, file)
                # Save only the uncorrupted image files
                if file.endswith(".jpg") and imghdr.what(file_path) == "jpeg":
                    self.image_paths.append(file_path)
                    self.image_names.append(file)

        # Transform to tensor and normalize
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[1.0, 1.0, 1.0])
        ])


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return img_name, image
    

class CaptionDataset(Dataset):
    def __init__(self, dataloader, caption_file=None):
        """
        Args:
            image_data (list): List of images with their names.
            model: The model for generating captions.
            processor: Processor for the images.
            device: The device ('cpu' or 'cuda').
            caption_file: Path to CSV with pre-existing captions.
        """
        
        self.image_names = []
        self.captions = []

        # Check if CSV file with captions already exists
        if caption_file and os.path.exists(caption_file):
            df = pd.read_csv(caption_file, sep=";")
            self.image_names = df['image_name'].tolist()
            self.captions = df['caption'].tolist()
        
        else:

            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = "microsoft/git-base"
            processor = AutoProcessor.from_pretrained(checkpoint)
            model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

            try:
                for names, images in tqdm(dataloader, desc="Generating captions"):
                    images = images.to(device)
                    for name, image in zip(names, images):
                        inputs = processor(images=image, return_tensors="pt").to(device)
                        with torch.no_grad():
                            generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)
                            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                            self.image_names.append(name)
                            self.captions.append(caption)

                # Save captions to CSV if they were generated
                if caption_file:
                    df = pd.DataFrame({'image_name': self.image_names, 'caption': self.captions})
                    df.to_csv(caption_file, index=False, sep=";")
                    
            except KeyboardInterrupt:
                print("Caption generation interrupted. Saving partial results.")
                if caption_file:
                    df = pd.DataFrame({'image_name': self.image_names, 'caption': self.captions})
                    df.to_csv(caption_file, index=False, sep=";")
                raise

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        caption = self.captions[idx]
        return img_name, caption
    

class TextCaptionDataset(Dataset):

    def __init__(self, text_ner_path, caption_path):
        self.tweets = pd.read_csv(text_ner_path, sep=';')
        self.tweets = self.tweets[['id', 'tweet']]
        self.captions = pd.read_csv(caption_path, sep=';')
        self.data = self.tweets.merge(self.captions, left_on='id', right_on='image_name', how='inner')
    
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return item['id'], item['tweet'], item['caption']


def create_text_ner_csvs(paths, cases):
    for case in cases:
        if os.path.exists(f"data/csv/text_and_ner_{case}.csv"):
            continue
        df = pd.DataFrame(columns=['id', 'tweet', 'ner'])
        for path in paths:
            id = ''
            tweet = ''
            ner = ''
            with open(path+case+'.txt', 'r', encoding='utf-8') as file:
                idx = 0
                for line in file:
                    idx += 1
                    if line.strip() != '':
                        line = line.strip()
                        if line[:5] == "IMGID":
                            if id:
                                tweet = tweet.rstrip()
                                ner = ner.rstrip()
                                df.loc[len(df)] = [f"{id}.jpg", tweet, ner]
                            id = line[6:]
                            tweet = ''
                            ner = ''
                        else:
                            tokens = line.split()
                            if len(tokens) == 2:
                                if tokens[0] == ';':
                                    tokens[0] = ','
                                if tokens[1] == ';':
                                    tokens[1] = ','
                                tweet += f"{tokens[0]} "
                                ner += f" {tokens[1]} "
        df.to_csv(f"data/csv/text_and_ner_{case}.csv", sep=";")


def load_data(batch_size=32, num_workers=0):
    """
    Initializes the image and caption datasets and returns both.

    Args:
        batch_size (int): Batch size for loading data.
        num_workers (int): Number of workers for loading data.
        caption_file (str): Path to CSV file containing captions.

    Returns:
        image_dataset (ImageDataset): Dataset containing the images.
        caption_dataset (CaptionDataset): Dataset containing the generated captions.
    """
    root_dirs = ["data/twitter2015_images/", "data/twitter2017_images/"]
    caption_file = "data/csv/captionsImages.csv"
                
    image_dataset = ImageDataset(root_dirs)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    caption_dataset = CaptionDataset(dataloader, caption_file)

    create_text_ner_csvs(
        paths=["data/twitter2015/", "data/twitter2017/"],
        cases=["train", "valid", "test"]
    )

    return image_dataset, caption_dataset


if __name__ == "__main__":
    batch_size = 1
    num_workers = 0

    image_dataset, caption_dataset = load_data(batch_size=batch_size, num_workers=num_workers)
    
    sample_idx = 0
    img_name, caption = caption_dataset[sample_idx]

    print(f"Image Name: {img_name}")
    print(f"Caption: {caption}")