import os
import torch
import imghdr
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class ImageDataset(Dataset):
    """
    Custom PyTorch dataset to load JPEG images from multiple directories.

    Args:
        root_dirs (list): List of directories to search for image files.
    """

    def __init__(self, root_dirs):
        self.image_paths = []
        self.image_names = []

        # Collect valid JPEG images from all directories
        for root_dir in root_dirs:
            for file in os.listdir(root_dir):
                file_path = os.path.join(root_dir, file)
                if file.endswith(".jpg") and imghdr.what(file_path) == "jpeg":
                    self.image_paths.append(file_path)
                    self.image_names.append(file)

        # Define image transformation: resizing, tensor conversion, normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1.0, 1.0, 1.0]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return img_name, image


def create_captions(root_dirs, caption_file, batch_size=1, num_workers=0):
    """
    Generates image captions using a pre-trained model and stores them in a CSV file.

    Args:
        root_dirs (list): Directories containing image files.
        caption_file (str): Path to store the generated captions.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of worker threads for DataLoader.
    """

    if os.path.exists(caption_file):
        return

    image_names = []
    captions = []

    dataset = ImageDataset(root_dirs)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

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
                    generated_ids = model.generate(
                        pixel_values=inputs.pixel_values, max_length=50
                    )
                    caption = processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]

                    image_names.append(name)
                    captions.append(caption)

        # Save generated captions to CSV
        if caption_file:
            df = pd.DataFrame({"image_name": image_names, "caption": captions})
            df.to_csv(caption_file, index=False, sep=";")

    except KeyboardInterrupt:
        print("Caption generation interrupted. Saving partial results.")
        if caption_file:
            df = pd.DataFrame({"image_name": image_names, "caption": captions})
            df.to_csv(caption_file, index=False, sep=";")


def process_data(paths, caption_file, cases):
    """
    Builds a CSV dataset for training using text files and generated image captions.

    Args:
        paths (list): Directories containing .txt files (e.g., with tweets and NER).
        caption_file (str): Path to CSV with image captions.
        cases (list): List of dataset partitions to process (e.g., ["train", "valid", "test"]).
    """

    sentiment_analyzer = pipeline("sentiment-analysis")
    captions_df = pd.read_csv(caption_file, sep=";")

    for case in cases:
        print(f"Processing {case}...")

        output_csv = f"data/csv/{case}.csv"
        if os.path.exists(output_csv):
            continue

        # Create empty DataFrame for the processed data
        df = pd.DataFrame(columns=["id", "tweet", "caption", "ner", "sa"])

        for path in paths:
            current_id, tweet, ner = "", "", ""

            with open(os.path.join(path, f"{case}.txt"), "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith("IMGID"):
                        # Save previous sample if valid
                        if (
                            current_id
                            and f"{current_id}.jpg" in captions_df["image_name"].values
                        ):
                            caption = captions_df.loc[
                                captions_df["image_name"] == f"{current_id}.jpg",
                                "caption",
                            ].values[0]
                            ner += "O " * len(caption.split())
                            sentiment = sentiment_analyzer(tweet + caption)[0]["label"]
                            df.loc[len(df)] = [
                                current_id,
                                tweet.rstrip(),
                                caption,
                                ner.rstrip(),
                                sentiment,
                            ]

                        # Start new record
                        current_id = line[6:]
                        tweet = ""
                        ner = ""

                    else:
                        # Append tokens and NER labels
                        tokens = line.split()
                        if len(tokens) == 2:
                            word, tag = tokens
                            if word == ";":
                                word = ","
                            if tag == ";":
                                tag = ","
                            tweet += f"{tokens[0]} "
                            ner += f"{tokens[1]} "

                # Handle final entry
                if (
                    current_id
                    and f"{current_id}.jpg" in captions_df["image_name"].values
                ):
                    caption = captions_df.loc[
                        captions_df["image_name"] == f"{current_id}.jpg", "caption"
                    ].values[0]
                    ner += "O " * len(caption.split())
                    sentiment = sentiment_analyzer(tweet + caption)[0]["label"]
                    df.loc[len(df)] = [
                        current_id,
                        tweet.rstrip(),
                        caption,
                        ner.rstrip(),
                        sentiment,
                    ]

        df.to_csv(output_csv, sep=";", index=False)


def main():
    """
    Main pipeline to generate image captions and create processed CSV datasets.
    """

    os.makedirs("data/csv", exist_ok=True)

    data_dirs = ["data/twitter2015/", "data/twitter2017/"]
    image_dirs = ["data/twitter2015_images/", "data/twitter2017_images/"]
    caption_file = "data/csv/captionsImages.csv"
    cases = ["train", "valid", "test"]

    create_captions(image_dirs, caption_file)
    process_data(data_dirs, caption_file, cases)


if __name__ == "__main__":
    main()
