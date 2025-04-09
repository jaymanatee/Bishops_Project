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
from transformers import pipeline


class ImageDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes desde múltiples directorios.

    Args:
        root_dirs (list): Lista de directorios donde buscar las imágenes.
    """
    def __init__(self, root_dirs):
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
    

def create_captions(root_dirs, caption_file, batch_size=1, num_workers=0):
    if os.path.exists(caption_file):
        return
    image_names = []
    captions = []
    image_dataset = ImageDataset(root_dirs)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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
                    image_names.append(name)
                    captions.append(caption)

        # Save captions to CSV if they were generated
        if caption_file:
            df = pd.DataFrame({'image_name': image_names, 'caption': captions})
            df.to_csv(caption_file, index=False, sep=";")
            
    except KeyboardInterrupt:
        print("Caption generation interrupted. Saving partial results.")
        if caption_file:
            df = pd.DataFrame({'image_name': image_names, 'caption': captions})
            df.to_csv(caption_file, index=False, sep=";")
        raise


def process_data(paths, caption_file, cases):
    """
    A partir de las captions creadas, crea archivos CSV conteniendo id, tweet, caption, ner y sa para 
    futuro entrenamiento
    Args:
        paths (list): Directorios donde se encuentran los archivos .txt.
        caption_file (string): Ruta de las captions en csv
        cases (list): Casos a procesar (por ejemplo, train, valid, test).
    """
    sentiment_analyzer = pipeline("sentiment-analysis")
    captions = pd.read_csv(caption_file, sep=';')
    for case in cases:
        print(f'processing {case}...')
        if os.path.exists(f"data/csv/{case}.csv"):
            continue
        df = pd.DataFrame(columns=['id', 'tweet', 'caption', 'ner', 'sa'])
        for path in paths:
            id, tweet, ner = '', '', ''
            with open(path+case+'.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip() != '':
                        line = line.strip()
                        if line[:5] == "IMGID":
                            if id and f'{id}.jpg' in captions['image_name'].values:
                                caption = captions.loc[captions['image_name'] == f'{id}.jpg', 'caption'].values[0]
                                ner += "O " * len(caption.split())
                                sa = sentiment_analyzer(tweet + caption)[0]['label']
                                df.loc[len(df)] = [id, tweet.rstrip(), caption, ner.rstrip(), sa]
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
                                ner += f"{tokens[1]} "
                if f'{id}.jpg' in captions['image_name'].values:
                    caption = captions.loc[captions['image_name'] == f'{id}.jpg', 'caption'].values[0]
                    ner += "O " * len(caption.split())
                    sa = sentiment_analyzer(tweet + caption)[0]['label']
                    df.loc[len(df)] = [id, tweet.rstrip(), caption, ner.rstrip(), sa]
        df.to_csv(f"data/csv/{case}.csv", sep=";", index=False)


def main():
    """
    Carga y prepara los datasets con todos los datos necesarios en forma de csv.
    """
    if not os.path.exists('data/csv'):
        os.mkdir('data/csv')
    data_dirs = ["data/twitter2015/", "data/twitter2017/"]
    image_dirs = ["data/twitter2015_images/", "data/twitter2017_images/"]
    caption_file = "data/csv/captionsImages.csv"
    cases = ["train", "valid", "test"]
    
    create_captions(image_dirs, caption_file)
    process_data(data_dirs, caption_file, cases)


if __name__ == "__main__": 
    main()