import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root_dirs):
        """
        Args:
            root_dirs (list): Lista de carpetas donde buscar imágenes.
        """
        self.image_paths = []
        self.image_names = []
        for root_dir in root_dirs:
            for file in os.listdir(root_dir):
                if file.endswith(".jpg"):
                    self.image_paths.append(os.path.join(root_dir, file))
                    self.image_names.append(file)

        # Transform to tensor and normalize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return img_name, image


def load_image_data(batch_size=32, num_workers=0):
    """
    Carga las imágenes de ambas carpetas en DataLoaders.

    Args:
        batch_size (int): Tamaño del batch.
        num_workers (int): Número de hilos para cargar datos.

    Returns:
        DataLoader con imágenes y nombres.
    """
    root_dirs = ["data/twitter2015_images", "data/twitter2017_images"]
    dataset = ImageDataset(root_dirs)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader


dataloader = load_image_data()

for img_names, img_tensors in dataloader:
    print(img_names)
    print(img_tensors.shape)  # Batch de imágenes en forma de tensor
    break  # Para mostrar solo un batch
