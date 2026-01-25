from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import zipfile
import os
from glob import glob
import gdown


def _download_and_extract(url, download_path, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)

    if not os.path.exists(download_path):
        print(f"Downloading dataset from {url}...")
        gdown.download(url, download_path, quiet=False)

    print(f"Extracting dataset to {extract_path}...")
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # os.remove(download_path)
        
        
def get_roi_dataset(download_url, data_dir='./data', download_path='eye_roi.zip', image_size=128):
    _download_and_extract(download_url, download_path, data_dir)
    
    dataset = ROIDataset(
        images_dir=data_dir + '/eye_roi',
        img_size=image_size,
    )
    
    return dataset

class ROIDataset(Dataset):
    def __init__(self, images_dir, img_size=128, transform=None):
        
        self.images = glob(images_dir + '/*.jpg')
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image.unsqueeze(0)
    
if __name__ == "__main__":
    dataset = get_roi_dataset(
        download_url='https://drive.google.com/uc?id=1Xv7VNz8P6Znay08VaIWFWgmAoaeqs0oj',  # Replace with actual URL
        data_dir='./data/roi_images',
        download_path='eye_roi.zip',
        image_size=128
    )
    print(f"Dataset size: {len(dataset)}")
    sample_image = dataset[0]
    print(f"Sample image shape: {sample_image.shape}")