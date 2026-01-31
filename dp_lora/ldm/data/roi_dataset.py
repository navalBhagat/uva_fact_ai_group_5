from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms.functional import resize, pad
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

from torchvision.transforms.functional import pad, resize

class PadToSize:
    def __init__(
        self,
        H=218,
        W=178,
        resize_h=64,
        resize_w=64,
        fill=0
    ):
        self.H = H
        self.W = W
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.fill = fill

    def __call__(self, img):
        # PIL size: (W, H)
        w, h = img.size

        pad_top = (self.H - h) // 2
        pad_bottom = self.H - h - pad_top
        pad_left = (self.W - w) // 2
        pad_right = self.W - w - pad_left

        img = pad(
            img,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill
        )

        img = resize(
            img,
            (self.resize_h, self.resize_w),
            antialias=True
        )
        return img

        
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
                PadToSize(resize_h=img_size, resize_w=img_size),
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