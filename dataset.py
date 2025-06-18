# dataset.py
from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor
import configs

IMG_EXTS = {".png", ".jpg", ".jpeg"}

def list_images(subdir):
    """列出子目錄中所有影像檔案路徑"""
    base = Path(configs.DATA_DIR, subdir)
    return [str(p) for p in base.rglob("*.*") if p.suffix.lower() in IMG_EXTS]

class ImgDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        feat = processor.image_processor if hasattr(processor, "image_processor") else processor.feature_extractor
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=feat.image_mean, std=feat.image_std),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

def make_loaders(proc):
    # 讀取影像路徑並切分
    real = list_images("Real_youtube"); fs = list_images("FaceSwap"); nt = list_images("NeuralTextures")
    random.seed(42)
    random.shuffle(real); random.shuffle(fs)
    n_real = len(real)
    train = real[:int(0.8*n_real)] + fs[:int(0.9*len(fs))]
    val   = real[int(0.8*n_real):int(0.9*n_real)] + fs[int(0.9*len(fs)):]
    test  = real[int(0.9*n_real):] + nt
    train_samples = [(p,0) for p in train] + [(p,1) for p in fs[:int(0.9*len(fs))]]
    val_samples   = [(p,0) for p in val]   + [(p,1) for p in fs[int(0.9*len(fs)):]]
    test_samples  = [(p,0) for p in test]  + [(p,1) for p in nt]

    train_loader = DataLoader(ImgDataset(train_samples, proc), batch_size=configs.BATCH_SIZE,
                              shuffle=True, num_workers=4)
    val_loader   = DataLoader(ImgDataset(val_samples, proc),   batch_size=configs.BATCH_SIZE,
                              shuffle=False, num_workers=4)
    test_loader  = DataLoader(ImgDataset(test_samples, proc),  batch_size=configs.BATCH_SIZE,
                              shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, test_samples
