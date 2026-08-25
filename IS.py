import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def inception_score(dataloader, device, splits=10):
    model = inception_v3(weights='IMAGENET1K_V1', transform_input=False).to(device)
    model.eval()

    preds = []

    for batch in tqdm(dataloader, desc="预测中"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = torch.nn.functional.softmax(model(batch), dim=1)
        preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    scores = []

    split_size = preds.shape[0] // splits
    for k in range(splits):
        part = preds[k * split_size:(k + 1) * split_size, :]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean(entropy(part.T, py[:, np.newaxis]))))

    return np.mean(scores), np.std(scores)


if __name__ == '__main__':
    generated_images_path = r"C:\Users\wsh\Desktop\q-diffusion-master\twostage_w8a8_cali_iters_a_7000\samples\2026-04-10-11-39-31\images"

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolderDataset(generated_images_path, transform=transform)
    print(f"图片数: {len(dataset)}")
    print(f"路径: {generated_images_path}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("开始计算 Inception Score...")
    mean_is, std_is = inception_score(dataloader, device, splits=10)
    print(f"Inception Score (IS): 平均值={mean_is:.4f}, 标准差={std_is:.4f}")
