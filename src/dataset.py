from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import torch

class CustomDataset(Dataset):
    def __init__(self, pd_data, tokenizer, max_len=40, is_train=True):
        self.pd_data = pd_data
        self.text_list = list(pd_data['text'])
        self.label_list = list(pd_data['label'])
        self.img_path_list = list(pd_data['img_path'])
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]) if is_train else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pd_data)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        text_tokens = self.tokenizer(
            text, padding='max_length', 
            max_length=self.max_len, 
            truncation=True, 
            return_tensors="pt"
        )['input_ids'][0]

        img = Image.open(self.img_path_list[idx]).convert('RGB')
        return (
            self.transform(img),
            text_tokens,
            torch.tensor(self.label_list[idx]).long(),
            self.img_path_list[idx]
        )
