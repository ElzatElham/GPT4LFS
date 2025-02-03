import torch
from torch import nn
from transformers import RobertaModel
from torchvision import models

class bigModel(nn.Module):
    def __init__(self, n_class=4, device='cpu'):
        super().__init__()
        self.img_block = models.convnext_tiny(pretrained=True)
        self.img_block.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.text_block = RobertaModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_class)
        )

    def forward(self, img, text):
        x1 = self.img_block(img)
        x2 = self.text_block(text).last_hidden_state[:, 0, :]
        return self.classifier(torch.cat([x1, x2], dim=1))
