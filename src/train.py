import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import CustomDataset
from model import bigModel
from utils import save_confusion_matrix, calculate_metrics, compute_cohens_kappa

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    for batch in dataloader:
        img, text, labels, _ = batch
        img, text, labels = img.to(device), text.to(device), labels.to(device)

        outputs = model(img, text)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            img, text, labels, _ = batch
            img, text, labels = img.to(device), text.to(device), labels.to(device)

            outputs = model(img, text)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

def main():
    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-4
    epochs = 50
    n_class = 4
    patience = 10

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPT4o annotated data preparation, refer to src/api.py for API calls
    train_pd = pd.read_csv('data/processed_data/train_data.csv')
    val_pd = pd.read_csv('data/processed_data/val_data.csv')

    # Datasets
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    train_set = CustomDataset(train_pd, tokenizer)
    val_set = CustomDataset(val_pd, tokenizer, is_train=False)

    # Data loaders
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_set, batch_size=8, shuffle=False)

    # Model initialization
    model = bigModel(n_class=n_class, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')
    counter = 0
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        t_loss, t_acc = train_one_epoch(train_dl, model, loss_fn, optimizer, device)
        v_loss, v_acc = evaluate(val_dl, model, loss_fn, device)

        train_loss_list.append(t_loss)
        train_acc_list.append(t_acc)
        val_loss_list.append(v_loss)
        val_acc_list.append(v_acc)

        print(f"Training Loss: {t_loss:.4f}, Training Accuracy: {t_acc:.4f}")
        print(f"Validation Loss: {v_loss:.4f}, Validation Accuracy: {v_acc:.4f}")

        # Early stopping mechanism
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), 'outputs/models/best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    # Visualize training process
    epochs_range = range(len(train_loss_list))
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_list, label='Training Accuracy')
    plt.plot(epochs_range, val_acc_list, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_list, label='Training Loss')
    plt.plot(epochs_range, val_loss_list, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig('outputs/figures/training_validation_results.png')
    plt.show()

    # 统计指标分析
    model.load_state_dict(torch.load('outputs/models/best_model.pth'))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_dl:
            img, text, labels, _ = batch
            img, text = img.to(device), text.to(device)
            outputs = model(img, text)
            preds = outputs.argmax(1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    # Classification report
    report = classification_report(y_true, y_pred, digits=3)
    print(report)
    with open('outputs/logs/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # 混淆矩阵
    cnf_matrix = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(y_true, y_pred, 'outputs/figures/confusion_matrix.png', normalized=False)
    save_confusion_matrix(y_true, y_pred, 'outputs/figures/normalized_confusion_matrix.png', normalized=True)

    # Calculate sensitivity and specificity
    accuracy, sensitivity, specificity = calculate_metrics(y_true, y_pred)
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    with open('outputs/logs/sensitivity_specificity.txt', 'w', encoding='utf-8') as f:
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")

    # 计算Cohen's Kappa
    kappa, ci, p_value = compute_cohens_kappa(cnf_matrix)
    print(f"Cohen's Kappa: {kappa}")
    print(f"95% 置信区间: {ci}")
    print(f"p-value: {p_value}")
    with open('outputs/logs/kappa_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"Cohen's Kappa: {kappa}\n")
        f.write(f"95% 置信区间: {ci}\n")
        f.write(f"p-value: {p_value}\n")

if __name__ == "__main__":
    main()
