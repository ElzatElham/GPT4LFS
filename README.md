# GPT4LFS

> 多模态深度学习分类项目

## 目录结构

your-project/
├── data/
│   ├── raw/
│   ├── processed/
├── src/
│   ├── data_preparation.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── notebooks/
│   └── exploration.ipynb
├── outputs/
│   ├── models/
│   ├── figures/
│   └── logs/
├── README.md
├── requirements.txt
└── .gitignore
```

## 安装

```

pip install -r requirements.txt
````

## 使用方法

1. **数据准备**
   - 将原始数据放置在 `data/raw/` 目录下。
   - 运行数据准备脚本：
     ```bash
     python src/data_preparation.py
     ```
   - 处理后的数据将保存在 `data/processed_data/` 目录下。
   - 调用GPT4o生成标注数据请参考src/api.py

2. **训练模型**
   - 运行训练脚本：
     ```bash
     python src/train.py
     ```
   - 训练结果和日志将保存在 `outputs/` 目录下。

## 参考文献

- [PyTorch](https://pytorch.org/)
