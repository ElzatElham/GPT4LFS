# GPT4LFS: generative pre-trained transformer 4 omni for lumbar foramina stenosis

Welcome to the official PyTorch implementation of our groundbreaking research paper: 

**GPT4LFS (generative pre-trained transformer 4 omni for lumbar foramina stenosis): enhancing lumbar foraminal stenosis image classification through large multimodal models**

## Directory Structure

```plaintext
your-project/
├── data/
│   ├── raw/
│   └── processed/
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

## Installation

Clone the repository and install the required packages with:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
- Place raw data in the `data/raw/` directory.
- Run the data preparation script:
  ```bash
  python src/data_preparation.py
  ```
- The processed data will be saved in the `data/processed/` directory.
- If you need to call GPT4o to generate labeled data, please refer to `src/api.py`.

### 2. Training the Model
- Run the training script:
  ```bash
  python src/train.py
  ```
- Training results and logs will be saved in the `outputs/` directory.

## References

[PyTorch](https://pytorch.org/)
