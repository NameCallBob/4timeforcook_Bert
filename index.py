import torch
import torch.nn as nn
# 自己的model
from dataset import BertNERDataset
# tokenizer
from transformers import BertTokenizerFast

# 使用GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"目前使用{device}進行預測!")


model = torch.load("model.pth")

model.eval()

# 暫時用
from train import training_data_load
df_train, df_test = training_data_load()
texts = df_train[0][500:520]

from dataset import evaluate_one_text

for i in texts:
    sentence, label = evaluate_one_text(model, i)
    print(sentence) ; print(label)