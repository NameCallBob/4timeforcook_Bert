# module
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD
import os


# inside
from model import BertModel
from dataset import BertNERDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
# load testdata
labels_to_ids = {
        'O': 0,
        "num": 1,
        "B-ING":2,"B-DIS":3,"B-NUT":4,
        "B-ALG":5,"B-STP":6,"B-TME":7,
        "B-UDO":8,"B-TAG":9,"I-ING":10,"I-TAG":11
        }
LEARNING_RATE = 1e-2
EPOCHS = 100
BATCH_SIZE = 32


def training_data_load(train_size=0.9):
    import pandas as pd
    import numpy as np

    data = pd.read_excel("./Train_data/trans_result.xlsx")
    # print(data.columns)
    sentence = data['text'].tolist()
    labels = data['entity'].tolist()
    for i in range(len(sentence)):
        labels[i] = eval(labels[i])

    # 將索引打亂
    indices = np.arange(len(sentence))
    np.random.shuffle(indices)

    # 根據打亂的索引重新排列 sentence 和 labels
    sentence = [sentence[i] for i in indices]
    labels = [labels[i] for i in indices]

    train_num = round(len(sentence) * train_size)
    train = [sentence[:train_num], labels[:train_num]]
    test = [sentence[train_num:], labels[train_num:]]
    # print(train) ; print(test)
    return train, test


def save_pretrained(model, path):
    """將訓練好的模型進行保存"""
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


def evaluate(model, df_test):

    test_dataset = BertNERDataset(df_test)

    test_dataloader = DataLoader(
        test_dataset, num_workers=4, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("WARNING:Now is using CPU to train")
    device = torch.device("cuda" if use_cuda else "cpu")
    CUDA_LAUNCH_BLOCKING=1
    if use_cuda:
        model = model.cuda()

    total_acc_test = 0.0

    for test_data, test_label in test_dataloader:

        test_label = test_label.to(device)
        mask = test_data['attention_mask'].squeeze(1).to(device)

        input_id = test_data['input_ids'].squeeze(1).to(device)

        loss, logits = model(input_id, mask, test_label)

        for i in range(logits.shape[0]):

                logits_clean = logits[i][test_label[i] != -100]
                label_clean = test_label[i][test_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_test += acc

    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {total_acc_test / len(df_test): .3f}')


def train_loop(model, df_train, df_val):

    train_dataset = BertNERDataset(df_train)
    val_dataset = BertNERDataset(df_val)

    train_dataloader = DataLoader(
        train_dataset, num_workers=6, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, num_workers=6, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("Warning:Using CPU to train")
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    # 設置早停參數
    import numpy as np
    best_loss = np.inf  # 初始最佳损失为无穷大
    patience = 5        # 如果连续5个epoch验证集损失没有改善，则停止训练
    counter = 0           # 当前连续没有改善的epoch数

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0
        total_samples_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            total_loss_train += loss.item() * input_id.size(0)  # 每個 batch 的損失乘以 batch size
            total_samples_train += input_id.size(0)  # 更新總樣本數

            predictions = logits.argmax(dim=1)
            acc = ((predictions == train_label) & (train_label != -100)).float().sum()
            total_acc_train += acc.item()  # 累加正確預測數量

            loss.backward()
            optimizer.step()

        # 計算平均訓練損失和準確率
        avg_train_loss = total_loss_train / total_samples_train
        avg_train_acc = total_acc_train / total_samples_train

        model.eval()

        total_acc_val = 0
        total_loss_val = 0
        total_samples_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            with torch.no_grad():  # 在評估模式下不需要計算梯度
                loss, logits = model(input_id, mask, val_label)

            total_loss_val += loss.item() * input_id.size(0)  # 每個 batch 的損失乘以 batch size
            total_samples_val += input_id.size(0)  # 更新總樣本數

            predictions = logits.argmax(dim=1)
            acc = ((predictions == train_label) & (train_label != -100)).float().sum()
            total_acc_val += acc.item()  # 累加正確預測數量

        # 計算平均驗證損失和準確率
        avg_val_loss = total_loss_val / total_samples_val
        avg_val_acc = total_acc_val / total_samples_val

        print(f'Epochs: {epoch_num + 1} | Loss: {avg_train_loss:.3f} | Accuracy: {avg_train_acc:.3f} | Val_Loss: {avg_val_loss:.3f} | Total_Accuracy: {avg_val_acc:.3f}')

        # 檢查是否要停止訓練
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break


def main():
    from dataset import evaluate_one_text
    df_train, df_test = training_data_load()
    model = BertModel()
    train_loop(model, df_train, df_test)
    evaluate(model, df_test)
    save_pretrained(model, "pre_model")
    res_sentence = []
    res_label = []
    for i in df_test[0]:
        sentence, label = evaluate_one_text(model, i)
        res_sentence.append(sentence)
        res_label.append(label)
        print(sentence)
        print(label)
        print("-"*50)


if __name__ == "__main__":
    print("執行開始")
    main()
