import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, random_split


droplist = [
    "V8",
    "V13",
    "V15",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Time",
]


class CreditCardDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data = self.data.drop(droplist, axis=1)
        self.features = self.data.columns[:-1]
        self.labels = self.data["Class"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data.loc[idx, self.features].values, dtype=torch.float32)  # type: ignore
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


def splitter(datasets, partition: float = 0.8):
    train_size = int(partition * len(datasets))
    test_size = len(datasets) - train_size
    train_set, test_set = random_split(datasets, [train_size, test_size])
    return train_set, test_set


def decimation():
    data = pd.read_csv("creditcard.csv")

    number_records_fraud = len(data[data.Class == 1])# 计算异常样本的个数
    fraud_indices = np.array(data[data.Class == 1].index) # 异常样本在原数据的索引值

    normal_indices = data[data.Class == 0].index # 获得原数据正常样本的索引值
    
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False) # 通过索引进行随机的选择
    random_normal_indices = np.array(random_normal_indices)

    under_sample_indices = np.concatenate([fraud_indices,random_normal_indices]) # 将class=1和class=0 的选出来的索引值进行合并

    under_sample_data = data.iloc[under_sample_indices,:]

    under_sample_data.to_csv("creditcard-decimation.csv", index=False)


if __name__ == "__main__":
    from args import parser

    args = parser.parse_args()
    data = pd.read_csv("creditcard.csv")
    test = CreditCardDataset(data)
    print(len(test))
