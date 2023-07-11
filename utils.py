import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import transformers
from sklearn.model_selection import train_test_split
import torch

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextClassificationDataset(
        texts=df.text.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size)

def load_data(folder, path):
    return pd.read_csv(os.path.join(folder,path))

def process_data(data, tokenizer, test_size, max_len, batch_size):
    train_data, val_data = train_test_split(data, test_size=test_size)
    train_data_loader = create_data_loader(train_data, tokenizer, max_len, batch_size)
    val_data_loader = create_data_loader(train_data, tokenizer, max_len, batch_size)
    return train_data_loader, val_data_loader


def load_model(bert_type, learning_rate, device):
    if bert_type == 'bert-base':
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif bert_type == 'distilbert':
        tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = transformers.DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased').to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f'{bert_type} not supported')
    return tokenizer, model, optimizer
