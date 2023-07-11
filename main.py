from utils import *
import os
import random
TEST_DATA_PCT = 0.8
LEARNING_RATE = 1e-5
# BERT_TYPE = 'bert-base' # Supported: ['bert-base', 'distilbert']
BERT_TYPE = 'distilbert' # Supported: ['bert-base', 'distilbert']
EPOCHS = 1
BATCH_SIZE = 128
MAX_LEN = 150
CRITERION = torch.nn.CrossEntropyLoss()

def train(train_data_loader, model, optimizer, epochs, device):
    for _ in range(epochs):
        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = CRITERION(outputs.logits, labels)
            loss.backward()
            optimizer.step()
    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    # device = torch.device("cpu")
    data = load_data('data', 'shuffled_data.csv')
    tokenizer, model, optimizer = load_model(BERT_TYPE, LEARNING_RATE, device)
    train_data_loader, test_data_loader = process_data(data, tokenizer, test_size=TEST_DATA_PCT, max_len=MAX_LEN, batch_size=BATCH_SIZE)
    model = train(train_data_loader, model, optimizer, EPOCHS, device)







