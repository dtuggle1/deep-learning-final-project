from utils import *
import os
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DATA_PCT = 0.8
LEARNING_RATE = 1e-5
BERT_TYPE = 'bert-base' # Supported: ['bert-base', 'dilbert']
epochs = 1
BATCH_SIZE = 128
MAX_LEN = 150
CRITERION = torch.nn.CrossEntropyLoss()

def train(train_data_loader, model, optimizer, epochs):
    for _ in range(epochs):
        for batch in train_data_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = CRITERION(outputs.logits, labels)
            loss.backward()
            optimizer.step()
    return model

if __name__ == '__main__':
    data = load_data('data', 'shuffled_data.csv')
    tokenizer, model, optimizer = load_model(BERT_TYPE, LEARNING_RATE)
    train_data_loader, test_data_loader = process_data(data, tokenizer, test_size=TEST_DATA_PCT, max_len=MAX_LEN, batch_size=BATCH_SIZE)
    model = train(train_data_loader, model, optimizer, EPOCHS)







