import pandas as pd
from utils import *
import os
import yaml
import pprint
import copy
import time

CONFIG = 'bert_1.yml'
CRITERION = torch.nn.CrossEntropyLoss()
MODEL_METADATA_FILENAME = 'model_metadata.csv'

def train(train_data_loader, model, optimizer, epochs, device):
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
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

def pick_device(config):
    if config['device'] == 'gpu':
        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"
        print(f"using device type: {device_type}")
        device = torch.device(device_type)
    return device

def save_model(model, config):
    time_now_str = str(time.time()).replace('.','_')
    torch.save(model, os.path.join('models',f'{time_now_str}.pth'))

    model_metadata_path = os.path.join('models', MODEL_METADATA_FILENAME)
    file_exists = os.path.isfile(model_metadata_path)
    local_model_metadata = copy.deepcopy(config)
    local_model_metadata['time'] = time_now_str
    if file_exists:
        model_metadata_df = pd.read_csv(model_metadata_path)
    else:
        model_metadata_df = pd.DataFrame()
    model_metadata_df = pd.concat([model_metadata_df, pd.DataFrame(config, index=[0])], ignore_index=True)
    model_metadata_df.to_csv(model_metadata_path)

if __name__ == '__main__':
    with open(os.path.join('configs', CONFIG),'r') as file:
        config = yaml.safe_load(file)
    print('config:')
    pprint.pprint(config)
    device = pick_device(config)
    torch.cuda.empty_cache()
    # device = torch.device("cpu")
    data = load_data('data', config['data_file'])
    tokenizer, model, optimizer = load_model(config['bert_type'], config['learning_rate'], device)
    train_data_loader, test_data_loader = process_data(data, tokenizer, test_size=config['test_data_pct'], max_len=config['max_len'], batch_size=config['batch_size'])
    model = train(train_data_loader, model, optimizer, config['epochs'], device)
    save_model(model, config)

