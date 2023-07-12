import pandas as pd
from utils import *
import os
import yaml
import pprint
import copy
import time
import datetime as dt

CONFIG = 'bert_1.yml'
MODEL_METADATA_FILENAME = 'model_metadata.csv'
MACHINE_USED = 'dereks_desktop'

CRITERION = torch.nn.CrossEntropyLoss()

def train(train_data_loader, test_data_loader, model, optimizer, epochs, device, accuracies):
    start_train_time = dt.datetime.now()
    print(f'Starting training at {start_train_time}')
    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}/{epochs}')
        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = CRITERION(outputs.logits, labels)
            loss.backward()
            optimizer.step()
        # accuracies[f'Epoch {epoch+1} Train'] = evaluate(model, train_data_loader, device)
        # accuracies[f'Epoch {epoch+1} Test'] = evaluate(model, test_data_loader, device)
        # print("Training Set Accuracy: ", accuracies[f'Epoch {epoch+1} Train'])
        # print("Testing Set Accuracy: ", accuracies[f'Epoch {epoch+1} Test'])
    end_train_time = dt.datetime.now()
    print(f'Training completed at: {end_train_time}')
    training_duration = end_train_time - start_train_time
    training_duration = training_duration.total_seconds()
    training_duration_minutes = int(training_duration/60)
    training_hours = training_duration_minutes // 60
    training_minutes = training_duration_minutes %60
    print(f"Training duration: {training_hours}:{training_minutes}")
    return model, training_duration_minutes

def evaluate(model, data_loader, device):
    model_eval = model.eval()
    return 1
    correct = 0
    total = 0


    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model_eval(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            correct += torch.sum(preds == labels)
            total += labels.shape[0]
    accuracy = correct/total
    return accuracy

def pick_device(config):
    if config['device'] == 'gpu':
        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"
        print(f"using device type: {device_type}")
        device = torch.device(device_type)
    return device

def save_model(model, config, training_duration, accuracies):
    time_now_str = str(time.time()).replace('.','_')
    torch.save(model, os.path.join('models',f'{time_now_str}.pth'))

    model_metadata_path = os.path.join('models', MODEL_METADATA_FILENAME)
    file_exists = os.path.isfile(model_metadata_path)
    local_model_metadata = copy.deepcopy(config)

    local_model_metadata['timestamp'] = time_now_str
    training_hours = training_duration // 60
    training_minutes = training_duration %60
    local_model_metadata['training duration'] = f'{training_hours}:{training_minutes}'
    local_model_metadata['device'] = MACHINE_USED
    local_model_metadata.update(accuracies)

    if file_exists:
        model_metadata_df = pd.read_csv(model_metadata_path)
    else:
        model_metadata_df = pd.DataFrame()
    model_metadata_df = pd.concat([model_metadata_df, pd.DataFrame(local_model_metadata, index=[0])], ignore_index=True)
    model_metadata_df.to_csv(model_metadata_path)

if __name__ == '__main__':
    with open(os.path.join('configs', CONFIG),'r') as file:
        config = yaml.safe_load(file)
    print('config:')
    pprint.pprint(config)
    device = pick_device(config)
    torch.cuda.empty_cache()
    data = load_data('data', config['data_file'])
    tokenizer, model, optimizer = load_model(config['bert_type'], float(config['learning_rate']), device)
    train_data_loader, test_data_loader = process_data(data, tokenizer, test_size=config['test_data_pct'], max_len=config['max_len'], batch_size=config['batch_size'])
    accuracies = {}
    accuracies['initial train'] = evaluate(model, train_data_loader, device)
    accuracies['initial test'] = evaluate(model, test_data_loader, device)
    print("Initial Training Set Accuracy: ", accuracies['initial train'])
    print("Initial Testing Set Accuracy: ", accuracies['initial test'])
    model, training_duration = train(train_data_loader, test_data_loader, model, optimizer, config['epochs'], device, accuracies)
    accuracies['final train'] = evaluate(model, train_data_loader, device)
    accuracies['final test'] = evaluate(model, test_data_loader, device)
    print("Final Training Set Accuracy: ", accuracies['final train'])
    print("Final Testing Set Accuracy: ", accuracies['final train'])

    save_model(model, config, training_duration, accuracies)