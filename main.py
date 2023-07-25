import pandas as pd
from utils import *
import os
import yaml
import pprint
import copy
import time
import datetime as dt
from sklearn.metrics import classification_report

CSV_CONFIG = True
YAML_CONFIG = False
CONFIG = 'bert_1.yml'
MODEL_METADATA_FILENAME = 'model_metadata.csv'
MACHINE_USED = 'dereks_desktop'
EVALUATE_TRAIN_DATA = True
EVALUATE_TEST_DATA = True
EVALUATE_INITIAL_DATA = True
EVALUATE_EACH_EPOCH = True
EVALUATE_FINAL_DATA = False

CRITERION = torch.nn.CrossEntropyLoss()

def update_output_table(metric_report, table, stage, epoch=True):
    for metric_key in metric_report.keys():
        if epoch:
            table_key = f'Epoch {stage}'
        else:
            table_key = stage
        table[f'{table_key} {metric_key}'] = metric_report[metric_key]
    return table

def train(train_data_loader, test_data_loader, model, optimizer, epochs, device,
          output_table):
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
        if EVALUATE_EACH_EPOCH:
            if EVALUATE_TRAIN_DATA:
                evaluate_report = evaluate(model, train_data_loader, device)
                output_table = update_output_table(copy.deepcopy(evaluate_report),
                                            output_table, epoch, epoch=True)
                print("Training Set Accuracy: ", evaluate_report['accuracy'])
            if EVALUATE_TEST_DATA:
                evaluate_report = evaluate(model, test_data_loader, device)
                output_table = update_output_table(copy.deepcopy(evaluate_report), output_table,
                                    epoch, epoch=True)
                print("Testing Set Accuracy: ", evaluate_report['accuracy'])
    end_train_time = dt.datetime.now()
    print(f'Training completed at: {end_train_time}')
    training_duration = end_train_time - start_train_time
    training_duration = training_duration.total_seconds()
    training_duration_minutes = int(training_duration/60)
    training_hours = training_duration_minutes // 60
    training_minutes = training_duration_minutes %60
    print(f"Training duration: {training_hours}:{training_minutes}")
    return model, training_duration_minutes, output_table

def evaluate(model, data_loader, device):
    model = model.eval()

    correct_predictions = 0
    total_predictions = 0
    total_log_prob = 0.0

    all_preds = []
    all_labels = []
    dloader_index = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.shape[0]

            # Compute the total_log_prob
            for i, label in enumerate(labels):
                total_log_prob += outputs.logits[i, label].item()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            dloader_index +=1
            if dloader_index >=5:
                break

    accuracy = correct_predictions.double() / total_predictions
    total_log_prob_tensor = torch.tensor(-total_log_prob / total_predictions,
                                         device=device)
    perplexity = torch.exp(total_log_prob_tensor).item()

    print(f"Accuracy: {accuracy}")
    print(f"Perplexity: {perplexity}")

    # If you want to compute precision, recall, f1-score, you'll need to accumulate all_preds and all_labels outside the loop.
    # report = classification_report(all_labels, all_preds, zero_division=0)
    # print(report)

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    report = report['1']
    report['accuracy'] = accuracy.item()
    report['perplexity'] = perplexity

    return report

def pick_device(config):
    if config['device'] == 'gpu':
        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"
        print(f"using device type: {device_type}")
        device = torch.device(device_type)
    return device

def save_model(model, config, training_duration, output_table):
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
    local_model_metadata.update(output_table)

    if file_exists:
        model_metadata_df = pd.read_csv(model_metadata_path)
    else:
        model_metadata_df = pd.DataFrame()
    model_metadata_df = pd.concat([model_metadata_df, pd.DataFrame(local_model_metadata,index=[0])], ignore_index=True)
    drop_cols = []
    for key in model_metadata_df.keys():
        if 'Unnamed' in key:
            drop_cols.append(key)
    model_metadata_df = model_metadata_df.drop(labels=drop_cols,axis=1)
    model_metadata_df.to_csv(model_metadata_path)

def main(config):
    print('config:')
    pprint.pprint(config)
    device = pick_device(config)
    torch.cuda.empty_cache()
    data = load_data('data', config['data_file'])
    tokenizer, model, optimizer = load_model(config['bert_type'], float(config['learning_rate']), device)
    train_data_loader, test_data_loader = process_data(data, tokenizer, test_size=config['test_data_pct'],
                                                       max_len=config['max_len'], batch_size=config['batch_size'])
    output_table = {}
    if EVALUATE_INITIAL_DATA:
        if EVALUATE_TRAIN_DATA:
            evaluate_report = evaluate(model, train_data_loader, device)
            output_table = update_output_table(copy.deepcopy(evaluate_report), output_table,
                                'initial train', epoch=False)
            print("Initial Training Set Accuracy: ", evaluate_report['accuracy'])
        if EVALUATE_TEST_DATA:
            evaluate_report = evaluate(model, test_data_loader, device)
            output_table = update_output_table(copy.deepcopy(evaluate_report), output_table,
                                'initial test', epoch=False)
            print("Initial Testing Set Accuracy: ", evaluate_report['accuracy'])
    model, training_duration, accuracies = train(train_data_loader, test_data_loader, model, optimizer, config['epochs'], device,
                                     output_table)
    if EVALUATE_FINAL_DATA:
        if EVALUATE_TRAIN_DATA:
            evaluate_report = evaluate(model, train_data_loader, device)
            output_table = update_output_table(copy.deepcopy(evaluate_report), output_table,
                                'final train', epoch=False)
            print("Final Training Set Accuracy: ", evaluate_report['accuracy'])
        if EVALUATE_TEST_DATA:
            evaluate_report = evaluate(model, test_data_loader, device)
            output_table = update_output_table(copy.deepcopy(evaluate_report), output_table,
                                'final test', epoch=False)
            print("Final Testing Set Accuracy: ", evaluate_report['accuracy'])

    save_model(model, config, training_duration, output_table)

def evaluate_model_test():
    with open(os.path.join('configs', CONFIG), 'r') as file:
        config = yaml.safe_load(file)
    device = pick_device(config)
    torch.cuda.empty_cache()
    data = load_data('data', config['data_file'])
    tokenizer, model, optimizer = load_model(config['bert_type'], float(config['learning_rate']), device)
    train_data_loader, test_data_loader = process_data(data, tokenizer, test_size=0.05,
                                                       max_len=config['max_len'], batch_size=config['batch_size'])
    evaluate(model, test_data_loader, device)

if __name__ == '__main__':
    evaluate_model_test()
    # if CSV_CONFIG:
    #     configs = pd.read_csv(os.path.join('configs', CONFIG))
    #     config_idx = 0
    #     for _, row in configs.iterrows():
    #         print('config index: ', config_idx)
    #         try:
    #             config = copy.deepcopy(row.to_dict())
    #             main(config)
    #         except:
    #             print('config did not work')
    #         config_idx +=1
    # elif YAML_CONFIG:
    #     with open(os.path.join('configs', CONFIG), 'r') as file:
    #         config = yaml.safe_load(file)
    #     main(config)