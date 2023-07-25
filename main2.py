def main(config):
    print('config:')
    pprint.pprint(config)
    device = pick_device(config)
    torch.cuda.empty_cache()
    data = load_data('data', config['data_file'])
    tokenizer, model, optimizer = load_model(config['bert_type'], float(config['learning_rate']), device)
    train_data_loader, test_data_loader = process_data(data, tokenizer, test_size=config['test_data_pct'],
                                                       max_len=config['max_len'], batch_size=config['batch_size'])
    accuracies = {}
    if EVALUATE_INITIAL_DATA:
        if EVALUATE_TRAIN_DATA:
            accuracies['initial train'] = evaluate(model, train_data_loader, device)
            print("Initial Training Set Accuracy: ", accuracies['initial train'])
        if EVALUATE_TEST_DATA:
            accuracies['initial test'] = evaluate(model, test_data_loader, device)
            print("Initial Testing Set Accuracy: ", accuracies['initial test'])
    model, training_duration = train(train_data_loader, test_data_loader, model, optimizer, config['epochs'], device,
                                     accuracies)
    if EVALUATE_FINAL_DATA:
        if EVALUATE_TRAIN_DATA:
            accuracies['final train'] = evaluate(model, train_data_loader, device)
            print("Final Training Set Accuracy: ", accuracies['final train'])
        if EVALUATE_TEST_DATA:
            accuracies['final test'] = evaluate(model, test_data_loader, device)
            print("Final Testing Set Accuracy: ", accuracies['final test'])

    save_model(model, config, training_duration, accuracies)

if __name__ == '__main__':
    if CSV_CONFIG:
        configs = pd.read_csv(os.path.join('configs', CONFIG))
        for config in configs:
            main(config)
    elif YAML_CONFIG:
        with open(os.path.join('configs', CONFIG), 'r') as file:
            config = yaml.safe_load(file)
        main(config)