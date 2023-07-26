import matplotlib.pyplot as plt
import os
import pandas as pd

MODEL_METADATA_FILENAME = 'model_metadata.csv'

PLOTS = {
    'Learning Rate':
        {
            'idea_key': 'vary learning rates',
            'x_data': 'learning_rate',
            'y_data': 'final test',
            'xlabel': 'Learning Rate',
            'ylabel': 'Test Set Accuracy (%)',
            'title': 'Learning Rate vs. Test Set Accuracy',
            'exclude_param': 'learning_rate'
        },
    'Batch Size':
        {
            'idea_key': 'change batch size',
            'x_data': 'batch_size',
            'y_data': 'final test',
            'xlabel': 'Batch Size',
            'ylabel': 'Test Set Accuracy (%)',
            'title': 'Batfh Size vs. Test Set Accuracy',
            'exclude_param': 'batch_size'
        },
    'Max Len':
        {
            'idea_key': 'change max len',
            'x_data': 'max_len',
            'y_data': 'final test',
            'xlabel': 'Max Length (tokens)',
            'ylabel': 'Test Set Accuracy (%)',
            'title': 'Max Length vs. Test Set Accuracy',
            'exclude_param': 'max_len'
        },
}

def strip_data(data,idea_key):
    return data.loc[data['idea'] == idea_key]

def standard_plot(data, plot_dict):
    local_data = strip_data(data, plot_dict['idea key'])
    plt.figure()
    plt.scatter(local_data[plot_dict['x_data']], local_data[plot_dict['y_data']])
    plt.xlabel(plot_dict['xlabel'])
    plt.ylabel(plot_dict['ylabel'])
    plt.title(plot_dict['title'])
    plt.savefig(f'{plot_dict["title"]}.png')

def epoch_plot(data):
    local_data = strip_data(data, 'change epochs')
    row_indeces = [5,10]
    plt.figure()
    col_key_2 = 'accuracy'
    for row_index in row_indeces:
        row_data = local_data[row_index]
        for col_key_1 in ['test', 'train']:
            epochs, y_data, plot_keys = get_data_from_epochs(col_key_1, col_key_2, row_data)
            plt.plot(epochs, y_data, label = f'Epochs: {epochs[-1]}')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Epochs')
    plt.savefig('Test Accuracy vs Epochs.png')

def get_data_from_epochs(col_key_1, col_key_2, df):
    plot_keys = []
    epochs = []
    y_data = []
    for key in df.keys():
        if 'epoch' in key.lower() and col_key_1 in key.lower() and col_key_2 in \
                key.lower():
            plot_keys.append(key)
            epochs.append(int(key[6]))
            y_data.append(df[key])
    return epochs, y_data, plot_keys

def plot_metrics_best_model(data, best_model_index=5):
    data_local = data[best_model_index]
    metrics = ['perplexity', 'recall', 'f1-score', 'support', 'precision']
    data_types = ['train', 'test']
    for metric in metrics:
        plt.figure()
        for data_type in data_types:
            epochs, y_data, plot_keys = get_data_from_epochs(metric, data_type, data_local)
            plt.plot(epochs, y_data, label=data_type)
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(f'{metric} Change over Epochs for Best Model')
            plt.savefig(f'{metric} Change over Epochs for Best Model.png')
if __name__ == '__main__':
    data = pd.read_csv(os.path.join('data', MODEL_METADATA_FILENAME))
    for key in PLOTS:
        plot_dict = PLOTS[key]
        standard_plot(plot_dict)

# Final accuracy vs different learning rates
# adjusting batch size. Batch size vs final accuracy
# changing max len. Max len vs final accuracy
# Changing number of epochs. Accuracy over each epoch for training and test data. Show overfitting trend
# Show curves of all metrics for best model

