import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import copy
import numpy as np

MODEL_METADATA_FILE = os.path.join('models','model_metadata - model_metadata.csv')
OUTPUT_FOLDER = 'outputs'
BEST_MODEL_INDEX = 9

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18

PLOTS = {
    'Learning Rate':
        {
            'idea_key': 'vary learning rates',
            'x_data': 'learning_rate',
            'y_data': 'Epoch 8 test  accuracy',
            'xlabel': 'Learning Rate',
            'ylabel': 'Test Set Accuracy (%)',
            'title': 'Learning Rate vs. Test Set Accuracy',
            'exclude_param': 'learning_rate'
        },
    'Batch Size':
        {
            'idea_key': 'change batch size',
            'x_data': 'batch_size',
            'y_data': 'Epoch 8 test  accuracy',
            'xlabel': 'Batch Size',
            'ylabel': 'Test Set Accuracy (%)',
            'title': 'Batch Size vs. Test Set Accuracy',
            'exclude_param': 'batch_size'
        },
    'Max Len':
        {
            'idea_key': 'change max len',
            'x_data': 'max_len',
            'y_data': 'Epoch 8 test  accuracy',
            'xlabel': 'Max Length (tokens)',
            'ylabel': 'Test Set Accuracy (%)',
            'title': 'Max Length vs. Test Set Accuracy',
            'exclude_param': 'max_len'
        },
}


def strip_data(data,idea_key):
    return data.loc[data['idea'].str.contains(idea_key, na = False)]

def standard_plot(data, plot_dict):
    local_data = strip_data(data, plot_dict['idea_key'])
    plt.figure()
    plt.scatter(local_data[plot_dict['x_data']], local_data[plot_dict['y_data']])
    plt.xlabel(plot_dict['xlabel'])
    plt.ylabel(plot_dict['ylabel'])
    if plot_dict['xlabel'] == 'Learning Rate':
        plt.xscale('log')
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{plot_dict["title"]}.png'),bbox_inches='tight')

def epoch_plot(data):
    local_data = strip_data(data, 'change epochs')
    plt.figure()
    col_key_2 = 'accuracy'
    for row_index in range(len(local_data)):
        row_data = local_data.iloc[row_index]
        # row_data = local_data.loc[row_index]
        for col_key_1 in ['test', 'train']:
            epochs, y_data, plot_keys = get_data_from_epochs(col_key_1, col_key_2, row_data)
            plt.plot(epochs, y_data, label = f'{col_key_1.capitalize()}, Epochs: '
                                             f'{copy.deepcopy(epochs[-1])}')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylabel('Accuracy (%)')
    plt.title('Test and Train Accuracies over Training')
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'Test and Train Accuracies over '
                                            'Training.png'),bbox_inches='tight')

def get_data_from_epochs(col_key_1, col_key_2, df):
    plot_keys = []
    epochs = []
    y_data = []
    for key in df.keys():
        if 'epoch' in key.lower() and col_key_1 in key.lower() and col_key_2 in \
                key.lower():
            if str(df[key]) == 'nan':
                continue
            plot_keys.append(key)
            epoch = re.search(r'\d+', key)
            epoch = int(epoch.group())+1
            epochs.append(epoch)
            y_data.append(df[key])
    return epochs, y_data, plot_keys

def plot_metrics_best_model(data):
    data_local = data.loc[BEST_MODEL_INDEX]
    metrics = ['perplexity', 'recall', 'f1-score', 'support', 'precision', 'accuracy']
    data_types = ['train', 'test']
    for metric in metrics:
        plt.figure()
        for data_type in data_types:
            epochs, y_data, plot_keys = get_data_from_epochs(metric, data_type, data_local)
            plt.plot(epochs, y_data, label=data_type.capitalize())
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            title = f'Best Model {metric.capitalize()} Over Training'
            plt.title(title)
            plt.savefig(os.path.join(OUTPUT_FOLDER, f'{title}.png'),
                        bbox_inches='tight')

if __name__ == '__main__':
    data = pd.read_csv(MODEL_METADATA_FILE)
    for key in PLOTS:
        plot_dict = PLOTS[key]
        standard_plot(data, plot_dict)
    plot_metrics_best_model(data)
    epoch_plot(data)
# Final accuracy vs different learning rates
# adjusting batch size. Batch size vs final accuracy
# changing max len. Max len vs final accuracy
# Changing number of epochs. Accuracy over each epoch for training and test data. Show overfitting trend
# Show curves of all metrics for best model

