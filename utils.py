import pickle
from pathlib import Path
import math


def save_object(object_to_save, path, name, filetype='pkl'):
    Path(path).mkdir(parents=True, exist_ok=True)  # Check if map exists, if not, create it

    with open(path + name + '.' + filetype, 'wb') as file:
        pickle.dump(object_to_save, file, pickle.HIGHEST_PROTOCOL)
        file.close()

    return


def save_history(object_to_save, path, name):
    if hasattr(object_to_save, '__len__'):
        histories = []
        for history_i in object_to_save:
            model_history = convert_history_to_object(history_i)
            histories.append(model_history)
    else:
        histories = HistoryTrainedModel(object_to_save.history, object_to_save.epoch, object_to_save.params)

    save_object(object_to_save=histories, path=path, name=name)


def convert_history_to_object(history):
    return HistoryTrainedModel(history.history, history.epoch, history.params)


def save_models(models, path):
    for model in models:
        model.save(path)


def load_object(path, name, filetype='pkl', history=False):
    file = open(path + name + '.' + filetype, 'rb')  # Load a file with read properties
    loaded = pickle.load(file)
    file.close()

    return loaded


def receptive_field(layers, kernel_size, print_in_console=False):
    """Calculates the receptive field of a TCN. Optional bool to print outcome
    Formula used: 2^(Layers-1) * kernel size
    """
    total = 0
    for i in range(0, layers):
        total += math.pow(2, i)

    outcome = int(total * (kernel_size - 1) * 2 + 1)
    if print_in_console:
        print(f'Receptive field size = {outcome}')
    return outcome


class HistoryTrainedModel(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params
