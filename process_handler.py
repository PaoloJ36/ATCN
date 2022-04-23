import model
import matplotlib.pyplot as plt
import math
import utils
import numpy as np
import config as config_class
import copy
import time


def build_models(models_to_build, configs, use_single_config=False):
    """
    Builds an array of models given an array of configs with similar size
    Opt: use_single_config - Uses a single config file for all models instead
    """
    # If not using a single config file, check if amount of models and configs match
    if not use_single_config:
        assert len(models_to_build) == len(configs), "Length of models and configs do not match"
    else:
        config = configs

    models = []

    for i, model_i in enumerate(models_to_build):
        if not use_single_config:
            config = configs[i]

        model_built = build_model(model_to_build=model_i, config=config)
        models.append(model_built)

    return models


def build_model(model_to_build, config):
    return model.Model(model_name=model_to_build, config=config)


def train_models(models, train, val, verbose=2):
    """
    Train an array of models given an array of configs with similar size
    Opt: use_single_config - Uses a single config file for all models instead
    """

    histories = []

    for i, model_i in enumerate(models):

        history = train_model(model_to_train=model_i, train=train, val=val, verbose=verbose)
        histories.append(history)

    return histories


def train_model(model_to_train, train, val, verbose=2):
    start = time.time()
    history = model_to_train.model.fit(train, validation_data=val,
                                       batch_size=model_to_train.cfg.batch_size, epochs=model_to_train.cfg.epochs,
                                       callbacks=[model_to_train.get_callback()], verbose=verbose)
    end = time.time()
    print(f"Time it took to train {model_to_train.model.name}: {end - start}")
    return history


def plot_results(histories, model_names, combine=False, share_y=False, ylim=None):
    """Function to plot the results given an array of histories and an array of model names
    Optional: share_y = Boolean, share the same y-axis (mse) by all graphs if not combined
    Optional: ylim = [min, max] array with min and maximum of the y axis.
    """

    if combine:
        labels = []

        for count, history in enumerate(histories):
            plt.plot(history.history['mse'])
            plt.plot(history.history['val_mse'])

            labels.append('Train ' + model_names[count])
            labels.append('Val ' + model_names[count])

        plt.ylabel('Mean Square Error')
        plt.xlabel('Epoch')

        plt.legend(labels, loc='upper left')

    if not combine:
        amount_of_plots = len(histories)

        figure, axis = plt.subplots(2, math.ceil(amount_of_plots / 2), sharey=share_y)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error')

        if ylim is not None:
            plt.ylim(ylim)

        for count, history in enumerate(histories):
            axis[count % 2, math.floor(count / 2)].plot(history.history['mse'])
            axis[count % 2, math.floor(count / 2)].plot(history.history['val_mse'])

            # axis[count % 2, math.floor(count / 2)].xlabel('Epoch')
            # axis[count % 2, math.floor(count / 2)].ylabel('Mean Square Error')

            axis[count % 2, math.floor(count / 2)].legend(['Train', 'Val'])
            axis[count % 2, math.floor(count / 2)].set_title(model_names[count])

    plt.show()
    return


def grid_search(model_names, config, save_path, train, val, stock=None):
    configs = create_configs_from_dict(config.parameter_grid)
    results = {}

    # For all models
    for i, model_to_train in enumerate(model_names):
        """
        Creates a dictionary of dictionaries with histories, scores, parameters and model path inside
        Example to call: results['TCN']['Histories']
        """

        results[model_names[i]] = {}

        histories = []
        best_scores = []
        parameters = []
        model_paths = []

        # For all parameters
        for cfg in configs:

            """If receptive field is unwanted, skip this config"""
            if skip_config(cfg):
                continue

            print(f'Creating {model_to_train} with: Learning rate = {cfg.learning_rate}, filters = {cfg.filters}, '
                  f'layers = {cfg.layers}, '
                  f'Kernel size = {cfg.kernel_size}, epochs = {cfg.epochs}')

            """Build the model"""
            built_model = build_model(model_to_train, cfg)

            """
            ***** Train the model, append the history *****
            """
            # Train the model
            history = train_model(built_model, train=train, val=val, verbose=0)

            """
            ***** Save the results, parameters and model in a unique folder *****
            """
            # Create unique filename
            filename = create_filename(cfg)

            # Save the parameters
            parameters.append(cfg)

            # Convert history to object and save it
            history = utils.convert_history_to_object(history)
            histories.append(history)
            utils.save_history(history, path=save_path + 'Histories\\' + model_names[i] + '\\', name=filename)

            # Store the best score (val mse)
            best_score = np.array(history.history['val_mse']).min()
            best_scores.append(best_score)
            print(f'Training completed. {model_names[i]} achieved a val_mse of: {best_score}')

            # Save the model
            model_path = save_path + 'Models\\' + str(model_names[i]) + '\\' + str(filename)
            model_paths.append(model_path)
            built_model.save(path=model_path)

            """
            ***** Update the dictionary and save it ***** 

            """
            # Save results in dict at the end of every attempt
            results[model_names[i]]['History'] = histories
            results[model_names[i]]['Best scores'] = best_scores
            results[model_names[i]]['Parameters'] = parameters
            results[model_names[i]]['Model path'] = model_paths

            # Save results dict
            utils.save_object(results, path=save_path, name='Results_dict')

    return results


def skip_config(config):
    """Temporary function to block out models with unwanted receptive fields"""
    if config.layers == 6 and config.kernel_size != 3:
        return True
    if config.layers == 3 and config.kernel_size != 19:
        return True
    if config.layers == 2 and config.kernel_size != 43:
        return True
    return False


def create_configs_from_dict(parameter_grid, depth=0, config=config_class.Config(), configs=[]):
    """
    Method that generates an array of configs given a dictionary of all possible parameter values as input.

    :param parameter_grid: A dictionary containing all the parameters and their possible values
    :param depth: Do not fill in, used for recursive
    :param config: Do not fill in, used for recursive
    :param configs: Do not fill in, used for recursive
    :return: An array of config objects with all the possible settings in a parameter grid
    """

    """To do: Implement an easy way to keep track of what the parameters implemented are"""

    # Get the parameter name so we can figure out which one to edit
    parameter_name = list(parameter_grid.items())[depth][0]

    for parameter_setting in list(parameter_grid.values())[depth]:
        # Set parameter in config
        config.__setattr__(parameter_name, parameter_setting)

        # If it is the last parameter, create copies of all parameter settings and add them to the list
        if depth is len(list(parameter_grid.values())) - 1:
            config2 = copy.copy(config)
            configs.append(config2)

            # If it is the last setting for the final parameter, return the list of configs
            if parameter_setting == list(parameter_grid.values())[0][len(list(parameter_grid.values())[0]) - 1]:
                return configs

            # If depth is final, continue with the loop as to not create another depth layer
            continue

        # If depth is not final, increase depth
        configs = create_configs_from_dict(parameter_grid=parameter_grid, depth=depth + 1, config=config,
                                           configs=configs)

    # Returns when done with the current parameter
    return configs


def create_filename(config):
    """Temporary function that returns the parameters used in this experiment as a filename"""
    return 'lr' + str(config.learning_rate) + '_' + 'filt' + str(config.filters) + '_' + \
           'la' + str(config.layers) + '_' + 'ks' + \
           str(config.kernel_size) + '_epoch' + str(config.epochs)


def print_grid_results(results):
    best_score_list = []
    best_params_list = []
    best_history_list = []
    model_list = []

    # For each model, keep track of best results
    for model_ in results.keys():
        best_score = 999.9
        best_params = None
        best_history = None

        # For each result, print it
        for result in range(0, len(results[model_]['Best scores'])):
            score = list(results[model_]['Best scores'])[result]
            parameters = get_parameters_from_config(list(results[model_]['Parameters'])[result])

            print(f'{model_} with params {parameters} scored: {score}')
            if score < best_score:
                best_score = score
                best_params = parameters
                best_history = list(results[model_]['History'])[result]

        # Save the best results
        best_score_list.append(best_score)
        best_params_list.append(best_params)
        model_list.append(model_)
        best_history_list.append(best_history)

    # Print best results for each model
    for i, model_ in enumerate(results.keys()):
        print(f'Best score for {model_list[i]} is: {best_score_list[i]} with {best_params_list[i]}')

    return model_list, best_history_list


def get_parameters_from_config(config):
    return 'Learning rate: ' + str(config.learning_rate) + ' Filters: ' + str(config.filters) + \
           ' Layers: ' + str(config.layers) + ' Kernel size: ' + str(config.kernel_size) + ' Epochs: ' + str(config.epochs)


def experiment1(config, train, val, test, stock, setting=None):
    """

    Args:
        config: Config to use
        train: Training data
        val:  Validation data
        test: Test data
        stock: Stock name to predict
        setting: set to 'Save' if you want to save the models

    Returns: models, training data, results on test set

    """

    save = False

    if setting == 'Save':
        save = True

    # In other settings, create the models and train them
    else:
        if stock == 'Apple':
            models = create_models_experiment1_apple()
        if stock == 'Heineken':
            models = create_models_experiment1_heineken()
        if stock == 'PostNL':
            models = create_models_experiment1_postnl()

        histories_val = train_models(models=models, train=train, val=val, verbose=0)

        # If setting is save, save the models and results
        if setting == 'Save':
            # Save models
            for i, model_ in enumerate(models):
                model_.model.save(config.path + 'Experiment1\\' + stock + '\\'+str(model_.model.name))

            # Save results
            utils.save_history(histories_val, path=config.path + 'Experiment1\\' + stock + '\\', name='histories_val')

    # Plot the results on validation data
    model_names = []

    for model_ in models:
        model_names.append(model_.model.name)

    try:
        plot_results(histories=histories_val, model_names=['TCN', 'HATCN', 'TCAN', 'ATCN'], combine=False,
                     share_y=True, ylim=[0, 0.05])
        plot_results(histories=histories_val, model_names=['TCN', 'HATCN', 'TCAN', 'ATCN'], combine=False,
                     share_y=True, ylim=[0, 0.01])
    except:
        print(f'Could not be plotted')

    # Test model on test set
    results = run_experiment(models=models, test=test)

    # Save results
    if setting == 'Save':
        utils.save_object(results, path=config.path + 'Experiment1\\' + stock + '\\', name='results')

    return models, histories_val, results


def experiment2(config, train, val, test, stock, model_name, model_index, train4, val4, test4, train5, val5, test5):
    """


    :param config:
    :param train:
    :param val:
    :param test:
    :param stock:
    :param model_name:
    :param model_index:
    :param train4: Train set for receptive field 271
    :param val4: Val set for receptive field 271
    :param test4: Test set for receptive field 271
    :param train5: Train set for receptive field 311
    :param val5: Val set for receptive field 311
    :param test5: Test set for receptive field 311
    :return:
    """

    if stock == 'Heineken':
        models_temp = create_models_experiment1_heineken()
    else:
        raise NotImplementedError

    cfg = models_temp[model_index].cfg

    # Create models and train them
    models = create_models_experiment2(model_name=model_name, config=cfg)

    histories_val = []

    for i, model_ in enumerate(models):
        if i < 2 or i >= 4:
            history = train_model(model_, train=train, val=val, verbose=0)
        if i == 2:
            history = train_model(model_, train=train4, val=val4, verbose=0)
        if i == 3:
            history = train_model(model_, train=train5, val=val5, verbose=0)

        histories_val.append(history)

    # Save models
    for i, model_ in enumerate(models):
        model_.model.save(config.path + 'Experiment2\\' + stock + '\\'+str(model_.model.name) + str(i + 2))

    # Save training results
    utils.save_history(histories_val, path=config.path + 'Experiment2\\' + stock + '\\', name='histories_val')

    # Plot results
    try:
        plot_results(histories=histories_val, model_names=['TCN', 'HATCN', 'TCAN', 'ATCN'], combine=False,
                     share_y=True, ylim=[0, 0.05])
        plot_results(histories=histories_val, model_names=['TCN', 'HATCN', 'TCAN', 'ATCN'], combine=False,
                     share_y=True, ylim=[0, 0.01])
    except:
        print("Could not plot graph")

    # Run experiment and save results
    results = run_experiment(models=models[:2], test=test)
    results2 = run_experiment(models=models[2], test=test4)
    results3 = run_experiment(models=models[3], test=test5)
    results4 = run_experiment(models=models[4], test=test)

    results.append(results2[0])
    results.append(results3[0])
    results.append(results4[0])

    utils.save_object(results, path=config.path + 'Experiment2\\' + stock + '\\', name='results')

    return models, histories_val, results


def create_models_experiment1_apple():
    models = []

    config1 = config_class.Config()
    config2 = config_class.Config()
    config3 = config_class.Config()
    config4 = config_class.Config()

    """TCN"""
    config1.learning_rate = 0.0001
    config1.filters = 100
    config1.layers = 3
    config1.kernel_size = 19
    config1.epochs = 1000
    model_ = build_model('TCN', config=config1)
    models.append(model_)

    """HATCN"""
    config2.learning_rate = 0.001
    config2.filters = 25
    config2.layers = 3
    config2.kernel_size = 19
    config2.epochs = 1000
    model_ = build_model('HATCN', config=config2)
    models.append(model_)

    """TCAN"""
    config3.learning_rate = 0.001
    config3.filters = 25
    config3.layers = 2
    config3.kernel_size = 43
    config3.epochs = 1000
    model_ = build_model('TCAN', config=config3)
    models.append(model_)

    """ATCN"""
    config4.learning_rate = 0.001
    config4.filters = 25
    config4.layers = 3
    config4.kernel_size = 19
    config4.epochs = 1000
    model_ = build_model('ATCN', config=config4)
    models.append(model_)

    return models


def create_models_experiment1_heineken():
    models = []

    config1 = config_class.Config()
    config2 = config_class.Config()
    config3 = config_class.Config()
    config4 = config_class.Config()

    """TCN"""
    config1.learning_rate = 0.0001
    config1.filters = 100
    config1.layers = 3
    config1.kernel_size = 19
    config1.epochs = 1000
    model_ = build_model('TCN', config=config1)
    models.append(model_)

    """HATCN"""
    config2.learning_rate = 0.001
    config2.filters = 25
    config2.layers = 6
    config2.kernel_size = 3
    config2.epochs = 1000
    model_ = build_model('HATCN', config=config2)
    models.append(model_)

    """TCAN"""
    config3.learning_rate = 0.001
    config3.filters = 25
    config3.layers = 2
    config3.kernel_size = 43
    config3.epochs = 1000
    model_ = build_model('TCAN', config=config3)
    models.append(model_)

    """ATCN"""
    config4.learning_rate = 0.001
    config4.filters = 25
    config4.layers = 3
    config4.kernel_size = 19
    config4.epochs = 1000
    model_ = build_model('ATCN', config=config4)
    models.append(model_)

    return models


def create_models_experiment1_postnl():
    models = []

    config1 = config_class.Config()
    config2 = config_class.Config()
    config3 = config_class.Config()
    config4 = config_class.Config()

    """TCN"""
    config1.learning_rate = 0.0001
    config1.filters = 100
    config1.layers = 3
    config1.kernel_size = 19
    config1.epochs = 1000
    model_ = build_model('TCN', config=config1)
    models.append(model_)

    """HATCN"""
    config2.learning_rate = 0.001
    config2.filters = 25
    config2.layers = 2
    config2.kernel_size = 43
    config2.epochs = 1000
    model_ = build_model('HATCN', config=config2)
    models.append(model_)

    """TCAN"""
    config3.learning_rate = 0.001
    config3.filters = 25
    config3.layers = 2
    config3.kernel_size = 43
    config3.epochs = 1000
    model_ = build_model('TCAN', config=config3)
    models.append(model_)

    """ATCN"""
    config4.learning_rate = 0.001
    config4.filters = 25
    config4.layers = 3
    config4.kernel_size = 19
    config4.epochs = 1000
    model_ = build_model('ATCN', config=config4)
    models.append(model_)

    return models


def create_models_experiment2(model_name, config):
    models = []
    config.epochs = 1000
    config.learning_rate = 0.001
    config.filters = 25

    config1 = copy.copy(config)
    config1.layers = 2
    config1.kernel_size = 43
    model_ = build_model(str(model_name), config=config1)
    models.append(model_)

    config2 = copy.copy(config)
    config2.layers = 3
    config2.kernel_size = 19
    model_ = build_model(str(model_name), config=config2)
    models.append(model_)

    config3 = copy.copy(config)
    config3.layers = 4
    config3.kernel_size = 9
    model_ = build_model(str(model_name), config=config3)
    models.append(model_)

    config4 = copy.copy(config)
    config4.layers = 5
    config4.kernel_size = 5
    model_ = build_model(str(model_name), config=config4)
    models.append(model_)

    config5 = copy.copy(config)
    config5.layers = 6
    config5.kernel_size = 3
    model_ = build_model(str(model_name), config=config5)
    models.append(model_)

    return models


def run_experiment(models, test):
    results = []

    try:
        for model_ in models:
            result = model_.model.evaluate(test)
            results.append(result)

    except:
        results = []
        result = models.model.evaluate(test)
        results.append(result)

    return results

