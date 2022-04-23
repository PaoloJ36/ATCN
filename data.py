import math
import yfinance as yf
import numpy as np
import utils
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


class Data:
    """
    Loads the data and handles the pre-processing.

    Arguments:
        'company' requires a string with the company name of the data to be used
        'method' requires a number from 1 to 6 for which preprocessing to use.
        In case save all is used, all preprocessing functions will be used regardless but only the specified
        method one will be stored in this object as self.data.
        (Optional) 'mode' requires 'Save', 'Save all', 'Load' or 'Standalone' for what to do with the data
        retrieval/storing. Standalone means it will simply retrieve but not load or save from files.

    """

    def __init__(self, company: str, method: int, config, mode=None):

        self.tickers = {"Apple": "AAPL",
                        "Heineken": "HEIA.AS",
                        "PostNL": "PNL.AS"}

        # Dict of all preprocess methods
        self.preprocess_dict = {
            '1': self.__preprocess_method_1,
            '2': self.__preprocess_method_2
        }

        self.config = config
        self.raw_data = None
        self.data = None
        self.company = company
        self.method = str(method)

        # Handles the optional arguments. When no arguments, this does nothing
        if mode is not None:
            self.config.mode = mode

        # Handle the data collection
        self.__handle_mode(company)

        # Build a window
        self.window = self.__build_window()

    def __handle_mode(self, company):
        """
        If mode is:
        Load: Load the data
        Standalone: Retrieve, preprocess specified method
        Save: Retrieve, preprocess specified method and then save
        Save all: Retrieve, preprocess in all methods and save them independently.
        """

        if self.config.mode == 'Load':
            self.__load_data()
            return

        self.raw_data = self.__retrieve_data()

        if self.config.mode == 'Standalone':
            self.data = self.__preprocess_data(self.raw_data, self.method)
            return

        if self.config.mode == 'Save':
            # Save raw data
            self.__save_data(data=self.raw_data, name='_raw')

            # Preprocess and then save data
            self.data = self.__preprocess_data(self.raw_data, self.method)
            self.__save_data(data=self.data, name=self.method)

        if self.config.mode == 'Save all':
            # Save raw data
            self.__save_data(data=self.raw_data, name='_raw')

            # Use all preprocessing methods and save data independently
            for method in range(1, len(self.preprocess_dict)):
                data = self.__preprocess_data(self.raw_data, str(method))
                self.__save_data(data=data, name=str(method))

                # Use the method given as parameter as the data to use in this run
                if str(method) == self.method:
                    self.data = data

    def __retrieve_data(self):
        """
        Calls all functions to retrieve the data from various sources.
        """

        data = self.__get_data_from_yfinance(self.company)
        # split_index = data.index.get_loc(self.config.train_test_split_date)
        return data

    def __load_data(self):
        """
        Loads the raw data and preprocessed data from the filepath specified in the config

        """
        path = self.config.path + 'Data\\' + self.company + '\\'
        name = 'Data' + str(self.method)
        raw_name = 'Data_raw'

        self.data = utils.load_object(path=path, name=name, filetype='obj')
        self.raw_data = utils.load_object(path=path, name=raw_name, filetype='obj')

        if self.data is None:
            print(f"Data could not be loaded from path {path}{name}.obj")
        else:
            print(f"Data loaded from to {path}{name}.obj")

        if self.raw_data is None:
            print(f"Raw data could not be loaded from path {path}{raw_name}.obj")
        else:
            print(f"Raw data loaded from to {path}{raw_name}.obj")

    def __save_data(self, data, name: str):
        """
        Saves the data to the savepath and uses the company name as filename
        If raw data is provided, also save that separately
        """
        path = self.config.path + 'Data\\' + self.company + '\\'
        name = 'Data' + name

        utils.save_object(object_to_save=data, path=path, name=name, filetype='obj')
        print(f"Data saved to {path}{name}.obj")

    def __preprocess_data(self, raw_data, method):
        """Handles all the preprocessing, including normalization"""
        data = self.preprocess_dict[method](self, data=raw_data)  # Calls the function stated in the dict
        data = self.__correct_dates(data)
        data = data.fillna(value=0)  # If there are NA's because of a divide by zero, make them zero
        return data

    def __get_data_from_yfinance(self, company):
        """Gets the data from yfinance according to the tickers and dates listed in the config
        Uses the company name to find the ticker in the self.tickets dict"""
        ticker = yf.Ticker(self.tickers[company])
        data = ticker.history(start=self.config.corrected_start_date, end=self.config.end_date)
        return data

    def __correct_dates(self, data):
        """Ensures only data from the start date is in the data set. Needed because some methods changes relative to
        previous values.
        """
        start_index = data.index.get_loc(self.config.start_date)
        data = data[start_index:]
        return data

    def __build_window(self):
        """Returns a windowgenerator with train, val and test sets using the data from this object"""
        train_df, val_df, test_df = self.__split_data()
        sequence_length = utils.receptive_field(layers=self.config.layers, kernel_size=self.config.kernel_size)

        return WindowGenerator(input_width=sequence_length, label_width=1, shift=1,
                               label_columns=['Close'], train_df=train_df, val_df=val_df, test_df=test_df,
                               config=self.config)

    def __split_data(self):
        """
        Calculates the slices for training, validation and testing set, then generates the sets and returns
        the dataframes

        Retrieves the boundary from training and testing from train_test_split_date in the config
        Calculates the boundary from training and validation based on the validation percentage in the config
        """
        split_index = self.data.index.get_loc(self.config.train_test_split_date)
        validation_index = split_index - (math.floor(split_index * self.config.validation_percentage / 100))

        train_df = pd.DataFrame(self.data[:validation_index])
        val_df = pd.DataFrame(self.data[validation_index:split_index])
        test_df = pd.DataFrame(self.data[split_index:])

        return train_df, val_df, test_df

    """ 
    ********************************
    Helper functions used for normalization of data
    ********************************
    """

    @staticmethod
    def __normalize(self, data):
        """Normalizes the values of each column in a range of 0 to 1 according to the min-max method"""

        normalized = (data - data.min()) / (data.max() - data.min())
        return normalized

    @staticmethod
    def __standardize(self, data):
        """Standardizes the value of each column with a std of 1 and a mean of 0"""
        mean = np.mean(data)
        std = np.std(data)
        standardized = (data - mean) / std
        return standardized

    """ 
    ********************************
    Full preprocess methods for normalization of data
    ********************************
    """

    @staticmethod
    def __preprocess_method_1(self, data):
        """ 'Standardize everything'
        Open, High, Low, Close:         Standardizes
        Volume:                         Standardizes
        Dividends:                      Standardizes
        Stock splits:                   Standardizes
        """

        data = self.__standardize(self, data=data)
        return data

    @staticmethod
    def __preprocess_method_2(self, data):
        """ 'Normalize everything'
        Open, High, Low, Close:         Normalizes [0, 1]
        Volume:                         Normalizes [0, 1]
        Dividends:                      Normalizes [0, 1]
        Stock splits:                   Normalizes [0, 1]
        """

        data = self.__normalize(self, data)
        return data


class WindowGenerator:
    """Generates a sliding window of the data.
    input_width = How many timesteps are in the window.
    label_width = How many variables to predict
    shift = How many timesteps between the input values and prediction. Use 1 to predict next timestep.
    train_df = Dataframe containing training data
    val_df = Dataframe containing validation data
    test_df = Dataframe containing test data
    label_columns = Array of labels for the prediction variables. Format like 'label_columns=['Open', 'Close']'


    Based on https://www.tensorflow.org/tutorials/structured_data/time_series
    """

    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df, config, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.config = config

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self._example = None

    def __repr__(self):
        """Function that indicates how to print the object"""
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        """
        Given a list of consecutive inputs, the split_window method will convert them to a window of inputs and a
        window of labels.
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col=None, max_subplots=3):
        """Function that plots the split window"""
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

            plt.xlabel('Days')

    def make_dataset(self, data):
        """
        This function will take a time series DataFrame and convert it to a tf.data.Dataset of
        (input_window, label_window) pairs using the keras preprocessing.timeseries_dataset_from_array function
        """

        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.config.batch_size)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
