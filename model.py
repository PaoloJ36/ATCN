import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
import math
from tcn import TCN, tcn_full_summary
import utils
import numpy as np


class Model:
    """Everything related to the model. Can build, load or save it."""

    def __init__(self, model_name, config):
        """Handles everything model-wise. Load or build it and optionally save it.
        Model specs are optional arguments, defaults to the config file.
        """

        self.model_choice = {
            'TCN': self.__build_tcn,
            'HATCN': self.__build_hatcn,
            'TCAN': self.__build_tcan,
            'ATCN': self.__build_atcn,
            'TCN_LIB': self.__build_prebuilt_tcn
        }
        self.cfg = config

        # Does everything needed for the model like building, saving, loading
        self.model = self.get_model(self, self.model_choice[model_name])

    def load(self, path):
        pass

    @staticmethod
    def get_model(self, build_function):
        model = build_function()
        return model

    @staticmethod
    def get_callback():
        return keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000, mode='min', restore_best_weights=True)


    """
    **********************
            Models
    **********************
    """

    def __build_tcn(self):
        """
        Builds the standard TCN and returns the model.

        Input layer shape = sequence length, channels
        """
        receptive_field = utils.receptive_field(layers=self.cfg.layers, kernel_size=self.cfg.kernel_size,
                                                print_in_console=False)
        inputs = keras.Input(shape=(receptive_field, self.cfg.input_variables), name='input_layer')
        skip_x = inputs
        x = inputs

        # Create layers according to the amount of layers in the config
        for i in range(0, self.cfg.layers):
            if self.cfg.residual == 'Layer':
                skip_x = self.__layer_skip_convolution(layer_number=i)(x)

            x = self.__layer_tcn(layer_number=i)(x)  # Normal TCN layer

            x = self.__residual_handler(x=x, skip_input=skip_x, i=i)

        # Output layer
        x = layers.Flatten()(x)
        outputs = layers.Dense(1, name='Output_layer')(x)

        # Create and compile the model
        model = keras.Model(inputs, outputs, name="Temporal_Convolutional_Network")
        optimizer = keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mse", "mae"])

        return model

    def __build_prebuilt_tcn(self):
        """Only used for comparison with the selfmade TCN, to test if everything works as expected"""

        tcn_layer = TCN(input_shape=(self.cfg.sequence_length, self.cfg.input_variables),
                        nb_filters=self.cfg.filters, kernel_size=self.cfg.kernel_size, dilations=[1, 2, 4, 8, 16, 32]
                        , nb_stacks=1, padding='causal', use_skip_connections=True, activation="relu",
                        use_weight_norm=self.cfg.weight_normalization)
        print('Receptive field size =', tcn_layer.receptive_field)

        model = keras.models.Sequential([
            tcn_layer,
            keras.layers.Dense(1)
        ])

        optimizer = keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mse", "mae"])

        return model

    def __layer_tcn(self, layer_number):
        """
        Creates a basic TCN layer consisting of a causal dilated 1x1 convolution
        dilation rate = 2^i
        where i = layer number
        kernel size = how many timestep activations lead to a single point in the next layer
        """

        layer = layers.Conv1D(strides=1, activation="relu", padding="causal",
                              kernel_size=self.cfg.kernel_size,
                              filters=self.cfg.filters,
                              dilation_rate=int(math.pow(2, layer_number)),
                              name='Causal_Dilated_Conv_' + str(layer_number + 1))

        if self.cfg.weight_normalization:
            layer = tfa.layers.WeightNormalization(layer)

        return layer

    @staticmethod
    def __layer_batch_norm(layer_number):
        layer = layers.BatchNormalization(name='batch_norm'+str(layer_number), momentum=0.1)

        return layer

    def __residual_handler(self, x, skip_input, i):

        if self.cfg.residual == 'Layer':
            x = layers.Add(name='Add_residuals_' + str(i + 1))(
                [skip_input, x])  # Add result of TCN layer to skip connection

        if self.cfg.residual == 'Block' and i == self.cfg.layers - 1:
            skip_input = self.__layer_skip_convolution(layer_number=i)(skip_input)

            x = layers.Add(name='Add_residuals_' + str(i + 1))(
                [skip_input, x])  # Add result of TCN layer to skip connection

        return x

    def __layer_skip_convolution(self, layer_number):
        """
        Creates a skip connection using a 1x1 convolution to match the output shape of this layer
        """
        layer = layers.Conv1D(filters=self.cfg.filters, padding="same",
                              kernel_size=1, name='Skip_Conv_' + str(layer_number + 1))

        return layer

    def __build_hatcn(self):

        receptive_field = utils.receptive_field(layers=self.cfg.layers, kernel_size=self.cfg.kernel_size,
                                                print_in_console=False)
        inputs = keras.Input(shape=(receptive_field, self.cfg.input_variables), name='input_layer')
        skip_x = inputs
        x = inputs

        # Create layers according to the amount of layers in the config
        for i in range(0, self.cfg.layers):

            # if self.cfg.residual == 'Layer':
                # skip_x = self.__layer_skip_convolution(layer_number=i)(x)

            x = self.__layer_tcn(layer_number=i)(x)  # Normal TCN layer

            # x = self.__residual_handler(x=x, skip_input=skip_x, i=i)

            x2 = Custom_Attention(weight_columns=self.cfg.filters, weight_rows=1, transpose_input=True,
                                  attention_type='Hierarchical')(x)
            # ^ Within layer attention

            if i == 0:
                xx = x2
            else:
                xx = tf.concat([xx, x2], -1)  # Add result of attention to matrix

        # Between layer attention
        x = Custom_Attention(weight_columns=self.cfg.filters, weight_rows=1, transpose_input=False,
                             attention_type='Hierarchical')(xx)
        # Output layer
        x = layers.Flatten()(x)
        skip_x = layers.Flatten()(skip_x)
        x = tf.concat([x, skip_x], -1)

        # print(x.get_shape())
        outputs = layers.Dense(1, name='Output_layer')(x)

        # Create and compile the model
        model = keras.Model(inputs, outputs, name="Hierarchical_Temporal_Convolutional_Network")
        optimizer = keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mse", "mae"])

        return model

    def __build_tcan(self):
        receptive_field = utils.receptive_field(layers=self.cfg.layers, kernel_size=self.cfg.kernel_size,
                                                print_in_console=False)
        inputs = keras.Input(shape=(receptive_field, self.cfg.input_variables), name='input_layer')
        skip_x = inputs
        x = inputs

        # Create layers according to the amount of layers in the config
        for i in range(0, self.cfg.layers):

            if self.cfg.residual == 'Layer':
                skip_x = self.__layer_skip_convolution(layer_number=i)(x)

            x = self.__layer_tcn(layer_number=i)(x)  # Normal TCN layer
            x = Custom_Attention(weight_columns=receptive_field, weight_rows=self.cfg.filters,
                                 transpose_input=True, attention_type='Temporal')(x)  # Temporal attention

            x = self.__residual_handler(x=x, skip_input=skip_x, i=i)

        # Output layer
        x = layers.Flatten()(x)
        outputs = layers.Dense(1, name='Output_layer')(x)

        # Create and compile the model
        model = keras.Model(inputs, outputs, name="Temporal_Convolutional_Attention-based_Network")
        optimizer = keras.optimizers.Adam(learning_rate=self.cfg.learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mse", "mae"])

        return model

    def __build_atcn(self):
        receptive_field = utils.receptive_field(layers=self.cfg.layers, kernel_size=self.cfg.kernel_size,
                                                print_in_console=False)
        inputs = keras.Input(shape=(receptive_field, self.cfg.input_variables), name='input_layer')
        skip_x = inputs
        x = inputs

        # Create layers according to the amount of layers in the config
        for i in range(0, self.cfg.layers):

            # if self.cfg.residual == 'Layer':
                # skip_x = self.__layer_skip_convolution(layer_number=i)(x)

            x = self.__layer_tcn(layer_number=i)(x)  # Normal TCN layer
            x = Custom_Attention(weight_columns=receptive_field, weight_rows=self.cfg.filters,
                                 transpose_input=True, attention_type='Temporal')(x)  # Temporal attention

            # x = self.__residual_handler(x=x, skip_input=skip_x, i=i)

            x2 = Custom_Attention(weight_columns=self.cfg.filters, weight_rows=1, transpose_input=True,
                                  attention_type='Hierarchical')(x)  # Within-layer attention

            if i == 0:
                xx = x2
            else:
                xx = tf.concat([xx, x2], -1)  # Add result of attention to matrix

        # Between layer attention
        x = Custom_Attention(weight_columns=self.cfg.filters, weight_rows=1, transpose_input=False,
                             attention_type='Hierarchical')(xx)

        # Output layer
        x = layers.Flatten()(x)
        skip_x = layers.Flatten()(skip_x)
        x = tf.concat([x, skip_x], -1)

        outputs = layers.Dense(1, name='Output_layer')(x)

        # Create and compile the model
        model = keras.Model(inputs, outputs, name="Attentive_Temporal_Convolutional_Network")
        model.compile(optimizer=self.cfg.optimizer, loss="mse", metrics=["mse", "mae"])
        return model


class Custom_Attention(layers.Layer):
    """Custom attention layer
    Works according to the formula alpha = softmax(tanh(w_T * H))
    Gamma = ReLU(H * alpha_T)
    where w is a trainable weight vector, H is the input, and _T is transposed

    **** Hierarchical attention ****
    weight_columns = Amount of filter kernels.
    weight_rows = 1
    Transpose input = If within layer attention, then yes.

    **** Temporal attention ****
    weight_columns = Amount of filter kernels
    weight_rows = Amount of timesteps
    Transpose input = Always

    **********
    Output size = columns x rows

    """

    def __init__(self, weight_columns, weight_rows, transpose_input, attention_type):
        super(Custom_Attention, self).__init__()
        self.w = self.add_weight(
            shape=(weight_columns, weight_rows), initializer="random_normal", trainable=True, name='weights'
        )

        self.transpose_input = transpose_input
        self.type = attention_type
        # print(f'weight shape: {weight_columns, weight_rows}, should be 1 or timesteps x filter size')

    def call(self, inputs):

        if self.type == 'Hierarchical':
            alpha = tf.nn.softmax(
                tf.nn.tanh(tf.matmul(self.w, inputs, transpose_a=True, transpose_b=self.transpose_input)))

            gamma = tf.nn.relu(tf.matmul(inputs, alpha, transpose_a=self.transpose_input, transpose_b=True))

        if self.type == 'Temporal':
            alpha = tf.nn.softmax(
                tf.nn.tanh(tf.matmul(self.w, inputs, transpose_a=True)))

            gamma = tf.nn.relu(tf.matmul(inputs, alpha))

        # print(f'input shape: {inputs.get_shape()}')
        # print(f'alpha shape: {alpha.get_shape()}')
        # print(f'gamma shape: {gamma.get_shape()}')

        return gamma

    def get_config(self):
        """https://linuxtut.com/en/b2154b661e7baf56031e/"""
        config = {
            "transpose_input": self.transpose_input,
            "type": self.type,
            "w": np.array(self.w)  # tf.Tensor is not serializable, need to convert to numpy array
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Baseline(tf.keras.Model):
    """
    Baseline model that takes the value of Y at timestep x-1 as prediction value
    label_index argument requires window.column_indices['ColumnName']

    https://www.tensorflow.org/tutorials/structured_data/time_series#baseline
    """

    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]

