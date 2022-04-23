"""
Config file where you can edit the settings of the run
"""

"""
******************
   DATA CONFIGS
******************
"""


class Config:

    def __init__(self):
        self.corrected_start_date = "1999-12-29"  # Needs to be earlier because of relative changes being used in some normalizations
        self.start_date = "2000-1-3"  # Inclusive
        self.end_date = "2020-10-2"  # Not inclusive, needs to be one day later to account for test set results
        self.train_test_split_date = "2015-1-2"  # Not inclusive

        self.mode = 'Save all'  # 'Load', 'Save', 'Save all', 'Standalone' - Load the data from path below, Save the data to path
        #                    below, save all methods to the path or do a standalone, where it neither saves or loads
        self.path = 'D:\\Bureaublad mappen\\School\\Master AI\\Scriptie\\Code\\'  # Path where the project is found.

        """
        ******************
           MODEL CONFIGS
        ******************
        """

        self.batch_size = 256
        self.learning_rate = 0.001
        self.input_variables = 7
        self.filters = 50  # How many parallel filters are used for a 1v1 convolution
        self.kernel_size = 19  # How many inputs are used for one node
        self.layers = 3
        self.epochs = 200
        self.weight_normalization = True
        self.validation_percentage = 20
        self.residual = 'Block'  # 'Block', 'Layer' or 'None'

        self.parameter_grid = {'learning_rate': [0.001, 0.0001],
                               'filters': [25, 50, 100],
                               'layers': [2, 3, 6],
                               'kernel_size': [3, 19, 43],
                               'epochs': [100, 300, 1000]
                               }
        self.experiment1 = True
        self.experiment2 = True

