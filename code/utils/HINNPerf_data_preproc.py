import numpy as np
from numpy import genfromtxt

def system_samplesize(sys_name):
    """
    Define the training sample size of different datasets
    
    Args:
        sys_name: [string] the name of the dataset
    """
    l = np.linspace(0.01, 0.15, 15)

    if (sys_name == 'x264'):
        N_train_all = np.multiply(16, [1, 2, 4, 6])  # This is for X264
    elif (sys_name == 'BDBJ'):
        #N_train_all = (l * 180).astype('int')
        N_train_all = np.multiply(26, [1, 2, 4, 6])  # This is for BDBJ
    elif (sys_name == 'lrzip'):
        #N_train_all = (l * 432).astype('int')
        N_train_all = np.multiply(19, [1, 2, 4, 6])  # This is for LRZIP
    elif (sys_name == 'polly'):
        #N_train_all = (l * 60000).astype('int')
        N_train_all = np.multiply(39, [1, 2, 4, 6])  # This is for POLLY
    elif (sys_name == 'vp9'):
        #N_train_all = (l * 216000).astype('int')
        N_train_all = np.multiply(41, [1, 2, 4, 6])  # This is for VP9
    elif (sys_name == 'Dune'):
        #N_train_all = (l * 2304).astype('int')
        N_train_all = np.asarray([49, 78, 384, 600])  # This is for Dune
    elif (sys_name == 'hipacc'):
        #N_train_all = (l * 13485).astype('int')
        N_train_all = np.asarray([261, 528, 736, 1281])  # This is for hipacc
    elif (sys_name == 'hsmgp'):
        #N_train_all = (l * 3456).astype('int')
        N_train_all = np.asarray([77, 173, 384, 480])  # This is for hsmgp
    elif (sys_name == 'javagc'):
        #N_train_all = (l * 166975).astype('int')
        N_train_all = np.asarray([855, 2571, 3032, 5312])  # This is for javagc
    elif (sys_name == 'sac'):
        #N_train_all = (l * 62523).astype('int')
        N_train_all = np.asarray([2060, 2295, 2499, 3261])  # This is for sac
    else:
        raise AssertionError("Unexpected value of 'sys_name'!")

    return N_train_all

def seed_generator(sys_name, sample_size):
    """
    Generate the initial seed for each sample size (to match the seed of the results in the paper)
    This is just the initial seed, for each experiment, the seeds will be equal the initial seed + the number of the experiment

    Args:
        sys_name: [string] the name of the dataset
        sample_size: [int] the total number of samples
    """

    N_train_all = system_samplesize(sys_name)
    if sample_size in N_train_all:
        seed_o = np.where(N_train_all == sample_size)[0][0]
    else:
        seed_o = np.random.randint(1, 101)

    return seed_o

class DataPreproc():
    """ Generic class for data preprocessing """

    def __init__(self, X_train,Y_train):
        """
        Args:
            sys_name: [string] the name of the dataset
        """
        # self.sys_name = sys_name
        # self.data_dir = 'datasets/' + sys_name + '_AllNumeric.csv'
        self.lens = len(X_train)
        self.X_train_all = np.array(X_train)
        Y_train = [[i] for i in Y_train]
        self.Y_train_all = np.array(Y_train)
        


    
    def __read_whole_data(self):
        # print('Read whole dataset ' + self.sys_name + ' from csv file ...')
        # self.whole_data = genfromtxt(self.data_dir, delimiter=',', skip_header=1)
        (self.all_sample_num, config_num) = self.whole_data.shape
        self.config_num = config_num - 1

        self.X_all = self.whole_data[:, 0:self.config_num]
        self.Y_all = self.whole_data[:, self.config_num][:, np.newaxis]
    
    def __normalize(self, X, Y):
        """
        Normalize the data and labels
        Args:
            X: [sample_size, config_size] features
            Y: [sample_size, 1] labels
        """
        max_X = np.amax(X, axis=0)              # [sample_size, config_size] --> [config_size]
        if 0 in max_X: max_X[max_X == 0] = 1
        X_sample = np.divide(X, max_X)

        max_Y = np.max(Y)/100
        if max_Y == 0: max_Y = 1
        Y_sample = np.divide(Y, max_Y)

        return X_sample, Y_sample, max_X, np.array([max_Y])
    
    def __normalize_gaussian(self, X, Y):
        """
        Normalize the data and labels
        Args:
            X: [sample_size, config_size] features
            Y: [sample_size, 1] labels
        """
        mean_X = np.mean(X, axis=0)
        std_X = np.std(X, axis=0)
        X_sample = (X - mean_X) / (std_X + (std_X == 0) * .001)

        mean_Y = np.mean(Y, axis=0)
        std_Y = np.std(Y, axis=0)
        Y_sample = (Y - mean_Y) / (std_Y + (std_Y == 0) * .001)

        return X_sample, Y_sample, mean_X, std_X, mean_Y, std_Y
    
    def get_train_valid_samples(self, gnorm=False):
        """
        Args:
            sample_size: [int] the total number of training samples
        """
        # np.random.seed(seed)
        # permutation = np.random.permutation(self.all_sample_num)
        # training_index = permutation[0:sample_size]
        
        sample_size = self.lens
        
        sample_cross = int(np.ceil(sample_size * 2 / 3))
        # print(self.X_train_all)
        X_train = self.X_train_all[0:sample_cross]
        Y_train = self.Y_train_all[0:sample_cross]
        X_valid = self.X_train_all[sample_cross:sample_size]
        Y_valid = self.Y_train_all[sample_cross:sample_size]
        # print(X_train,Y_train,X_valid,Y_valid)

        if gnorm:
            X_train_norm, Y_train_norm, mean_X, std_X, mean_Y, std_Y = self.__normalize_gaussian(X_train, Y_train)
            X_valid_norm = (X_valid - mean_X) / (std_X + (std_X == 0) * .001)
            Y_valid_norm = (Y_valid - mean_Y) / (std_Y + (std_Y == 0) * .001)

            return X_train_norm, Y_train_norm, X_valid_norm, Y_valid_norm, mean_Y, std_Y
        else:
            X_train_norm, Y_train_norm, max_X, max_Y = self.__normalize(X_train, Y_train)
            X_valid_norm = np.divide(X_valid, max_X)
            Y_valid_norm = np.divide(Y_valid, max_Y)
            
            return X_train_norm, Y_train_norm, X_valid_norm, Y_valid_norm, max_Y

    def get_train_test_samples(self, gnorm=False):
        # np.random.seed(seed)
        # permutation = np.random.permutation(self.all_sample_num)
        # training_index = permutation[0:sample_size]
        X_train = self.X_train_all
        Y_train = self.Y_train_all
        # testing_index = np.setdiff1d(np.array(range(self.all_sample_num)), training_index)

        if gnorm:
            X_train_norm, Y_train_norm, mean_X, std_X, mean_Y, std_Y = self.__normalize_gaussian(X_train, Y_train)   

            return X_train_norm, Y_train_norm,  mean_Y, std_Y, mean_X, std_X
        else:
            X_train_norm, Y_train_norm, max_X, max_Y = self.__normalize(X_train, Y_train)

            return X_train_norm, Y_train_norm,  max_Y, max_X
