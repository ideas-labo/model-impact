import tensorflow.compat.v1 as tf
import tensorflow as tf2
tf.disable_v2_behavior()
import numpy as np
import sys

class MLPHierarchicalModel():
    """Class for hierarchical mlp models"""

    def __init__(self, config):
        """
        Args:
            config: configures to define model, which should contain:
                - input_dim: [int] number of configurations for the dataset (i.e., column dimension)
                - num_neuron: [int] number of neurons in each MLP layer
                - num_block: [int] number of blocks in the network
                - num_layer_pb: [int] number of layers in per block
                - decay: [float] fraction to decay learning rate
                - verbose: whether print the intermediate results
        """
        self.input_dim = config['input_dim']
        self.num_neuron = config['num_neuron']
        self.num_block = config['num_block']
        self.num_layer_pb = config['num_layer_pb']
        self.lamda = config['lamda']
        self.use_linear = config['linear']
        self.decay = config['decay']
        self.verbose = config['verbose']
        self.name = 'MLPHierarchicalModel'

        tf.reset_default_graph()  # Saveguard if previous model was defined
        tf.set_random_seed(1)     # Set tensorflow seed for paper replication

    def __build_neural_net(self):
        input_layer = self.X
        output = None
        for block_id in range(self.num_block):
            backcast, forecast = self.__create_block(input_layer)
            input_layer = tf.concat([input_layer, backcast], 1)
            if block_id == 0:
                output = forecast
            else:
                output = output + forecast

        if self.use_linear:
            linear_input = self.X
            linear_output = tf.layers.dense(linear_input, 1, kernel_regularizer=tf.keras.regularizers.l2(float(self.lamda)))
            output = output + linear_output
        
        return output
    
    def __create_block(self, x):
        layer = x
        for i in range(self.num_layer_pb):
            if i == 0:
                layer = tf.layers.dense(layer, self.num_neuron, tf.nn.relu,
                                        kernel_initializer=tf2.initializers.GlorotUniform(seed=1),
                                        kernel_regularizer=tf.keras.regularizers.l1(float(self.lamda)))
            else:
                layer = tf.layers.dense(layer, self.num_neuron, tf.nn.relu,
                                        kernel_initializer=tf2.initializers.GlorotUniform(seed=1))
        backcast = tf.layers.dense(layer, self.input_dim, tf.nn.relu)
        forecast = tf.layers.dense(layer, 1)

        return backcast, forecast
    
    def build_train(self):
        """Builds model for training"""
        self.__add_placeholders_op()
        self.__add_pred_op()
        self.__add_loss_op()
        self.__add_train_op()

        self.init_session()

    def __add_placeholders_op(self):
        """ Add placeholder attributes """
        self.X = tf.placeholder("float", [None, self.input_dim])
        self.Y = tf.placeholder("float", [None, 1])
        self.lr = tf.placeholder("float")  # to schedule learning rate
    
    def __add_pred_op(self):
        """Defines self.pred"""
        self.output = self.__build_neural_net()
    
    def __add_loss_op(self):
        """Defines self.loss"""
        l2_loss = tf.losses.get_regularization_loss()
        self.loss = l2_loss + tf.losses.mean_squared_error(self.Y, self.output)
    
    def __add_train_op(self):
        """Defines self.train_op that performs an update on a batch"""
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads, vs     = zip(*optimizer.compute_gradients(self.loss))
            grads, gnorm  = tf.clip_by_global_norm(grads, 1)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))
    
    def init_session(self):
        """Defines self.sess, self.saver and initialize the variables"""
        config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False) 
 
        config.gpu_options.allow_growth = True 
 
        self.sess = tf.Session(config=config)

        #self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def finalize(self):
        self.sess.close()
        tf.get_default_graph().finalize()

    def build_train_1(self):
        """Builds model for training"""
        self.__add_placeholders_op()
        self.__add_pred_op()
        self.__add_loss_op()
        self.__add_train_op()

        # self.init_session()
