from utils.HINNPerf_data_preproc import system_samplesize, seed_generator, DataPreproc
import numpy as np
import time
import tensorflow as tf

class ModelRunner():
    """Generic class for training models"""

    def __init__(self, data_preproc, model_class):
        """
        Args:
            data_preproc: [DataPreproc object] preprocess and generate training data
            model_class: [class] deep learning model class
        """
        self.data_preproc = data_preproc
        self.model_class = model_class
        self.model = None
    
    def train(self, config):
        """
        Train the model

        Args:
            config: configures to create a model object
        """

        if (config['gnorm']):
            X_train, Y_train, X_valid, Y_valid, mean_Y, std_Y = self.data_preproc.get_train_valid_samples(config['gnorm'])
        else:
            X_train, Y_train, X_valid, Y_valid, max_Y = self.data_preproc.get_train_valid_samples(config['gnorm'])
        # print(Y_valid)
        self.model = self.model_class(config)
        #print("Start training " + model.name + "...")
        self.model.build_train()
        
        lr = config['lr']
        decay = lr/1000
        train_seed = 0
        # print(X_train)
        # print(Y_train)
        # print(lr)
        for epoch in range(1, 500): #2000
            train_seed += 1
            _, cur_loss, pred = self.model.sess.run([self.model.train_op, self.model.loss, self.model.output],
                                               {self.model.X:X_train, self.model.Y:Y_train, self.model.lr:lr})
                        
            #if epoch % 500 == 0 or epoch == 1:
            #    rel_error = np.mean(np.abs(np.divide(Y_train.ravel() - pred.ravel(), Y_train.ravel())))
            #    if model.verbose:
            #        print("Cost function: {:.4f}", cur_loss)
            #        print("Train relative error: {:.4f}", rel_error)
                        
            lr = lr*1/(1 + decay*epoch)
                
        Y_pred_train = self.model.sess.run(self.model.output, {self.model.X: X_train})
        abs_error_train = np.mean(np.abs(Y_pred_train - Y_train))

        Y_pred_val = self.model.sess.run(self.model.output, {self.model.X: X_valid})
        # print(Y_pred_val,Y_valid)
        if (config['gnorm']):
            Y_pred_val = Y_pred_val * (std_Y + (std_Y == 0) * .001) + mean_Y
            Y_valid = Y_valid * (std_Y + (std_Y == 0) * .001) + mean_Y
        else:
            Y_pred_val = Y_pred_val * max_Y
            Y_valid = Y_valid * max_Y
        # print(Y_pred_val,Y_valid)
        abs_error_val = np.mean(np.abs(Y_pred_val - Y_valid))

        # self.model.finalize()
        self.sess = self.model.sess
        # print(self.model)
        return abs_error_train, abs_error_val

    def predict(self, X_test,best_config=dict(gnorm=True)):
        if best_config['gnorm']:
            X_train, Y_train, mean_Y, std_Y, mean_X, std_X = self.data_preproc.get_train_test_samples(best_config['gnorm'])
            X_test = (X_test - mean_X) / (std_X + (std_X == 0) * .001)
        else:
            X_train, Y_train,  max_Y, max_X = self.data_preproc.get_train_test_samples(best_config['gnorm'])
            X_test = np.divide(X_test, max_X)
        
        # model = self.model_class(best_config)
        # #print("Retraining " + model.name + " and testing ...")
        # model.build_train()

        # lr = best_config['lr']
        # decay = lr/1000
        # train_seed = 0
        # for epoch in range(1, 200):  #2000
        #     train_seed += 1
        #     _, cur_loss, pred = model.sess.run([model.train_op, model.loss, model.output],
        #                                        {model.X:X_train, model.Y:Y_train, model.lr:lr})
            
            # if epoch % 500 == 0 or epoch == 1:
            #     rel_error = np.mean(np.abs(np.divide(Y_train.ravel() - pred.ravel(), Y_train.ravel())))
                # if model.verbose:
                    # print("Cost function: {:.4f}", cur_loss)
                    # print("Train relative error: {:.4f}", rel_error)
            
            # lr = lr*1/(1 + decay*epoch)
        


        Y_pred_test = self.model.sess.run([self.model.output], {self.model.X: X_test})

        if best_config['gnorm']:
            Y_pred_test = Y_pred_test * (std_Y + (std_Y == 0) * .001) + mean_Y
        else: 
            Y_pred_test = max_Y * Y_pred_test
        # rel_error = np.mean(np.abs(np.divide(Y_test.ravel() - Y_pred_test.ravel(), Y_test.ravel())))
        # print('Prediction relative error (%): {:.2f}'.format(np.mean(rel_error)*100))

        # model.finalize()

        return Y_pred_test
    

    # def predict(self, best_config):
    #     model = self.model_class(best_config)
    #     #print("Retraining " + model.name + " and testing ...")
    #     model.build_train()
    #     Y_pred_test = model.sess.run([model.output], {model.X: X_test})
    #     return Y_pred_test
    def Build_train(self,best_config):
        self.model = self.model_class(best_config)
        #print("Start training " + model.name + "...")
        self.model.build_train()