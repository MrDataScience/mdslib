# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:27:17 2020

@author: Mr. Data Science
"""
import numpy as np

class Dataset:
    def __init__(self):
        ''' Constructor for this class. '''
        # training set variables
        self.X_train = []
        self.Y_train = []
        
        # test set variables
        self.X_test  = []
        self.Y_test  = []
            
    def display_info(self):
        # display training set info
        print('Training Set:')
        print(1 * '\t' + 'Shape:')
        print(2 * '\t' + 'X_train: ' + str(np.shape(self.X_train)))
        print(2 * '\t' + 'Y_train: ' + str(np.shape(self.Y_train)))
        
        # display test set info
        print('Test Set:')
        print(1 * '\t' + 'Shape:')
        print(2 * '\t' + 'X_test: '  + str(np.shape(self.X_test)))
        print(2 * '\t' + 'Y_test: '  + str(np.shape(self.Y_test)))
        
    def load_mnist(self, filename='mnist.npz'):
        temp = __file__
        temp = temp.replace('\\','/')
        fullpath = temp[:temp.rindex('/')+1]+'data/'+filename
        
        from tensorflow.keras.datasets import mnist
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data(path=fullpath)
        
if __name__ == '__main__':
    data = Dataset()
    
    # load and display mnist info
    data.load_mnist()
    data.display_info()