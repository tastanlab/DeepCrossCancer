# -*- coding: utf-8 -*-
"""
You can change the inputs (parameters, file locations..) in this file.

@author: Duygu Ay
"""

from __future__ import division
import os
from os.path import join as pjoin
import numpy as np
import sys
import tensorflow as tf


def home_out(path):
    local_dir = os.getcwd()
    return pjoin(local_dir, path)


NUM_GENES = 20533 #with age and gender
NUM_CLASSES = 9
NUM_NODES = [32, 16]

LEARNING_RATE_SUPERVISED = 0.0001
LEARNING_RATE = 0.05
NUM_SUPERVISED_BATCHES = 30
NUM_TRAIN_BATCHES =24
NUM_SUPERVISED_EPOCHS = 200
NUM_TRAIN_EPOCHS = 150
N_FOLD = 10

lambda_list = np.logspace(-2, -7, num=10, endpoint=True)
lambda_list = np.concatenate((lambda_list, [0.0]), axis=0)
LAMBDA_LIST = sorted(lambda_list)

ALPHA_BETA_LIST = np.linspace(0, 2, num=9).tolist()


ALPHA = [1.0,1.75,0.25,0.75,1.75,1.75,2.0]
BETA = 1.5
LAMBDA = 1.e-7
NUM_GROUPS = [10,20,30,40,50,70,100]
NUM_MODELS = 7 #We trained 7 model with different 7 numbers of clusters.
PERCENT = 0.01 #percentage of top ranked features with DeepSHAP
N = 10000
N_cnv = 1000

CROSS_CANCER_ID = 'TCGA.AC.A7VC.01' #cross-cancer patient id for the gene expression analysis.

DATA_DIR = '/cta/users/duyguay/cross-cancer-update/'
RESULTS_DIR = 'allcancers/results/'
PLOT_DIR = 'plots/'
TRAIN_FILE = 'train_cancers.csv'
TEST_FILE = 'test_cancers.csv'
TRAIN_UNNORM_FILE = 'train_cancers_unnorm.csv'
TEST_UNNORM_FILE = 'test_cancers_unnorm.csv'

RAND_DIR = 'rand_index/'
MUT_DATA_DIR = 'mut_data/'
CNV_DATA_DIR = 'cnv_data/'

flags = tf.app.flags
FLAGS = flags.FLAGS

# Autoencoder Architecture Specific Flags

flags.DEFINE_integer('hidden_layer_dim_2', NUM_NODES[1],
                     'Number of units in the final hidden layer.')

flags.DEFINE_integer('hidden_layer_dim_1', NUM_NODES[0],
                     'Number of units in the final hidden layer.')

flags.DEFINE_integer('num_classes', NUM_CLASSES,
                     'Number of prior known classes.')

flags.DEFINE_integer('dimension', NUM_GENES, 'Number units in input layers.')


flags.DEFINE_list('num_groups', NUM_GROUPS, 'Number of clusters list for over-clustering.')


# Constants

flags.DEFINE_float('learning_rate', LEARNING_RATE,
                   'Initial learning rate.')

flags.DEFINE_integer('n_fold', N_FOLD,
                   'Number of folds for cross-validation.')
                   
flags.DEFINE_float('learning_rate_supervised', LEARNING_RATE_SUPERVISED,
                   'Initial learning rate.')

flags.DEFINE_integer('num_pretrain', NUM_SUPERVISED_EPOCHS,
                     'Number of training steps for supervised training')
                     
flags.DEFINE_integer('num_train_epochs', NUM_TRAIN_EPOCHS,
                     'Number of training steps for supervised-unsupervised training')                     

flags.DEFINE_integer('batch_size_pretrain', NUM_SUPERVISED_BATCHES,
                     'Number of training steps in one epoch for supervised training')
                     
flags.DEFINE_integer('batch_size_train', NUM_TRAIN_BATCHES,
                     'Number of training steps in one epoch for supervised-unsupervised training')

flags.DEFINE_integer('num_models', NUM_MODELS,
                   'Number of models for over-clustering.')

flags.DEFINE_list('lambda_list', LAMBDA_LIST, 'sparsity loss coefficients to be optimized.')

flags.DEFINE_list('alpha_beta_list', ALPHA_BETA_LIST, 'loss coefficients to be optimized.')

flags.DEFINE_boolean('parameter_tune', False, 'Tune parameters or not.')

flags.DEFINE_boolean('visualize', False, 'visualize or not.')

#for cross-cancer patient analysis

flags.DEFINE_boolean('randomize', True, 'Choose random patients for cross-cancer analysis.')

flags.DEFINE_boolean('get_all_shaps', False, 'Run long time to out top important features of all patients with deepshap.')

flags.DEFINE_boolean('cross_cancer', True, 'The goal is to find cross-cancer patients or cancer subtypes.')

flags.DEFINE_float('percent', PERCENT,
                   'percentage of top ranked features with DeepSHAP.')

flags.DEFINE_integer('N', N,
                   'Number of permutations for randomize.')

flags.DEFINE_integer('N_CNV', N_cnv,
                   'Number of permutations for randomize of CNV.') #Since CNV analysis takes long time, the number of permutations are less.

flags.DEFINE_string('cross_cancer_id', CROSS_CANCER_ID,
                    'Cross-cancer patient id for the gene expression analysis.')

#optimal parameters
flags.DEFINE_list('alpha', ALPHA, 'K-means loss coefficients.')
flags.DEFINE_float('lambda1', LAMBDA, 'sparsity penalty.')
flags.DEFINE_float('beta', BETA, 'Cox loss coefficient.')


# Directories
flags.DEFINE_string('data_dir', home_out(DATA_DIR),
                    'Directory to put the training data.')

flags.DEFINE_string('train_dir', home_out(DATA_DIR + TRAIN_FILE),
                    'Train file location.')
                    
flags.DEFINE_string('test_dir', home_out(DATA_DIR + TEST_FILE),
                    'Test file location.')

flags.DEFINE_string('train_unnorm_dir', home_out(DATA_DIR + TRAIN_UNNORM_FILE),
                    'Unnormalized train file location.')
                    
flags.DEFINE_string('test_unnorm_dir', home_out(DATA_DIR + TEST_UNNORM_FILE),
                    'Unnormalized test file location.')
                    
flags.DEFINE_string('results_dir', home_out(DATA_DIR + RESULTS_DIR),
                    'Directory to put the results.')

flags.DEFINE_string('plot_dir', home_out(DATA_DIR + PLOT_DIR),
                    'Directory to put figures.')

flags.DEFINE_string('rand_dir', home_out(DATA_DIR + RAND_DIR),
                    'Directory to put indices of random patients for permutation test.')

flags.DEFINE_string('mut_data_dir', home_out(DATA_DIR + MUT_DATA_DIR),
                    'Directory to put mutation data.')

flags.DEFINE_string('cnv_data_dir', home_out(DATA_DIR + CNV_DATA_DIR),
                    'Directory to put CNV data.')
                
# Python
flags.DEFINE_string('python', sys.executable,
                    'Path to python executable')
 
def main(argv):
    print(FLAGS)


if __name__ == '__main__':
    tf.app.run() 