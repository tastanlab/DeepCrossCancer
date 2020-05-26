"""
This file contains the statistical analysis that outputs p-values of genes, mutations and cytobands of cross-cancer patients.

.....

Run in command line: python3 analysis.py

@author: Duygu Ay
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from gene_exp_test import *
from DeepShap import *
from permtest_mutation import *
from permtest_cnv import *
import os
import scipy.io as sio
from sklearn import cluster
from params import FLAGS

if __name__ == '__main__':
    
    tensorflow_gpu_switch()
       
    # Create folders
        
    if not os.path.exists(FLAGS.rand_dir):
        os.makedirs(FLAGS.rand_dir)  
    
    if not os.path.exists(FLAGS.plot_dir):
        os.makedirs(FLAGS.plot_dir)
        
    train = pd.read_csv(FLAGS.train_dir, index_col=0)
    test = pd.read_csv(FLAGS.test_dir, index_col=0)
    
    train_unnorm = pd.read_csv(FLAGS.train_unnorm_dir, index_col=0)
    test_unnorm = pd.read_csv(FLAGS.test_unnorm_dir, index_col=0)
    
    
    #take only primary samples
    train, test = primary_samples(train, test)
    train_unnorm, test_unnorm = primary_samples(train_unnorm, test_unnorm)
    
    data_all = pd.concat([train, test])
    
    """Gene expression analysis of a cross-cancer patients"""
    data_all_unnorm = pd.concat([train_unnorm, test_unnorm])
    
    cross_matrix = pd.read_csv(FLAGS.data_dir + 'cross_cancer_patients.csv', index_col=0)
    
    gene_exp(cross_matrix, data_all_unnorm, FLAGS)
    
    #Run DeepShap
    cross_can_type = cross_matrix['Cross-cancer Type'].values.tolist()   
    sub_index = data_all[data_all['label'].isin(cross_can_type)].index
    
    if FLAGS.get_all_shaps == True:
        shap_df = deepshap_top_feat(data_all, sub_index, True, FLAGS)
    else:
        shap_df = pd.read_csv(r'shaps_top_features.csv', index_col=1)
        
        
    cross_shap = cross_shap(data_all, cross_matrix, FLAGS)
    
    """permutation test for top important genes by DeepShap"""
    shap_p, rand_p = perm_test_main(cross_shap, shap_df, FLAGS)
    
    """permutation test for commonly mutated genes"""
    all_p = perm_test_mut_main(cross_matrix, rand_p, FLAGS)
    
    #adjusted p-values
    if len(all_p) != 0:
        adj_p_mut = BH_pvalue(all_p)
        adj_p_mut.to_csv('adjusted_pvalues_ofmutatedgenes.csv')
    
    else:
        print("The cross-cancer patients have not commonly mutated genes.")
    
    """permutation test for common cytobands"""
    events = ['amp', 'del']
    
    for i in events:
        all_p_c = perm_test_cyto_main(cross_matrix, i, data_all, rand_p, FLAGS)
        
        if len(all_p_c) != 0:
            adj_p_cnv = BH_pvalue(all_p_c)
            adj_p_cnv.to_csv('adjusted_pvalues_ofcytobands_' + str(i) + '.csv')
            
        else:
            print("The cross-cancer patients have not common cytobands in terms of " + str(i) + " events.")