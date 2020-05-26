# -*- coding: utf-8 -*-
"""
This file contains the model that outputs cross-cancer patients or clustering results.

.....

Run in command line: python3 model.py

@author: Duygu Ay
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from cross_utils import *
from find_opt_lambda_beta_alpha import *
import os
import scipy.io as sio
from sklearn import cluster
from params import FLAGS

if __name__ == '__main__':
        
    tensorflow_gpu_switch()
    
    # Create folders
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)

    if not os.path.exists(FLAGS.plot_dir):
        os.makedirs(FLAGS.plot_dir)
        
    train = pd.read_csv(FLAGS.train_dir, index_col=0)
    test = pd.read_csv(FLAGS.test_dir, index_col=0)
    
    #take only primary samples
    train, test = primary_samples(train, test)
    
    #split labels from train test data
    X_train, Y_train, Y_tr_gen, Y_tr_cl, X_test, Y_test, Y_te_cl, di = split_labels(train, test, FLAGS)
    data_all = pd.concat([train, test])
    
    
    if FLAGS.parameter_tune == True:
    
        #tune lambda and beta parameters
        p = {'lambda1' : FLAGS.lambda_list}
        error_total, error_mean, error_std, cr_total, cr_mean, cr_std, lamb_pair_total, lamb_pair_mean, lamb_pair_std = lambda_beta_tuning(X_train, Y_train, Y_tr_gen, Y_tr_cl, p, FLAGS)
        
        #choose optimal beta and lambda
        beta_optimal = cross_validation_param_selection(error_mean, error_std, FLAGS)
        lambda_optimal = cross_validation_param_selection(lamb_pair_mean, lamb_pair_std, FLAGS)

        sio.savemat(FLAGS.results_dir + 'trainParams.mat', {'cr_total': cr_total, 'error_total': error_total, 'cr_mean': cr_mean,
                                              'cr_std': cr_std, 'error_mean': error_mean, 'error_std': error_std,
                                              'lambda_optimal': lambda_optimal, 'lambda_list': FLAGS.lambda_list, 'lamp_pair' : lamb_pair_total, 
                                              'lamb_pair_mean': lamb_pair_mean, 'beta_optimal': beta_optimal, 'alpha_beta_list': FLAGS.alpha_beta_list})
    else:
        lambda_optimal, beta_optimal = FLAGS.lambda1, FLAGS.beta
        
    #train pre_model with the best parameters
    pre_model = create_model(lambda_optimal, beta_optimal, None, 'pretrain', FLAGS)
    
    history = pre_model.fit(X_train, [Y_train, Y_tr_cl], epochs=FLAGS.num_pretrain, batch_size=FLAGS.batch_size_pretrain, verbose=2)
    
    save_keras_models(pre_model, FLAGS, None, 'pretrain')
    
    #print the performance of pretrained model
    training_summary(history, pre_model, X_test, Y_test, Y_te_cl, None, 'pretrain', None, FLAGS)
    
    
    #If the goal is to find cancer subtypes.
    if FLAGS.cross_cancer == False:
        num_groups_initial = select_pre_k(list(range(FLAGS.num_classes-2, FLAGS.num_classes+6)), pre_model, X_train)
        FLAGS.num_groups = list(range(num_groups_initial, num_groups_initial+7))
    else:
        None
    
    
    all_case = [list()] * len(FLAGS.num_groups) #store of mapping of cluster predictions
    s_k = [] #store of silhouette coefficients
    
    #find best alpha for all numbers of clusters 
    for i,j in zip(FLAGS.num_groups, range(len(FLAGS.num_groups))):
    
        #tune alpha with one-standard-error rule
        if FLAGS.parameter_tune == True:
        
            error_total, error_mean, error_std, cr_total, cr_mean, cr_std = alpha_tuning(X_train, Y_train, Y_tr_gen, Y_tr_cl, pre_model, i, FLAGS)
            alpha_optimal = cross_validation_param_selection(error_mean, error_std, FLAGS)
            
            sio.savemat(FLAGS.results_dir + "trainingAlpha_" + str(i) + ".mat", {'cr_total': cr_total, 'error_total': error_total, 'cr_mean': cr_mean, 'error_mean': error_mean, 
                                                                            'cr_std': cr_std, 'error_std':error_std, 'alpha_list': FLAGS.alpha_beta_list, 'alpha_optimal': alpha_optimal})
        else:
            alpha_optimal=FLAGS.alpha[j]
        
        #get S and C variables from the encoding layer of pretrained model
        centers, total_M = create_M(X_train, i, pre_model)
        
        #train the model with the best parameters
        model = create_model(lambda_optimal, beta_optimal, alpha_optimal, 'train', FLAGS)
        history = model.fit(X_train, [Y_train, Y_tr_cl, total_M], epochs=FLAGS.num_train_epochs, batch_size=FLAGS.batch_size_train, verbose=0)
        
        #clustering on the encoder layer of model with all data
        hidden_output = model.predict(data_all.iloc[:,:FLAGS.dimension], verbose=0)[2]
        kmeans = cluster.KMeans(n_clusters=i, init = "k-means++", max_iter=50, tol=0.01).fit(hidden_output)
        ass_total = kmeans.predict(hidden_output)
        
        save_keras_models(model, FLAGS, i, 'train')
        
        #compute silhouette score for this model
        s = compute_silhouette(kmeans, hidden_output)
        sio.savemat(FLAGS.results_dir + "encoder_" + str(i) + ".mat", {'silhouette': s, 'hidden_output': hidden_output,'centers': centers, 'ass_total': ass_total})
        
        #performance evaluation
        _, total_M_t = create_M(X_test, i, pre_model)
        training_summary(history, model, X_test, Y_test, Y_te_cl, total_M_t, 'train', i, FLAGS)
      
        
        if FLAGS.cross_cancer == True:
            """Cross-cancer patients"""
            
            #mapping of cluster predictions
            int_label = data_all.label.map(di)
            pf = bestMap(int_label.values, ass_total)
            all_case[j] = pf
            
            #compute silhouette coefficients of samples
            s_samp = compute_silhouette_samp(kmeans, data_all['label'], hidden_output)
            s_k.append(s_samp)
            
        else:
            s_k.append(s)
            
        #TSNE Visualization
        if FLAGS.visualize == True:
            ho = pd.DataFrame(hidden_output, index = data_all.index)
            tsne(ho, data_all['label'].values, i, FLAGS)
        
    
    
    
    if FLAGS.cross_cancer == True:
        #create patients by patients matrix
        patient_matrix = create_patient_matrix(train['label'], test['label'], all_case, FLAGS)
        
        #Similarity map figure of all patients
        if FLAGS.visualize == True:
            similarity_map_fig(patient_matrix, FLAGS)
        
        #extract similar samples across cancers whose similarity score of 1
        cross_matrix, cross_index, cross_index_num = extract_sim_samples_across_cancers(patient_matrix, data_all)
        
        #subset silhouette coefficients of similar patients with index numbers
        s=[l[cross_index_num] for l in s_k]
            
        cross_cancer_p = extract_cross_cancer_samp(s, cross_matrix, cross_index, data_all, FLAGS)
    
    else: 
        ind = np.argmax(s_k)
        num_cancer_sub = FLAGS.num_groups[ind]
        print('The optimal number of cancer subtypes: ', num_cancer_sub) 

            
        