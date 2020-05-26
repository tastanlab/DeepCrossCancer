# -*- coding: utf-8 -*-
"""
This file consists of functions for the algorithm of getting top feature with DeepSHAP.

@author: Duygu Ay
"""

import numpy as np
import pandas as pd
import shap
from utils import *
import ast
import functools
import gc
import random


def deepshap_top_feat(data_all, sub_index, to_csv, FLAGS):

    """
    Getting top important features of patients.
    
    ....
    data_all: all data
    sub_index: Indices of patients to be found their top important featues.
    to_csv: Save dataframe or not
    ....
    
    Output: DataFrame of cancer patients with top important features.
    """
    
    models = []
    
    #subset of similar patients
    sub = data_all[data_all.index.isin(sub_index)]
    
    #DeepShap to extract features
    for num_groups in FLAGS.num_groups:
    
        #import trained model
        model=import_keras_models(FLAGS, num_groups, 'train')
        
        #initialize js methods for visualization
        shap.initjs()
        
        clustering_part = Model(
            inputs=model.inputs,
            outputs=model.outputs[2],  # specifying a single output for shap usage
        )
        
        # create an instance of the DeepSHAP which is called DeepExplainer
        explainer_shap = shap.DeepExplainer(model=clustering_part,
                                         data=data_all.iloc[:,0:FLAGS.dimension])
        
        
        # Fit the explainer on a subset of the data (you can try all but then gets slower)
        shap_values = explainer_shap.shap_values(X=sub.iloc[:,0:FLAGS.dimension].values,
                                              ranked_outputs=True)
        
        features = []
        #get top %1 pencentile features for each index
        for i in range(sub.shape[0]):
            abso = np.absolute(shap_values[0][0][i])
            ind = abso.argsort()[-round(FLAGS.dimension*FLAGS.percent):][::-1]
            feat = sub.columns.values[ind]
            features.append(feat)
        
        models.append(features)
      
        gc.collect()

    inter_features = []
    
    #get intersection of top features of models
    for i in range(sub.shape[0]):
    
        intsec = list(functools.reduce(set.intersection, [set(item[i]) for item in models]))
        inter_features.append(intsec)


    shap_df = pd.DataFrame(list(dict(zip(sub.index.values,inter_features)).items()),
                 columns=['patient','shaps'])
    
    shap_df['label'] = sub['label'].values
    
    if to_csv == True:
        shap_df.to_csv('shaps_top_features.csv', index=True)
    
    return shap_df


def cross_shap(data_all, cross_matrix, FLAGS):

    """
    Getting top important features between cross-cancer patients and patients similar to them.
    
    ....
    shap_df: Top important features of similar patients across cancers.
    cross_matrix: Cross-cancer patients
    ....
    
    Output: DataFrame of cross-cancer patients with top important features.
    """
    pairs=list(set([ast.literal_eval(str(a)) for b in cross_matrix['Patients Similar to'].str.strip('[]').str.split(', ') for a in b]))
    cross_index = list(set(cross_matrix.index.values) | set(pairs))
    shap_df = deepshap_top_feat(data_all, cross_index, False, FLAGS)
    
    #get intersection of top features of patients similar to a cross-cancer patient
    inter_pair=[]
    for i in cross_matrix.index:
    
        pair_index = cross_matrix.loc[i, 'Patients Similar to']
        pair_index = ast.literal_eval(pair_index)
        
        
        f = lambda x: ast.literal_eval(str(x))
        
        if len(pair_index)==1:
            common_items_pair = shap_df[shap_df['patient'].isin(pair_index)]['shaps']
            common_items_pair = common_items_pair.apply(f).loc[common_items_pair.index[0],]
        
        else:
            pair_shap = shap_df[shap_df['patient'].isin(pair_index)]['shaps']
            pair_shap = pair_shap.apply(f)
            common_items_pair = list(set.intersection(*map(set, pair_shap)))
        
        inter_pair.append(common_items_pair)
    
    cross_matrix['cross-cancer patients'] = cross_matrix.index
    cross_shap = cross_matrix.merge(shap_df, left_on='cross-cancer patients', right_on = 'patient', how = 'left')
    del cross_shap['patient']
    
    cross_shap['Shaps of Patients Similar to'] = inter_pair
    
    #get common features between a cross-cancer patient and patients similar to
    cross_shap['Common Features'] = [(set.intersection(*[set(i), set(j)])) for i, j in zip(cross_shap['shaps'], cross_shap['Shaps of Patients Similar to'])]
    cross_shap = cross_shap.set_index('cross-cancer patients')
    cross_shap.to_csv('cross_shaps.csv', index=True)

    return cross_shap


def remove_age_gender(o_k):

    #removing age and gender features that are revealed between important genes.
    age = 'age'
    gender = 'gender'
    while age in o_k: o_k.remove(age)
    while gender in o_k: o_k.remove(gender)
   
   
def perm_test_pvalue(o, n, p_genes, can1_shap, rand_can1, i, can1, FLAGS):

    """
    This funtion returns p-value of cross-cancer patients as a result of permutation test to the outputs DeepSHAP.
    
    ....
    
    o: The number of common genes found by DeepSHAP between a cross-cancer patient and patients similar to.
    n: The number of similar patients to the cross-cancer patient.
    p_genes: Important genes of cross-cancer patient.
    can1_shap: Important genes, found by DeepSHAP, of patients in cancer type of the cross-cancer patient.
    rand_can1: Indices of random patients drawn in patients of cancer type of the cross-cancer patient.  
    can1: Cross-cancer type.
    i: Index of the cross-cancer patient.
    ....
    
    Output
    p-value: p-value as a result of the permutation test.
    """
    
    count=0
    rand_patients = []
    
    for k in range(FLAGS.N):
    
        if FLAGS.randomize == True:
            rand_ind = random.sample(np.unique(can1_shap.index.values).tolist(),n)
            rand_patients.append(rand_ind)
        
        else:
            rand_ind = rand_can1.iloc[k,:].values
        
        if len(rand_ind) != 1:
            pair_shap = can1_shap[can1_shap.index.isin(rand_ind)]['shaps']
            pair_shap = pair_shap.apply(lambda x: ast.literal_eval(str(x)))
            f = set.intersection(*map(set, pair_shap))
            o_k = list(set(p_genes).intersection(f))
        
        else:
            pair_shap = can1_shap.loc[rand_ind, :]
            pair_shap = pd.Series(pair_shap.iloc[0,:])
            pair_shap = pair_shap.loc['shaps']
            o_k = list(set(p_genes).intersection(set(ast.literal_eval(pair_shap))))
        
        remove_age_gender(o_k)
        
        o_k = len(o_k)
        print(o_k)
        if o_k >= o:
            count+=1
    
    if len(rand_patients) != 0:
    
        rand_patients = pd.DataFrame(rand_patients)
        rand_patients.to_csv(FLAGS.rand_dir + 'rand_' + str(can1)+ '_for_'+str(i)+'.csv')
    
    p_value = count/FLAGS.N   
    
    return p_value, rand_patients
    
    
def perm_test_main(cross_shaps, shap_df, FLAGS):

    """
    This funtion returns all p-values of cross-cancer patients as a result of permutation test to the outputs DeepSHAP.
    
    .... 
    cross_shaps: DataFrame of cross-cancer patients with common top important features.
    shap_df: DataFrame of all patients with top important features.
    ....
    
    Output
    shap_p: DataFrame of all p-values with cross-cancer patients indices.
    """

    p_values = []
    common_muts = []
    rand_p = []
    
    for i in cross_shaps.index:
    
        can1=cross_shaps.loc[i, 'Cross-cancer Type']
        
        rand_can1 = None
        
        can1_shap = shap_df[shap_df['label']==can1]
        
        if FLAGS.randomize == False:
            rand_can1 =  pd.read_csv(FLAGS.rand_dir + "rand_" + str(can1) + "_for_"+str(i)+".csv", index_col=0)
        
        
        pvalue = "p"
        rand_patients = "r"
        
        main_mut = cross_shaps.loc[i,'shaps']
        
        n = cross_shaps.loc[i,'Number of Patients Similar to']
    
        common_mut = cross_shaps.loc[i,'Common Features']
        #common_mut = ast.literal_eval(common_mut)
        
        remove_age_gender(common_mut)
        o = len(common_mut)
        print(o)
        
        if len(common_mut) != 0:
            
            pvalue, rand_patients = perm_test_pvalue(o, n, main_mut, can1_shap, rand_can1, can1, i, FLAGS)
            print('patients '+ str(i)+' p-value for shaps', pvalue)
    
        p_values.append(pvalue)
        common_muts.append(common_mut)
        rand_p.append(rand_patients)
    
    shap_p = pd.DataFrame(list(dict(zip(cross_shaps.index.values, common_muts)).items()),
             columns=['patient','genes'])
    shap_p['p_value'] = p_values
    shap_p.to_csv("shap_pvalue.csv")
    
    print('Permutation test for top genes found by DeepSHAP is done.')
    
    return shap_p, rand_p