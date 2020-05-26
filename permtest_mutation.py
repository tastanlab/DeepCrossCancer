"""
This file consists of functions for the algorithm of permutation test of mutated genes for cross-cancer patients.

@author: Duygu Ay
"""

import numpy as np
import pandas as pd
import ast
import statsmodels.stats.multitest as smm

def prep_mut(cancer, FLAGS):

    """preprocessing of mutation data"""
    
    df = pd.read_csv(FLAGS.mut_data_dir + str(cancer) +'_mutations.txt', sep = "\t")
    
    f = lambda x:'-'.join(x for x in x.split('-')[:4])[:-1]
    df.Tumor_Sample_Barcode = df.Tumor_Sample_Barcode.map(f)
    df.Tumor_Sample_Barcode  =pd.Series(df.Tumor_Sample_Barcode).str.replace("-", ".")
    
    f2 = lambda x: True if x[-2:] == '01' or x[-2:] == '03' else False
    df.index = df.Tumor_Sample_Barcode
    df = df[df.index.map(f2)]
    
    return df
 
 
def perm_test_mut_pvalue(o, c, main_mut, can1_mut, rand_patient_set, FLAGS):

    """
    This funtion returns p-value of common mutated genes in cross-cancer patients as a result of permutation test.
    
    ....
    
    o: The number of similar patients who share mutated gene c with the cross-cancer patient.
    c: The common mutated gene in the cross-cancer patient.
    main_mut: Mutated genes of the cross-cancer patient.
    can1_mut: Mutation data of patients in cancer type of the cross-cancer patient.
    rand_patient_set: Indices of random patients drawn in patients of cancer type of the cross-cancer patient.  
    ....
    
    Output
    p-value: p-value as a result of the permutation test.
    """
    
    count=0
    for k in range(FLAGS.N): #number of permutations
    
        rand_ind = rand_patient_set.iloc[k,:]
        S_k = can1_mut[can1_mut.index.isin(rand_ind)]
        pairs_mut = np.unique(S_k['Hugo_Symbol'])
        common_mut=list(set(main_mut).intersection(set(pairs_mut)))
             
        if c in common_mut:
        
            patients_c = S_k[S_k['Hugo_Symbol']==c]
            o_k = len(np.unique(patients_c.index.values))
            
            if o_k >= o:
                count+=1
    
    p_value = count/FLAGS.N 
    
    return p_value

    
def perm_test_mut_main(cross_matrix, rand_p, FLAGS):

    """
    This funtion returns p-values of common mutated genes between cross-cancer patients and patients similar to them as a result of permutation test.
    
    .... 
    cross_matrix: DataFrame of cross-cancer patients.
    rand_p: List of the indices of random patients drawn in patients of cancer type of the cross-cancer patient.  
    ....
    
    Output
    all_p: List of dataframes that include p-values of common mutated genes in cross-cancer patients.
    """
    
    all_p = []
    for i,j in zip(cross_matrix.index, range(len(cross_matrix.index.values))):
    
        can1=cross_matrix.loc[i, 'Cross-cancer Type'] 
        can2=cross_matrix.loc[i, 'Cancer Type of Patients Similar to']
        pairs = cross_matrix.loc[i, "Patients Similar to"]
        pairs= ast.literal_eval(pairs)
        
        #preprocess of mutation data
        can1_mut = prep_mut(can1, FLAGS)
        can2_mut = prep_mut(can2, FLAGS)
        
        p_values = []
        obs_stat = []
        common_mut = [] #list of common mutated genes
        
        if i in can1_mut.index:
        
            main_mut = np.unique(can1_mut.loc[i,'Hugo_Symbol']) #mutated genes of the cross-cancer patient
            
            pairs_df = can2_mut[can2_mut.index.isin(pairs)]
            pairs_mut = np.unique(pairs_df['Hugo_Symbol'])
            common_mut=list(set(main_mut).intersection(set(pairs_mut)))
            
            if len(common_mut) != 0:  
            
                for c in common_mut: #permutation test for every common mutated genes
                
                    patients_c = pairs_df[pairs_df['Hugo_Symbol']==c]
                    o = len(np.unique(patients_c.index))
                    
                    p_value = perm_test_mut_pvalue(o, c, main_mut, can1_mut, rand_p[j], FLAGS)
                    
                    p_values.append(p_value)
                    obs_stat.append(o)
                    
                    print('P-value of mutated gene ' + str(c) + ' in patient ' + str(i) + ':', p_value)
        
        p = pd.DataFrame({'genes': common_mut, 'NumOfPatientsMutated': obs_stat, 'p-value': p_values})
        if not p.empty:
            p['Cross-cancer Patient'] = i
            all_p.append(p)
        
    return all_p


def BH_pvalue(all_p):

    """
    This funtion returns adjusted p-values with BH correction of all common mutated genes or cytobands.
    
    .... 
    all_p: List of dataframes that include p-values of common mutated genes or cytobands for every cross-cancer patient.  
    ....

    """
    if len(all_p)==1:
        concatenated_df = all_p[0]
        
    else:
        concatenated_df = pd.concat(all_p, ignore_index=True)
    
    concatenated_df.sort_values(by='p-value', inplace=True, ascending=True)
    
    rej, pval_corr = smm.multipletests(concatenated_df['p-value'], alpha = 0.1, method = 'fdr_i')[:2]
    
    concatenated_df['BH'] = pval_corr
    
    return concatenated_df
    