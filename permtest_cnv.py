"""
This file consists of functions for the algorithm of percnvation test of cytobands for cross-cancer patients.

@author: Duygu Ay
"""

import numpy as np
import pandas as pd
import ast
import statsmodels.stats.multitest as smm

def prep_cytobands(cancer, event, data_all, FLAGS):

    """
    preprocessing of CNV data
    ....
    event: amplification or deletion
    """
    
    df = pd.read_csv(FLAGS.cnv_data_dir + str(cancer) + '_cnv.txt', delimiter='\t', index_col=0).T
    
    cyto_samp = []
    for row in df.index[2:]:
    
        d = df.loc[row,:]
        
        if event == 'amp':
            gens = d.where(d > 0).dropna().index.unique() #get genes that have an amplification event.
        else:
            gens = d.where(d < 0).dropna().index.unique()
        
        g = pd.DataFrame(index=gens)
        g['gene'] = gens
        g['patient'] = row
        
        cyt = df.loc[['Cytoband','Locus.ID'],gens.tolist()].T
        cytg = g.merge(cyt, left_index=True, right_index=True) #get cytoband and locus ids of these genes

        cyto_samp.append(cytg)
    
    df = pd.concat(cyto_samp)
    
    df.index = df['patient']
    
    #getting only patients that are included also in our data for randomizing
    sub = data_all[data_all['label']==cancer].index.tolist()
    df = df[df.index.isin(sub)]
    
    return df
    

def perm_test_cyto(o, c, main_cnv, can1_cnv, rand_patient_set, FLAGS):

    """
    This funtion returns p-value of common cytobands in cross-cancer patients as a result of percnvation test.
    
    ....
    
    o: The number of similar patients who share cytoband c with the cross-cancer patient.
    c: The common cytoband in the cross-cancer patient.
    main_cnv: Cytobands of the cross-cancer patient.
    can1_cnv: CNV data of patients in cancer type of the cross-cancer patient.
    rand_patient_set: Indices of random patients drawn in patients of cancer type of the cross-cancer patient.  
    ....
    
    Output
    p-value: p-value as a result of the percnvation test.
    """
    count=0
    for k in range(FLAGS.N_CNV):
    
        rand_ind = rand_patient_set.iloc[k,:]
        
        if len(rand_ind) == 1:
        
            pairs_df = can1_cnv[can1_cnv.index.isin([rand_ind[0]])]['Cytoband']
            common_cnv=list(set(main_cnv).intersection(set(np.unique(pairs_df))))
            
            if c in common_cnv:
                count+=1
                
                
        else:
            S_k = can1_cnv[can1_cnv.index.isin(rand_ind)]
            pairs_cnv = np.unique(S_k['Cytoband'])
            common_cnv=list(set(main_cnv).intersection(set(pairs_cnv)))
                 
            if c in common_cnv:
                patients_c = S_k[S_k['Cytoband']==c]
                o_k = len(np.unique(patients_c.index.values))
                
                if o_k >= o:
                    count+=1
    
    p_value = count/FLAGS.N_CNV  
    
    return p_value
    

def perm_test_cyto_main(cross_matrix, event, data_all, rand_p, FLAGS):

    """
    This funtion returns p-values of common cytobands between cross-cancer patients and patients similar to them as a result of percnvation test.
    
    .... 
    cross_matrix: DataFrame of cross-cancer patients.
    event: amplification or deletion event.
    rand_p: List of the indices of random patients drawn in patients of cancer type of the cross-cancer patient.  
    ....
    
    Output
    all_p: List of dataframes that include p-values of common cytobands in cross-cancer patients.
    """
    
    all_p = []
    
    for i,l in zip(cross_matrix.index, range(len(cross_matrix.index.values))):
    
        can1=cross_matrix.loc[i, 'Cross-cancer Type'] 
        can2=cross_matrix.loc[i, 'Cancer Type of Patients Similar to']
        pairs = cross_matrix.loc[i, "Patients Similar to"]
        pairs= ast.literal_eval(pairs)
        
        #preprocess of cnv data
        can1_cnv = prep_cytobands(can1, event, data_all, FLAGS)
        can2_cnv = prep_cytobands(can2, event, data_all, FLAGS)
        
        p_values = []
        obs_stat = []
        common_cnv = []
        
        if i in can1_cnv.index:
        
            main_cnv = np.unique(can1_cnv.loc[i,'Cytoband'])
            
            if len(pairs) == 1:
            
                if pairs[0] in can2_cnv.index:
                
                    pairs_df = can2_cnv[can2_cnv.index.isin([pairs[0]])]
                    common_cnv=list(set(main_cnv).intersection(set(np.unique(pairs_df['Cytoband']))))
                    
                    for j in common_cnv:
                    
                        sub=pairs_df[pairs_df['Cytoband']==j]
                        o = 1
                        p_value = perm_test_cyto(o, j, main_cnv, can1_cnv, rand_p[l], FLAGS)
                        p_values.append(p_value)
                        obs_stat.append(o)
                        
                        print('P-value of cytoband ' + str(j) + ' in patient ' + str(i) + ':', p_value)
            
            else:
                rem = list(set(pairs) - set(can2_cnv.index.unique()))
                
                for r in rem:
                    while r in pairs: pairs.remove(r)

                n = len(pairs)
                
                pairs_df = can2_cnv[can2_cnv.index.isin(pairs)][['Cytoband', 'patient']]
                pairs_df = pairs_df.drop_duplicates()
                gens = list(set(main_cnv).intersection(set(np.unique(pairs_df['Cytoband']))))
                
                for j in gens:
                
                    sub=pairs_df[pairs_df['Cytoband']==j]
                    pts = np.unique(sub['patient'])
                    
                    if len(pts) >= 0.7*n:
                    
                        common_cnv.append(j)
                        o = len(pts)
                        p_value = perm_test_cyto(o, j, main_cnv, can1_cnv, rand_p[l], FLAGS)
                        p_values.append(p_value)
                        obs_stat.append(o)
                        
                        print('P-value of cytoband ' + str(j) + ' in patient ' + str(i) + ':', p_value)

        p = pd.DataFrame({'Cytobands': common_cnv, 'NumOfPatientswithCytoband': obs_stat, 'p-value': p_values})
        if not p.empty:
            p['Cross-cancer Patient'] = i
            all_p.append(p)

    return all_p
