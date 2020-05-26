# -*- coding: utf-8 -*-
"""
This file contains the necessary functions for creating cross-cancer patients.

@author: Duygu Ay
"""
import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics import silhouette_samples
import seaborn as sns



def bestMap(L1,L2):

    """ maps cluster prediction labels to the grand truth labels.
    
    L1: Array of grand truth labels
    L2: Array of cluster predictions
    .....
    
    Output: Map of cluster predictions
    """

    if not(len(L1)==len(L2)):
        print("Error! L1 size and L2 size not equal")

    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass,nClass))

    for i in range(nClass1):
        for j in range(nClass2):
            G[i,j] = len(np.where(L1[np.where(L2 == Label2[j])] ==Label1[i])[0])
    
    c,t = scipy.optimize.linear_sum_assignment(-G.T)
    newL2 = np.zeros(len(L2))
    for i in range(nClass2):
      idx = np.where(L2 == Label2[i])
      if t[i] >= 9:
        newL2[idx] = t[i]
      else:
        newL2[idx] = Label1[t[i]]
    return np.asarray(newL2,int)
    
    
def create_patient_matrix(Y_tr, Y_te, all_case, FLAGS):

    """ create patient by patient matrix that consists of pairwise similarity scores.
    
    Y_tr: Cancer type labels of train data
    Y_te: Cancer type labels of test data
    all_case: Mapping of cluster predictions for all numbers of clusters.
    .....
    
    Output: DataFrame of patient by patient matrix 
    """


    k= pd.concat([Y_tr, Y_te])  
    sample_class = pd.DataFrame(np.column_stack((all_case)), index = k.index, columns = FLAGS.num_groups)

    #patient by patient matrix
    test_val = sample_class.values

    # Base matrix that will be added to.
    curr_mat = np.zeros((test_val.shape[0], test_val.shape[0]))

    # Max index of matrix components (i.e. max_val + 1 is number of clusters/matrix components)
    max_val = np.max(test_val)

    for n_clus in range(max_val + 1):
        # Extract component matrix corresponding to current iteration.
        clus_mem = (test_val == n_clus).astype(int)
        curr_mat += np.dot(clus_mem, clus_mem.T)

    curr_mat /= len(FLAGS.num_groups)
    patients_matrix = pd.DataFrame(curr_mat, index=sample_class.index, columns=sample_class.index)
    patients_matrix['label'] = k
    patients_matrix = patients_matrix.sort_values(by=patients_matrix.columns[-1])
    df = patients_matrix.loc[:,~patients_matrix.columns.duplicated()]
    df2 = df.loc[~df.index.duplicated(keep='first')]
    temp = df2.iloc[:, :-1]
    i = temp.index.tolist()
    patient_matrix = temp[i]
    patient_matrix['label'] = df2[['label']]
    patient_matrix.to_csv('patients_matrix.csv', index=True)
    
    return patient_matrix
    

def similarity_map_fig(patient_matrix, FLAGS):

    """ patient similarity map figure.
    
    patient_matrix: DataFrame of pairwise similarity matrix of all patients and last column consist of labels
    """

    df = patient_matrix.set_index([patient_matrix.index, patient_matrix.columns[-1]])
    df.index.names = ['index', ""]
    labels = df.index.get_level_values("")
    
    network_pal = sns.cubehelix_palette(labels.unique().size,
                                        light=.9, dark=.1, reverse = True,
                                        start=1, rot=-2)
                                        
    network_lut = dict(zip(map(int, np.unique(labels).tolist()), network_pal))

    network_colors = pd.Series(labels, index=patient_matrix.columns[:-1]).map(network_lut)

    g = sns.clustermap(patient_matrix.iloc[:, :-1],

                      # Turn off the clustering
                      row_cluster=False, col_cluster=False,

                      # Add colored class labels
                      row_colors=network_colors, col_colors=network_colors,

                      # Make the plot look better when many rows/cols
                      linewidths=0, xticklabels=False, yticklabels=False,cmap='Reds')

    #Decoration
    for label in labels.unique():
        g.ax_col_dendrogram.bar(0, 0, color=network_lut[label],
                                label=label, linewidth=0)
    
    g.ax_col_dendrogram.legend(loc="center", ncol=5)
    g.cax.set_position([.97, .2, .03, .45])
    g.fig.suptitle('Patients Similarity Map', x = 0.6, y= 0.88)
    ax = g.ax_heatmap
    ax.set_xlabel('Patients')
    ax.set_ylabel('Patients')

    g.savefig(FLAGS.plot_dir + "patients_map.png", dpi=300)   
    
    
def extract_sim_samples_across_cancers(patient_matrix, data_all):

    """ extracts pairwise patients that have similarity score of 1 from different types of cancer.
    
    patient_matrix: DataFrame of patient by patient matrix that consists of pairwise similarity scores.
    data_all: all data
    .....
    
    Output: 
    DataFrame of similar patients across cancers with their cancer types.
    Index of these patients
    Index numbers of these patients
    """
    
    rowcols = np.nonzero(np.triu(patient_matrix.iloc[:, :-1], k=1))
    cross = list()
    
    for row, col in zip(rowcols[0], rowcols[1]):
       if (patient_matrix.iloc[row,-1] != patient_matrix.iloc[col,-1]) & (patient_matrix.iloc[row,col] == 1):
          cross.append([patient_matrix.index[row],patient_matrix.index[col], patient_matrix.iloc[row,-1], patient_matrix.iloc[col,-1]])

    cross_matrix = pd.DataFrame(cross, columns= ['First Patient', 'Second Patient', 'First Patient Cancer', 'Second Patient Cancer'])

    cross_matrix.to_csv('similar_patients_across_cancers.csv', index=True)
    
    cross_index = list(set(cross_matrix['First Patient']) | set(cross_matrix['Second Patient']))

    a=data_all.index.isin(cross_index)
    cross_index_num = [i for i, x in enumerate(a) if x]
    
    return cross_matrix, cross_index, cross_index_num
    
    
def compute_silhouette_samp(kmeans, labels, X):

    """calculate silhouette coefficient of samples.
    
    X: Input
    labels: Grand truth labels
    """
    
    kmeans.fit(X)
    silhouetteScore = silhouette_samples(X, labels, metric='euclidean')
    
    return silhouetteScore
    
    
def extract_cross_cancer_samp(s_k, cross_matrix, cross_index, data_all, FLAGS):

    """calculate silhouette coefficient of similar samples across cancers. Label as cross-cancer patients who have negative silhouette coefficient.
    
    s_k: Silhouette coefficients of similar samples across cancers for all models
    cross_matrix: DataFrame of similar patients across cancers with their cancer types.
    .....
    
    Output: Dataframe of cross-cancer patients and similar patients to them and cancer types.
    """
    
    s=np.concatenate([s_k], axis = 1).T
    
    a = list(map(str,FLAGS.num_groups))
    a.append('label')
    
    cross_df = data_all[data_all.index.isin(cross_index)]
    cross_silh=pd.DataFrame(np.column_stack((s,cross_df.iloc[:,FLAGS.dimension].values)),index = cross_df.index, columns = a)

    
    silh_neg_index = cross_silh[cross_silh.iloc[:,0:FLAGS.num_models] < 0].iloc[:,0:FLAGS.num_models].dropna().index
    pairs = set(cross_index) - set(silh_neg_index)

    cr = cross_silh[cross_silh.index.isin(silh_neg_index)]


    sim = []
    n = []
    cancer = []

    for i in silh_neg_index:

        sub = cross_matrix[(cross_matrix['First Patient']==i) | (cross_matrix['Second Patient']==i)]
        p_all = list(set().union(sub['First Patient'].tolist(),sub['Second Patient'].tolist()))
        
        p = list(set(p_all) & set(pairs))
        
        if len(p) != 0:
            num = len(p)
            
            subsub=sub[(sub['First Patient']==p[0]) | (sub['Second Patient']==p[0])]
            
            if (subsub['First Patient']==p[0]).any():
                can = subsub['First Patient Cancer'].values
            else:
                can = subsub['Second Patient Cancer'].values
                
        else:
            num=1
            if (sub['First Patient']==i).any():
                can = sub['Second Patient Cancer'].values
                p = sub['Second Patient'].tolist()
            else:
                can = sub['First Patient Cancer'].values
                p = sub['First Patient'].tolist()
        
        sim.append(p)
        n.append(num)
        cancer.append(can[0])
        
    cr['Patients Similar to'] = sim
    cr['Cancer Type of Patients Similar to'] = cancer
    cr['Number of Patients Similar to'] = n
    cro=cr.drop(cr.columns[0:FLAGS.num_models], axis=1)
    cro.rename(columns={'label':'Cross-cancer Type'}, inplace=True)
    
    cro.to_csv('cross_cancer_patients.csv', index=True)
    
    return cro
    
        

