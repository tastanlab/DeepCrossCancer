"""
This file consists of functions for the algorithm of gene expression analysis for a cross-cancer patient.

@author: Duygu Ay
"""

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels.stats.multitest as smm


def gene_exp(cross_matrix, data_all, FLAGS):

    """
    This funtion returns adjusted p-values of gene expressions to infer
    the association between cross-cancer patients and patients similar to them.
    
    .... 
    cross_matrix: DataFrame of cross-cancer patients.
    data_all: All data of unnormed gene expression values with labels.
    ....
    
    Output
    d: DataFrame of genes with BH corrected p-values.
    fig: Plot of the most significant 15 genes in can1 and can2 patients according to q-values.
    """
    
    pairs = cross_matrix.loc[FLAGS.cross_cancer_id, "Patients Similar to"]
    pairs= ast.literal_eval(pairs)
    can1=cross_matrix.loc[FLAGS.cross_cancer_id, 'Cross-cancer Type'] 
    can2=cross_matrix.loc[FLAGS.cross_cancer_id, 'Cancer Type of Patients Similar to']


    #g = data_all.loc[FLAGS.cross_cancer_id,'gender'] #for age and gender subset
    #ag = data_all.loc[FLAGS.cross_cancer_id,'age']
    #df = data_all[(data_all['gender'] == g) & (data_all['age'] == ag)]

    df = data_all
    
    #calculation of Z1
    x = np.log2(df[df.index.isin([FLAGS.cross_cancer_id])].iloc[:,0:FLAGS.dimension] + 1)
    x_bar1 = np.log2(df[df['label'] == can1].iloc[:,0:FLAGS.dimension] + 1).mean(axis = 0) #mean
    S1 = np.log2(df[df['label'] == can1].iloc[:,0:FLAGS.dimension]+1).std(axis = 0) #std

    Z1 = (x-x_bar1) / S1 #Z-score of can1 patients

    #Calculation of Z2 
    l = df[df['label'] == can2] #similar patients to the cross-cancer patients
    x_bar2 = np.log2(l[l.index.isin(pairs)].iloc[:,0:FLAGS.dimension]+1).mean(axis = 0) 
    S2 = np.log2(l[l.index.isin(pairs)].iloc[:,0:FLAGS.dimension]+1).std(axis = 0)
 
    Z2 = (x-x_bar2) / S2 

    #deltaZ calculation
    delta_z = (abs(Z2)-abs(Z1)).T
    delta_z.sort_values(by=FLAGS.cross_cancer_id, inplace=True, ascending=True)

    Z1 = Z1.T.reindex(index= delta_z.index.values)
    Z2 = Z2.T.reindex(index= delta_z.index.values)

    delta_z = delta_z.rename(columns={FLAGS.cross_cancer_id: 'delta_z'}).dropna()
    Z1 = Z1.rename(columns={FLAGS.cross_cancer_id: 'Z1'})
    Z2 = Z2.rename(columns={FLAGS.cross_cancer_id: 'Z2'})

    #p-values from deltaZ
    z_scores = delta_z/2
    z_scores = z_scores.rename(columns={'delta_z': 'Z score'})
    c = z_scores.index.values
    p_values = pd.DataFrame(scipy.stats.norm.cdf(z_scores), columns =['Pvalue'], index=c) #one-sided

    d = pd.concat([Z2[Z2.index.isin(c.tolist())],Z1[Z1.index.isin(c.tolist())],delta_z,z_scores,p_values], axis =1).dropna()

    d.sort_values(by='Pvalue', inplace=True, ascending=True)
    
    #adjusted p_values
    rej, pval_corr = smm.multipletests(d['Pvalue'], alpha = 0.1, method = 'fdr_i')[:2]
    d['BH'] = pval_corr


    """Plot of the first significant 15 genes: gene expression figure"""
    if FLAGS.visualize == True:
    
        #take the first 15 genes
        genes = d.index[:15].tolist()
        genes.append('label')

        #formatting to dataframe for plotting
        S = df[(df['label'] == can1) | (df['label'] == can2)][genes]

        col = list(S.columns.values)
        col.remove('label')
        df_un = []
        for g in col:
            G=S[[g, 'label']]
            G['class'] = g.split('.')[0]  
            G.columns = ['exp', 'label', 'class']
            df_un.append(G)

        df = pd.concat(df_un)

        #log2 expression values
        df['exp'] = np.log2(df['exp'] + 1)
        
        # Draw Plot
        plt.figure(figsize=(10.65,6.9), dpi= 300)
        
        #Draw box
        sns.boxplot(x='class', y='exp', data=df, hue='label', showfliers=False, linewidth = 1,  hue_order=[can1, can2], palette = ['white', 'white'])

        #draw other patients
        pairs.append(FLAGS.cross_cancer_id)
        sns.stripplot(x='class', y='exp', data=df[~df.index.isin(pairs)], hue='label', palette = ['black', 'black'],  hue_order=[can1, can2], dodge=True, size=1.5)
        
        #color of patients similar to the cross-cancer patient
        pairs.remove(FLAGS.cross_cancer_id)
        sns.stripplot(x='class', y='exp', data=df[df.index.isin(pairs)], hue='label', palette = ['black', 'rebeccapurple'],  hue_order=[can1, can2], dodge=True, size = 2)
        
        #color of the cross-cancer patient
        ax = sns.stripplot(x='class', y='exp', data=df[df.index.isin([FLAGS.cross_cancer_id])], hue='label', palette = ['gold', 'black'],  hue_order=[can1, can2], dodge=True, size = 4)

        #legend decoration
        ax.set_ylim(-1, 20)
        indexes = [0,1,6,5,7]
        handles, _ = ax.get_legend_handles_labels()
        plt.legend([handles[x] for x in indexes], [str(can1[0].upper()) + ': ' +  str(can1) + ' patients',
                    str(can2[0].upper()) + ': ' +  str(can2) + ' patients', 'The cross-cancer patient',
                    str(can2) + ' patients similar to the patient', 'Other patients'],
                    bbox_to_anchor=(0.715, 0.995), loc=2, borderaxespad=0.)


        # position of vertical lines
        for i in range(len(df['class'].unique())-1):
            plt.vlines(i+.5, ax.get_ylim()[0], ax.get_ylim()[1], linestyles='solid', colors='gray', alpha=0.2)


        # Make labels on the bottom of each column and adjust the position of them.
        labels = [str(can1[0].upper()), str(can2[0].upper())]*15
        x = -0.2

        for label,i in zip(labels, range(30)):
            ax.text(x, -0.7, label.capitalize(),
                    ha='center', va='bottom')
            
            if (i % 2) == 0:
                x = x + 0.4
                
            else:
                x = x + 0.6
            
        plt.ylabel('Log2 expression level')
        plt.xlabel(None)
        plt.savefig(FLAGS.plot_dir + 'GeneExp_test_for' + str(FLAGS.cross_cancer_id) + '.pdf', bbox_inches='tight')

    #save the all results
    d.index=[i.split('.')[0] for i in d.index]
    d.to_csv('GenStats_test_for' + str(FLAGS.cross_cancer_id) + '.csv', index = True)