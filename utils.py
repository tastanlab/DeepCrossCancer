# -*- coding: utf-8 -*-
"""
This file contains the necessary functions for training the model.

@author: Duygu Ay
"""
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from keras.models import Model, model_from_json
from keras.layers import Dense, Input
from keras import regularizers, metrics, optimizers 
import operator
import scipy.io as sio
import os
from sklearn import cluster
from sklearn.metrics import silhouette_score

"""for visualization"""
import random
from sklearn.manifold import TSNE #T-Distributed Stochastic Neighbor Embedding
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def tensorflow_gpu_switch():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)
    
    
def primary_samples(train, test):
    """taking only primary solid tumors"""
    
    f = lambda x: True if x[-2:] == '01' or x[-2:] == '03' else False
    test = test[test.index.map(f)]
    train = train[train.index.map(f)]
    
    return train, test


def split_labels(train, test, FLAGS):
    """Since we have two label type, we split cancer type and survival time labels."""
    
    X_train = train.iloc[:,0:FLAGS.dimension].values
    Y_tr_gen = train.iloc[:,FLAGS.dimension].values
    X_test = test.iloc[:,0:FLAGS.dimension].values
    Y_te_gen = test.iloc[:,FLAGS.dimension].values
    Y_tr_cl = train[['survival', 'vital_status']].values
    Y_te_cl = test[['survival', 'vital_status']].values
    
    cats = np.unique(train.label.values)
    di = dict(zip(cats,np.arange(len(cats))))
    
    Y_train = to_categorical(train.label.map(di))
    Y_test = to_categorical(test.label.map(di))
    
    return X_train, Y_train, Y_tr_gen, Y_tr_cl, X_test, Y_test, Y_te_cl, di


def loss(beta):
    """Cox regression loss with parameter beta"""
    
    def cox_loss(y_true, y_pred):
        #cox regression computes the risk score, we want the opposite
        score = -y_pred
    	#find index i satisfying event[i]==1
        ix = tf.where(tf.cast(y_true[:, 1], tf.bool)) # shape of ix is [None, 1]
    	#sel_mat is a matrix where sel_mat[i,j]==1 where time[i]<=time[j]
        sel_mat = tf.cast(tf.gather(y_true[:, 0], ix) <= y_true[:, 0], tf.float32)
    	#formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time[i]<=time[j] and event[i]==1
        p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum( sel_mat * tf.transpose(tf.exp(score)), axis=-1))
        loss = -tf.reduce_mean(p_lik)
        return beta * loss
    return cox_loss



def k_loss(alpha):
    """K-means loss with parameter alpha"""

    def custom_loss(y_true, y_pred):
        mse = K.mean(K.square(y_pred - y_true), axis=-1)
        return alpha*mse
        
    return custom_loss
  

def concordance_index(y_true, y_pred):

    """Calculate concordance index with survival times"""
    
    ## find index pairs (i,j) satisfying time[i]<time[j] and event[i]==1
    ix = tf.where(tf.logical_and(tf.expand_dims(y_true[:, 0], axis=-1)<y_true[:, 0], tf.expand_dims(tf.cast(y_true[:, 1], tf.bool), axis=-1)), name='ix')
    ## count how many score[i]<score[j]
    s1 = tf.gather(y_pred, ix[:,0])
    s2 = tf.gather(y_pred, ix[:,1])
    ci = tf.reduce_mean(tf.cast(s1<s2, tf.float32), name='c_index')
    
    return ci
    

def compute_silhouette(kmeans, X):
    
    """Compute silhouette score with the result of clustering"""
    
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouetteScore = silhouette_score(X, labels, metric='euclidean')
    return silhouetteScore
    
    
def cross_validation_param_selection(error_mean, error_std, FLAGS):

    """find the optimal parameters with one standard error rule"""
    
    error_min = min(error_mean)
    _,id_min = min( (error_mean[i],i) for i in range(len(error_mean)) )
    std_min = error_std[id_min]
    id_selected = (error_mean <= error_min + std_min).nonzero()
    id_selected = id_selected[0]
    para_selected = [FLAGS.alpha_beta_list[i] for i in id_selected]
    para_optimal = max(para_selected)
    
    return para_optimal
    
    
    
def create_M(X_train, num_groups, model):

    """Create cluster labels from encoded set"""
    
    # Extract the encoder
    encoder = K.function([model.layers[0].input], [model.layers[2].output])
    # Encode the training set
    encoded_set = encoder([X_train])[0]

    kmeans_all = cluster.KMeans(n_clusters=num_groups, init = "k-means++", max_iter=50, tol=0.01).fit(encoded_set)
    ass_all = kmeans_all.labels_
    centers_all = kmeans_all.cluster_centers_
    A_all = np.zeros([X_train.shape[0], num_groups])
    for i_sample in range(0, X_train.shape[0]):
        A_all[i_sample, ass_all[i_sample]] = 1
    total_M = np.dot(A_all, centers_all)
    
    return centers_all, total_M
    
  


def create_model(lambda_trained, beta_trained, alpha_trained, model_type, FLAGS):

    """pretraining and training models"""
    
    input_layer = Input(shape = (FLAGS.dimension, ))
    layer0 = Dense(FLAGS.hidden_layer_dim_1, activation='relu', kernel_regularizer=regularizers.l1(lambda_trained))(input_layer)
    layer1= Dense(FLAGS.hidden_layer_dim_2, activation='relu', name = 'layer1')(layer0)
    y_pred = Dense(FLAGS.num_classes, activation='softmax', name = 'y_pred')(layer1)
    event_out = Dense(1, activation='sigmoid', name = 'event_out')(layer1)
    
    if model_type == 'pretrain':
    
        model= Model(inputs=input_layer, outputs=[y_pred, event_out])
        model.compile(optimizer= optimizers.Adam(lr=FLAGS.learning_rate_supervised), loss={'y_pred': 'categorical_crossentropy', 'event_out': loss(beta_trained)}, metrics=['acc', concordance_index])

    else:
    
        model = Model(inputs=input_layer, outputs=[y_pred, event_out, layer1])
        model.compile(optimizer=optimizers.SGD(lr=FLAGS.learning_rate), loss={'y_pred': 'categorical_crossentropy', 'event_out': loss(beta_trained), 'layer1' : k_loss(alpha_trained)}, metrics=['acc', concordance_index, metrics.categorical_crossentropy])

    return model



def save_keras_models(model, FLAGS, num_groups, model_type):

    """save keras models"""
    
    path_model = FLAGS.results_dir + '/model/'
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    model_json = model.to_json()
    
    if model_type =='pretrain':
        with open(path_model + "pretained_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(path_model + "pretrained_model.h5")
    else:
        with open(path_model + model_type + "_model_" + str(num_groups) + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(path_model + model_type + "_model_"  + str(num_groups) + ".h5")
    print("Saved model to disk")

def import_keras_models(FLAGS, num_groups, model_type):

    """import keras models from results_dir"""
    
    trained_model_path = FLAGS.results_dir + '/model/'
    
    if model_type =='pretrain':
        json_file = open(trained_model_path + 'pretrained_model.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(pretrained_model_path + 'pretrained_model.h5')
        
    else:
        json_file = open(trained_model_path + model_type + '_model_' + str(num_groups) + '.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(trained_model_path + model_type + '_model_' + str(num_groups) + '.h5')
    
    print("Loaded model from disk")
    
    return model



def import_opt_parameters(FLAGS):

    """import optimal lambda and beta parameters after parameter tuning"""
    
    #import optimal lambda from pretrained model
    pretrained_file = FLAGS.results_dir + 'trainParams.mat'
    content = sio.loadmat(pretrained_file)
    lambda_trained =content['lambda_optimal']
    lambda_trained = lambda_trained[0][0]
    print('lambda_trained: ', lambda_trained)

    #import optimal beta from pretrained model
    beta_trained =content['beta_optimal']
    beta_trained = beta_trained[0][0]
    print('beta_trained: ', beta_trained)
    
    return lambda_trained, beta_trained
    
    
def training_summary(history, model, X_test, Y_test, Y_te_cl, total_M, model_type, num_groups, FLAGS):
   
   """summarize history for losses and get test scores"""
   
   plt.figure(figsize=(5,4), dpi= 300)
   plt.plot(history.history['loss'])
   plt.plot(history.history['y_pred_loss'])
   plt.plot(history.history['event_out_loss'])

   plt.ylabel('loss')
   plt.xlabel('epoch')
    
   if model_type == 'train':
        plt.plot(history.history['layer1_loss'])
        plt.title('model loss for ' + str(num_groups) + ' cluster')
        plt.legend(['total', 'cross-entropy', 'cox', 'kmeans'], loc='upper left')
        plt.savefig(FLAGS.plot_dir + 'train_loss' + str(num_groups) + '.png')

        score = model.evaluate(X_test, [Y_test, Y_te_cl, total_M], verbose=2)
        print(score)
        print("Test C-index for the number of clusters " + str(num_groups) +": %.2f%%" % (score[8]*100))
        print("Test Accuracy for the number of clusters " + str(num_groups) +": %.2f%%" % (score[4]*100))
    
   else:
        plt.title('model loss')
        plt.legend(['total', 'cross-entropy','cox'], loc='upper left')
        plt.savefig(FLAGS.plot_dir + 'pretrain_loss.png')
        
        score = model.evaluate(X_test, [Y_test, Y_te_cl], verbose=2)
        print("Test C-index for pretrain: %.2f%%" % (score[6]*100))
        print("Test Accuracy for pretrain: %.2f%%" % (score[3]*100))
 


def select_pre_k(k_list, model, X_train):

    """Find pre-optimal k by maximizing silhouette score from pretrained model"""

    # Extract the encoder
    encoder = K.function([model.layers[0].input], [model.layers[2].output])
    # Encode the training set
    encoded_set = encoder([X_train])[0]
    KMeans = [cluster.KMeans(n_clusters = k, init="k-means++").fit(encoded_set) for k in k_list]
    s = [compute_silhouette(kmeansi,encoded_set) for kmeansi in KMeans]
    index, value = max(enumerate(s), key=operator.itemgetter(1))
    kmeans_pretrained = KMeans[index]
    num_groups_initial= k_list[index]

    print('num_groups_initial: ', num_groups_initial) 
    
    return num_groups_initial
    
    
    
def tsne(h, labels, num_clusters, FLAGS):

    """
    h: Hidden output of the model with patient indices
    labels: Grand truth labels of patients
    num_clusters: Number of clusters
    mapping: String mapping of int labels
    ....
    output: T-SNE figure with perplexity 40
    """
    
    #plotX will hold the values we wish to plot
    h["Cluster"] = labels
    plotX = pd.DataFrame(np.array(h), index = h.index)
    plotX.columns = h.columns
    
    #Set our perplexity
    perplexity = 40
    
    #T-SNE with two dimensions
    tsne_2d = TSNE(n_components=2, perplexity=perplexity)
    
    #This DataFrame contains two dimensions, built by T-SNE
    TCs_2d = pd.DataFrame(tsne_2d.fit_transform(plotX.drop(["Cluster"], axis=1)), index = h.index)
    
    #And "TC2_2d" means: 'The second component of the components created for 2-D visualization, by T-SNE.'
    TCs_2d.columns = ["TC1_2d","TC2_2d"]
    plotX = pd.concat([plotX,TCs_2d], axis=1, join='inner')  

    #This is needed so we can display plotly plots properly
    init_notebook_mode(connected=True)
    
    label_names = np.unique(labels).tolist()
    num_class = len(label_names)
    cl = [0]*(num_class)
    for i in range(num_class):
      cl[i] = plotX[plotX["Cluster"] == label_names[i]]
    
    random.seed(30)
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(num_class)]
    
    trace = [0]*(num_class)
    for i in range(num_class):
      trace[i] = go.Scatter(x = cl[i]["TC1_2d"],
                            y = cl[i]["TC2_2d"],
                            mode = "markers",
                            name = label_names[i],
                            marker = dict(size=4, color = color[i]),
                            text = cl[i].index)

    title = "Visualizing " + str(num_clusters)+" Clusters in Two Dimensions Using T-SNE (perplexity=" + str(perplexity) + ")"
    
    layout = dict(title = title,
                  xaxis= dict(title= 'TC1',ticklen= 5,zeroline= False, titlefont=dict(family="Arial,sans-serif", size=12, color="black")),
                  yaxis= dict(title= 'TC2',ticklen= 5,zeroline= False, titlefont=dict(family="Arial,sans-serif", size=12, color="black")),
                  height=600,
                  font=dict(family="Arial, sans-serif",size=12),
                  legend=go.layout.Legend(x=1, y=0.9, traceorder="normal",itemsizing='constant',font=dict(family="Arial,sans-serif", size=11, color="black")))
    
    fig = dict(data = trace, layout = layout)
    plot(fig, image='svg', filename=FLAGS.plot_dir + 'tsne_' + str(num_clusters) + '.svg', image_width=500, image_height=500)