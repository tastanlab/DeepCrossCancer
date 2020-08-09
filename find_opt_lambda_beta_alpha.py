"""In this file, we find optimal lambda (regularization parameter) and beta (trade-off parameter of cox loss) parameters """

from utils import *
from sklearn.model_selection import StratifiedKFold
import talos as ta
import numpy as np

def lambda_beta_tuning(X_train, Y_train, Y_tr_gen, Y_tr_cl, p, FLAGS):
    
    """beta_list: List of numbers to be optimized
       n_fold: Number of folds for cross-validation
       Y_tr_gen: Cancer type labels
       Y_tr_cl: Survival labels
    """      
    
    def talos_model(x_train, y_train, x_val, y_val, params):
    
        """Created a pretrain model with validation split for talos optimization tool. """
    
        input_layer = Input(shape = (FLAGS.dimension, ))
        layer0 = Dense(FLAGS.hidden_layer_dim_1, activation='relu', kernel_regularizer=regularizers.l1(params['lambda1']))(input_layer)
        layer1= Dense(FLAGS.hidden_layer_dim_2, activation='relu', name = 'layer1')(layer0)
        y_pred = Dense(FLAGS.num_classes, activation='softmax', name = 'y_pred')(layer1)
        event_out = Dense(1, activation='sigmoid', name = 'event_out')(layer1)
        model = Model(inputs=input_layer, outputs=[y_pred, event_out])
        model.compile(optimizer= optimizers.Adam(lr=FLAGS.learning_rate_supervised), loss={'y_pred': 'categorical_crossentropy', 'event_out': loss(beta_this)}, metrics=['acc', concordance_index, metrics.categorical_crossentropy])
        history = model.fit(x = x_train, y= [y_train[0], y_train[1]] , validation_data=[x_val, [y_val[0], y_val[1]]] , epochs=FLAGS.num_pretrain, batch_size=FLAGS.batch_size_pretrain, verbose = 0)
        return history, model
    
    
    #one-standard error rule parameters
    cr_total = list()
    error_total = list()
    cr_mean = list()
    error_mean = list()
    cr_std = list()
    error_std = list()
    lamb_pair_total = list()
    lamb_pair_mean = list()
    lamb_pair_std = list()

    #beta estimation with one standard error rule
    skf = StratifiedKFold(n_splits=FLAGS.n_fold)
    
    for beta_this in FLAGS.alpha_beta_list:
    
        cr_this = list()
        error_this = list()
        lamb_pair_this = list()

        #strafied cross validation
        indices = skf.split(X_train, Y_tr_gen)
        
        for train_index, val_index in indices:
            scan_object = ta.Scan(x = X_train[train_index], y = [Y_train[train_index], Y_tr_cl[train_index]], 
                                  x_val = X_train[val_index], y_val = [Y_train[val_index], Y_tr_cl[val_index]], params=p,
                                  model=talos_model, fraction_limit=0.1, experiment_name='cancer',
                                  reduction_metric='val_event_out_concordance_index', clear_session=False)
            
            sc =scan_object.data
            lamb = sc['lambda1']
            error = 1-sc['val_y_pred_acc']
            cr = sc['val_y_pred_categorical_crossentropy']
            
            # print the performance on validation set
            print('Beta = ' +str(beta_this) + ', Classification error on Validation set: %f \n' % (error))
            lamb_pair_this.append(lamb)
            cr_this.append(cr)
            error_this.append(error)
            
        lamb_pair_mean.append(np.mean(np.array(lamb_pair_this)))
        lamb_pair_std.append(np.std(np.array(lamb_pair_this)))
        lamb_pair_total.append(lamb_pair_this)
        cr_total.append(cr_this)
        error_total.append(error_this)
        cr_mean.append(np.mean(np.array(cr_this)))
        cr_std.append(np.std(np.array(cr_this)))
        error_mean.append(np.mean(np.array(error_this)))
        error_std.append(np.std(np.array(error_this)))
        
    return error_total, error_mean, error_std, cr_total, cr_mean, cr_std, lamb_pair_total, lamb_pair_mean, lamb_pair_std
    
    
   
    
def alpha_tuning(X_train, Y_train, Y_tr_gen, Y_tr_cl, pre_model, num_groups, FLAGS):

    
    cr_total = list()
    error_total = list()
    cr_mean = list()
    error_mean = list()
    cr_std = list()
    error_std = list()
    
    skf = StratifiedKFold(n_splits=FLAGS.n_fold)
    
    #import optimal lambda and beta found previously
    lambda_optimal, beta_optimal = import_opt_parameters(FLAGS)
    
    for alpha_this in FLAGS.alpha_beta_list:
    
        cr_this = list()
        error_this = list()
        
        #given each num_groups and alpha, perform 10-fold CR
        #strafied cross validation
        indices = skf.split(X_train, Y_tr_gen)
        
        for train_index, val_index in indices:
            # get initial Q and fix S and C
            
            _,total_M_t=create_M(X_train[train_index], num_groups, pre_model)
            _,total_M_v=create_M(X_train[val_index], num_groups, pre_model)
            
            #create a new network
            
            model = create_model(lambda_optimal, beta_optimal, alpha_this,'train', FLAGS)
            model.fit(X_train[train_index], [Y_train[train_index], Y_tr_cl[train_index], total_M_t], epochs=FLAGS.num_train_epochs, batch_size=FLAGS.batch_size_train, 
                       validation_data= (X_train[val_index], [Y_train[val_index], Y_tr_cl[val_index], total_M_v]), verbose=0)
            
            score = model.evaluate(X_train[val_index], [Y_train[val_index], Y_tr_cl[val_index], total_M_v], verbose=0)

            error = 1 - score[4]
            cr = score[6]
            
            # print the performance on validation set
            print('Alpha is: %f, error on validation: %f \n' % (alpha_this, error))
            cr_this.append(cr)
            error_this.append(error)
            
        cr_total.append(cr_this)
        error_total.append(error_this)
        cr_mean.append(np.mean(np.array(cr_this)))
        cr_std.append(np.std(np.array(cr_this)))
        error_mean.append(np.mean(np.array(error_this)))
        error_std.append(np.std(np.array(error_this)))      
    
    return error_total, error_mean, error_std, cr_total, cr_mean, cr_std
    
    
