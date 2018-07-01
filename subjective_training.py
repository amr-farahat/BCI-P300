

from src import load_data, preprocessing, cross_validator, save_resutls

import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from utils import plot_confusion_matrix
import sys
import pdb
#%%

eeglabel =  ['Fz','Cz','Pz','Oz','Iz','Fp1','Fp2','F3','F4','F7','F8','T7','T8','C3','C4','P3','P4','O9','O10','P7',
             'P8','FC1','FC2','CP1','CP2','PO3','PO4','PO7','PO8','LMAST']

subjects = [sys.argv[1]]
#subjects = ['kq84']

batch_size = 64
lr = 0.001
early_stopping=True
patience=20
epochs=500
model_config={'bn':True, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'tanh'}



datasets = ['50_avg','250']


for dataset in datasets:
    model_names = ['deep_subjective_branched_'+str(dataset)+'_thesis1',
              'deep_subjective_eegnet_'+str(dataset)+'_thesis1',
              'deep_subjective_cnn_'+str(dataset)+'_thesis1',
              'lda_subjective_shrinkage_'+str(dataset)+'_thesis1',
              'lda_subjective_'+str(dataset)+'_thesis1',
              'deep_subjective_branched_no_bn_'+str(dataset)+'_thesis1',
              'deep_subjective_branched_no_dropout_'+str(dataset)+'_thesis1',
              'deep_subjective_branched_no_branched_'+str(dataset)+'_thesis1',
              'deep_subjective_branched_no_deep_'+str(dataset)+'_thesis1',
              'deep_subjective_branched_relu_'+str(dataset)+'_thesis1',
              'deep_subjective_branched_elu_'+str(dataset)+'_thesis1']

    model_configs = [{'bn':True, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'tanh'},
                    {'bn':True, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'tanh'},
                    {'bn':True, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'tanh'},
                    {'bn':True, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'tanh'},
                    {'bn':True, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'tanh'},
                    {'bn':False, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'tanh'},
                    {'bn':True, 'dropout':False, 'branched':True, 'deep':True, 'nonlinear':'tanh'},
                    {'bn':True, 'dropout':True, 'branched':False, 'deep':True, 'nonlinear':'tanh'},
                    {'bn':True, 'dropout':True, 'branched':True, 'deep':False, 'nonlinear':'tanh'},
                    {'bn':True, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'relu'},
                    {'bn':True, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'elu'}]


    for model_name, model_config in zip(model_names, model_configs):
        for subject in subjects:

            print 'working on subject : ', subject
            x, y, t, tr = load_data(subject, channels=range(29), frequency=dataset)
            x = preprocessing(x, frequency=dataset)
            metrics, histories, cnf_matrices = cross_validator((x, y, t, tr),subject,
                                                               n_splits=5, epochs=epochs,
                                                               batch_size=batch_size,
                                                               lr=lr,
                                                               model_name=model_name,
                                                               model_config=model_config,
                                                               early_stopping=early_stopping,
                                                               patience=patience)

            super_final_results = save_resutls(metrics, histories, subject, suffix=model_name, early_stopping=early_stopping,
                                               patience=patience)

            print 'DA:', model_name,'_', subject, ' is:', super_final_results[0]['val_recognition_acc']['mean']
            print 'BA:', model_name,'_', subject, ' is:', super_final_results[0]['val_balanced_acc']['mean']
            print 'Recall:', model_name,'_', subject, ' is:', super_final_results[0]['val_recalls']['mean']
            print 'precision:', model_name,'_', subject, ' is:', super_final_results[0]['val_precisions']['mean']
