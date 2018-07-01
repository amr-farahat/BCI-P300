_intersubjective
from src import collect_data_intersubjective, resample_transform, intersubjective_training, save_resutls, intersubjective_shallow, finetune
import os
from os.path import isfile, join
from keras.models import load_model, clone_model

from keras.optimizers import SGD
import sys
#import csv
import pdb
#%%

#channels_sorted = np.load('plots/saliency_maps/channels_sorted.npy')
#significant_channels = channels_sorted[:len(channels_sorted)/2]
#insignificant_channels = channels_sorted[len(channels_sorted)/2:]
#significant_channels.sort()
#insignificant_channels.sort()

subjects = [name for name in os.listdir("./data/50/subjects/")]
#test_subjects = subjects
#test_subjects = ['ab82']
test_subjects = [sys.argv[1]]
# pdb.set_trace()

batch_size = 64
lr = 0.001
early_stopping = True
epochs = 500
patience = 20

model_config={'bn':True, 'dropout':True, 'branched':True, 'deep':True, 'nonlinear':'tanh'}

ft = False
ft_mode = 'all'
ft_trials = [10, 20, 30, 40, 50, 60, 70]
#
#model_name = 'deep_branched_intersubjective-111-'+str(dataset)
#dataset = '50_avg'

datasets = ['50_avg', '250']


for dataset in datasets:
    model_names = ['deep_intersubjective_branched_'+str(dataset)+'_thesis1',
              'deep_intersubjective_eegnet_'+str(dataset)+'_thesis1',
              'deep_intersubjective_cnn_'+str(dataset)+'_thesis1',
              'lda_intersubjective_shrinkage_'+str(dataset)+'_thesis1',
              'lda_intersubjective_'+str(dataset)+'_thesis1',
              'deep_intersubjective_branched_no_bn_'+str(dataset)+'_thesis1',
              'deep_intersubjective_branched_no_dropout_'+str(dataset)+'_thesis1',
              'deep_intersubjective_branched_no_branched_'+str(dataset)+'_thesis1',
              'deep_intersubjective_branched_no_deep_'+str(dataset)+'_thesis1',
              'deep_intersubjective_branched_relu_'+str(dataset)+'_thesis1',
              'deep_intersubjective_branched_elu_'+str(dataset)+'_thesis1']

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


        for test_subject in test_subjects:
            print 'working on subject', test_subject
            #test_subject = subject
            #collecting the data
            print 'Collecting data ...'
            x_train, y_train, x_test, y_test, o_t_test, o_tr_test = collect_data_intersubjective(subjects,
                                                                                                 test_subject,
                                                                                                 mode='eeg',
                                                                                                 channels=range(29),
                                                                                             frequency=dataset)


            #Train
            print "Training ..."
            if model_name.startswith('deep'):
                metrics, history, cnf_matrix = intersubjective_training((x_train, y_train, x_test, y_test, o_t_test, o_tr_test),
                                                                        model_name,
                                                                        test_subject,
                                                                        epochs=epochs,
                                                                        lr=lr,
                                                                        batch_size=batch_size,
                                                                        model_config=model_config,
                                                                        early_stopping=early_stopping,
                                                                        patience=patience)
                super_final_results = save_resutls([metrics], [history], test_subject,
                                                   suffix=model_name,
                                                   early_stopping=early_stopping,
                                                   patience=patience)
            else:
                metrics, history, cnf_matrix, clf = intersubjective_shallow((x_train, y_train, x_test, y_test, o_t_test, o_tr_test),
                                                                       model_name)
                super_final_results = save_resutls(metrics, history, test_subject, suffix=model_name, clf=clf)

            print 'DA:', model_name,'_', test_subject, ' is:', super_final_results[0]['val_recognition_acc']['mean']
            print 'BA:', model_name,'_', test_subject, ' is:', super_final_results[0]['val_balanced_acc']['mean']
            print 'Recall:', model_name,'_', test_subject, ' is:', super_final_results[0]['val_recalls']['mean']
            print 'precision:', model_name,'_', test_subject, ' is:', super_final_results[0]['val_precisions']['mean']

            if model_name.startswith('deep_intersubjective_branched') and ft:
                model=clone_model(history.model)
                weights = history.model.get_weights()

                for i in ft_trials:
                    model_name_modified = 'deep_intersubjective_branched_ft_'+str(i)+'_trials_'+str(dataset)+'_thesis2'
                    model_name_modified = model_name+'_ft_'+str(i)+'_trials'
                    model.set_weights(weights)
                    metrics, history, cnf_matrix = finetune(model,
                                                            (x_test, y_test, o_t_test, o_tr_test),
                                                            model_name_modified,
                                                            test_subject,
                                                            epochs=epochs,
                                                            train_trials=i,
                                                            mode=ft_mode,
                                                            early_stopping=early_stopping,
                                                            patience=patience)

                    super_final_results = save_resutls([metrics], [history], test_subject,
                                                   suffix=model_name_modified,
                                                   early_stopping=early_stopping,
                                                   patience=patience)
                    print 'DA:', model_name_modified,'_', test_subject, ' is:', super_final_results[0]['val_recognition_acc']['mean']
                    print 'BA:', model_name_modified,'_', test_subject, ' is:', super_final_results[0]['val_balanced_acc']['mean']
                    print 'Recall:', model_name_modified,'_', test_subject, ' is:', super_final_results[0]['val_recalls']['mean']
                    print 'precision:', model_name_modified,'_', test_subject, ' is:', super_final_results[0]['val_precisions']['mean']
