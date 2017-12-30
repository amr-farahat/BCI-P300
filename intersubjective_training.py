
from src import *
from os import listdir
from os.path import isfile, join
from keras.models import load_model, clone_model
from keras.optimizers import SGD
from keras.models import clone_model
import csv

## leave one out cross validation. (training on all the subjects data except one and test on him)

# computing the signifcant channels based on the absolute value of the saliency maps.

all_grads = np.load('plots/saliency_maps/all_grads_eegnet.npy')
all_grads_mean = np.mean(all_grads, axis=0)
all_grads_mean = np.swapaxes(all_grads_mean, 0, 1)[:,:,0]
ch_significance = np.mean(np.abs(all_grads_mean), axis=1)
eeglabel =  np.array(['Fz','Cz','Pz','Oz','Iz','Fp1','Fp2','F3','F4','F7','F8','T7','T8','C3','C4','P3','P4','O9','O10','P7',
             'P8','FC1','FC2','CP1','CP2','PO3','PO4','PO7','PO8','LMAST'])
channels_sorted = np.argsort(-ch_significance)
significant_channels = channels_sorted[:5]
insignificant_channels = channels_sorted[26:]
significant_channels.sort()
insignificant_channels.sort()

subjects = [name for name in os.listdir("./data/50/subjects/")]
test_subjects = subjects

# leave one subject out cross validation.
for test_subject in test_subjects:
    print 'working on subject', test_subject
    #test_subject = subject
    #collecting the data
    x_train, y_train, x_test, y_test, o_t_test, o_tr_test = collect_data_intersubjective(subjects, test_subject, channels=[])

    #rescaling and oversampling
    x_train, y_train, x_test, y_test = resample_transform((x_train, y_train, x_test, y_test))

    #Train
    metrics, history, cnf_matrix = intersubjective_training((x_train, y_train, x_test, y_test, o_t_test, o_tr_test),
                                                      epochs=100)
    # saving the results.
    super_final_results = save_resutls([metrics], [history], test_subject, suffix='intersubjective_bayesian_permenant20')
    
    # finetuning the model with varied number of subject's own trials.
    model=clone_model(history.model)
    weights = history.model.get_weights()

    for i in [10, 20, 30, 50, 70]:
        model.set_weights(weights)
        metrics, history, cnf_matrix = finetune(model, (x_test, y_test, o_t_test, o_tr_test),
                                        epochs=500, train_trials=i , mode='all')

        super_final_results = save_resutls([metrics], [history], test_subject, suffix='intersubjective_ft_'+str(i)+'eegnet')



# collecting the saved results of a certain exdperiment and creating a csv file with the results of the metric
# that we are interested in.
trial_names = ['intersubjective_bayesian_permenant20']
metric = 'val_recognition_acc'
g_mus = []
for trial_name in trial_names:
    path = './results/subjects/'
    subjects = [name for name in os.listdir(path)]
    mus = []
    stds=[]
    for subject in subjects:
        files = [f for f in listdir(join(path, subject)) if isfile(join(path, subject, f))]
        myfile = [file for file in files if trial_name in file ][0]
        d = np.load(join(path, subject, myfile))
        sfr = d['super_final_results']
        mu = sfr[0][metric]['mean']
        std = sfr[0][metric]['std']
        mus.append(mu)
        stds.append(std)
#     g_mu = np.mean([u*100 for u in mus])
    g_mus.append([round(ff*100,2) for ff in mus])

g_mus = np.array(g_mus)
g_mus = g_mus.T
subjects = np.array(subjects)
subjects = np.expand_dims(subjects, axis=1)

with open('temp.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['']+trial_names)
    wr.writerows(np.hstack((subjects,g_mus)))
