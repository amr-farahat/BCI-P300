
from src import *
from os import listdir
from os.path import isfile, join
import csv

# training on only the subject's data with 5 fold cross validation.


subjects = [name for name in os.listdir("./data/50/subjects/")]
results = []

for subject in subjects:
    x, y, t, tr = load_data(subject)
    x = preprocessing(x)
    metrics, histories, cnf_matrices = cross_validator((x, y, t, tr), n_splits=5, epochs=500)

    super_final_results = save_resutls(metrics, histories, subject, suffix='eegnet_subjective')

    print 'working on subject : ', subject




trial_names = ['eegnet_subjective']
g_mus = []
for trial_name in trial_names:
    path = './results/subjects/'
    subjects = [name for name in os.listdir(path)]
    metric = 'val_recognition_acc'
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
    g_mus.append([round(ff*100,2) for ff in mus])

g_mus = np.array(g_mus)
g_mus = g_mus.T
subjects = np.array(subjects)
subjects = np.expand_dims(subjects, axis=1)

with open('temp.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['']+trial_names)
    wr.writerows(np.hstack((subjects,g_mus)))
