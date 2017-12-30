## Reading the EEG data from matlab .mat files and segmenting them to create the training examples.

from scipy.io import loadmat
import numpy as np
import os

# create list of files. one file per subject data.
path = './data/p300/resampled_50/'
files = [f for f in os.listdir(path) if f.endswith('.mat')]

for fname in files:

    print 'working on file', fname
    data = loadmat(path+fname)
    sampling_rate = 5.086299945003593e+02/10
    # creating variables and organizing axeses
    # EEG data
    eeg = data['eeg']
    eeg = np.swapaxes(eeg, 0,2)
    eeg = np.swapaxes(eeg, 1, 2)
    # the attended targets
    target = data['target']
    target = np.reshape(target, (target.shape[0],))
    # the stimuli
    trigger = data['trigger']
    trigger = np.swapaxes(trigger, 0,1)
    # number of trials
    trials = eeg.shape[0]
    # choosing the pre and post stimulus interval for each segment
    prev = int(round(sampling_rate*0.1))
    aft = int(round(sampling_rate*0.7))
    segments = np.empty(shape=(trials*60,30,prev+aft))
    labels = []
    triggers = []
    # looping through all the trials of the subjects and segment them. There are 60 segments per trial.
    for k in range(trials):
        indices = np.nonzero(trigger[k])
        trigs = trigger[k,indices]
        indices = np.round(indices[0]/10.0).astype(int)
        for j,i in enumerate(indices[0:60]):
            segment = eeg[k,:,i-prev:i+aft]
            segments[k*60+j] = segment
            if trigs[0,j] == target[k]:
                labels.append(1)
            else:
                labels.append(0)
            triggers.append(trigs[0,j])
    labels = np.array(labels)
    triggers = np.array(triggers)

    print segments.shape, labels.shape, triggers.shape, target.shape
    
    # create new directory for the subject and save tje arrays.
    subject = os.path.splitext(fname)[0].split('_')[1]
    os.makedirs('./data/50/subjects/'+subject)
    # save the arrays to files to be used agian for modelling.
    np.save(open('./data/50/subjects/'+subject+'/trials.npy', 'w'), eeg)
    np.save(open('./data/50/subjects/'+subject+'/segments.npy', 'w'), segments)
    np.save(open('./data/50/subjects/'+subject+'/labels.npy', 'w'), labels)
    np.save(open('./data/50/subjects/'+subject+'/triggers.npy', 'w'), triggers)
    np.save(open('./data/50/subjects/'+subject+'/target.npy', 'w'), target)
