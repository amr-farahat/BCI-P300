from __future__ import division
import numpy as np
import tensorflow as tf
import random as rn
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
from models import *
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, LeavePGroupsOut, GroupKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import itertools
from utils import plot_confusion_matrix
import pdb
import os
from os import listdir
from os.path import isfile, join
import time
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.externals import joblib
from keras.models import load_model


def load_data(subject, mode='eeg', channels=[], frequency=50):
    data_path = 'data/'+str(frequency)+'/subjects/'
    if mode=='eeg':
        x = np.load(open(data_path+subject+'/segments.npy'))
    elif mode=='meg':
        x = np.load(open(data_path+subject+'/meg_segments.npy'))
    else:
        print 'Wrong mode. you can only choose eeg or meg'
    y = np.load(open(data_path+subject+'/labels.npy'))
    t = np.load(open(data_path+subject+'/target.npy'))
    tr = np.load(open(data_path+subject+'/triggers.npy'))
    temp_t = []
    for i in t:
        trial_index = np.ones((60,))*i
        temp_t.extend(np.ones((60,))*i)
    t = np.array(temp_t).astype(int)

    if len(channels):
        x = x[:,channels,:]

#    indexes = np.random.permutation(len(tr))
#    tr = tr[indexes]
    return x, y, t, tr
def preprocessing(x, frequency='50_avg'):
    x = np.swapaxes(x,1,2)
    ## baseline correction
    if frequency == '50_avg':
        bl = 5
    elif frequency=='250':
        bl = 25
#    bl = int(frequency/10)
    corrected = np.empty_like(x)
    for i in range(x.shape[0]):
        baselines = np.mean(x[i,0:bl,:], axis=0)
        corrected[i] = x[i] - baselines
    x = corrected
    return x

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)
def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)
def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)
def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)
def balanced_accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall_p = float(tp) / (tp + fn)
    recall_n = float(tn) / (tn + fp)
    return (recall_p + recall_n) /2.0

#def recognition_accuracy(probs):
#    trials = int(len(probs)/60)
#    n_correct = 0
#    matrix = []
#    for i in range(trials):
#        classes = np.zeros((12,))
#        for k in range(i*60,i*60+60):
#
#            classes[tr_test[k]-1] += probs[k]
#        choosen = np.argmax(classes)+1
#        if choosen == t_test[i*60]:
#            n_correct+=1
#        confidence = classes/float(np.sum(classes))
#        row = np.array([confidence, t_test[i*60]])
#        matrix.append(row)
#    return np.array([n_correct/float(trials), np.array(matrix)])
def recognition_accuracy(probs):
#    pdb.set_trace()
    trials = int(len(probs)/60)
    n_correct = np.zeros((5,))
    matrix = []
    for i in range(trials):
        classes = [[] for s in range(12)]
        for k in range(i*60,i*60+60):

            classes[tr_test[k]-1].append(probs[k])
        classes = np.array(classes)
#        pdb.set_trace()
        if classes.ndim > 1:
            classes = np.cumsum(classes, axis=1)
            choosen = np.argmax(classes, axis=0)+1
            choices = []
            for j in range(5):
                if choosen[j] == t_test[i*60]:
                    n_correct[j]+=1
                choices.append((t_test[i*60], choosen[j]))
    #            confidence = classes/float(np.sum(classes))
    #            row = np.array([confidence, t_test[i*60]])
    #            matrix.append(row)
            matrix.append(choices)
    return np.array([n_correct/float(trials), np.array(matrix)])

def bitperminute(p, n, t):
    p[p==1] = 1-np.finfo(float).eps
    B = np.log2(n)+p*np.log2(p)+(1-p)*np.log2((1-p)/(n-1).astype(float))
    return B*(float(60)/t)

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.val_aucs = []
        self.val_balanced_acc = []
        self.val_recognition_acc = []
        self.val_bpm = []
        self.test_acc = []
        self.test_loss = []
    def on_epoch_end(self, epoch, logs={}):
#        probs = self.model.predict(self.validation_data[0])
#        y_pred = np.round(probs)
#        y_true = self.validation_data[1]
        probs = self.model.predict(x_test)
        probs = probs.ravel()
        y_pred = np.round(probs)
        y_true = y_test
        self.test_acc.append(accuracy_score(y_true, y_pred))
        self.test_loss.append(self.model.evaluate(x_test, y_test, batch_size = 64, verbose=0)[0])
        self.val_precisions.append(precision(y_true, y_pred))
        self.val_recalls.append(recall(y_true, y_pred))
        self.val_f1s.append(f1(y_true, y_pred))
        self.val_aucs.append(auc(y_true, probs))
        self.val_balanced_acc.append(balanced_accuracy(y_true, y_pred))
        self.val_recognition_acc.append(recognition_accuracy(probs))
        self.val_bpm.append(bitperminute(self.val_recognition_acc[-1][0], np.ones((5,))*12,np.arange(2,12,2)))
        return


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
#    if epoch:
#        lrate = initial_lrate/np.sqrt(epoch)
#    else:
#        return initial_lrate
    return lrate

def cv_splitter(x, n_splits=5):

    n_segments = x.shape[0]
    n_trials = int(n_segments/60)
    groups = []
    for i in range(n_trials):
        trial_index = np.ones((60,))*i
        groups.extend(trial_index)
    groups = np.array(groups)

    window = int(np.round(n_trials/float(n_splits)))

    intervals=[]
    for i in range(0,n_trials,window):
        intervals.append(range(i,np.minimum(i+window, n_trials)))

    if len(intervals[-1])<window/2.0:
        intervals[-2].extend(intervals[-1])
        intervals.pop()

    folds = []

    indices = np.arange(len(groups))
    for interval in intervals:
        test_indices =  np.array([])
        for trial in interval:
            test_indices = np.append(test_indices, np.where(groups==trial)).astype(int)
        train_indices = indices[np.invert(np.isin(indices, test_indices))]

        folds.append((train_indices, test_indices))

    return folds

def compute_metrics(metrics, probs, y_predict, y_test):
    metrics['val_acc'].append(accuracy_score(y_test, y_predict))
    metrics['val_precisions'].append(precision(y_test, y_predict))
    metrics['val_recalls'].append(recall(y_test, y_predict))
    metrics['val_f1s'].append(f1(y_test, y_predict))
    metrics['val_aucs'].append(auc(y_test, probs))
    metrics['val_balanced_acc'].append(balanced_accuracy(y_test, y_predict))
    metrics['val_recognition_acc'].append(recognition_accuracy(probs)[0])
    metrics['val_bpm'].append(bitperminute(metrics['val_recognition_acc'][-1], np.ones((5,))*12,np.arange(2,12,2)))

    return metrics

def cross_validator(data,subject, n_splits=5, epochs=10, lr=0.0003, batch_size=64, model_name="",
                    model_config={'bn':True, 'dropout':True, 'branched':True, 'nonlinear':'tanh'},
                    early_stopping=True,
                    use_deep_features=False,
                    patience=10):

    if model_name.startswith('deep'):

        metrics = []
        histories = []
    else:
        m = ['acc', 'val_acc', 'val_precisions', 'val_recalls', 'val_f1s', 'val_aucs',
                 'val_balanced_acc', 'val_recognition_acc', 'val_bpm']
        metrics = {key: [] for key in m }
        histories = False

    cnf_matrices = []
    x = data[0]
    y = data[1]
    t = data[2]
    tr = data[3]

    if use_deep_features:
        path = './models/subjects/'
        load_model_name = 'deep_subjective_branched_250_thesis1'
        files = [f for f in listdir(join(path, subject)) if isfile(join(path, subject, f))]
        myfiles = [file for file in files if load_model_name in file ]
        myfiles.sort()
#    skf = StratifiedKFold(n_splits=5)
#    for train, test in skf.split(x, y):

#    sss = StratifiedShuffleSplit(n_splits=1, test_size=.1, random_state=0)
#    for train, test in sss.split(x, y):

    for i, (train, test) in enumerate(cv_splitter(x, n_splits=n_splits)):

        if use_deep_features:
            base_model = load_model(join(path, subject, myfiles[i]))
            model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)


#        print train
#        print test
#        continue
        global y_test
        x_tv, x_test, y_tv, y_test = x[train], x[test], y[train], y[test]

        global t_test
        t_test = t[test]
        global tr_test
        tr_test = tr[test]

        if model_name.startswith('deep') and early_stopping:
            x_train, x_valid, y_train, y_valid = train_test_split(x_tv, y_tv,
                                                                  stratify=y_tv,
                                                                  random_state=42,
                                                                  test_size=0.2)
        else:
            x_train = x_tv
            y_train = y_tv
            if use_deep_features:
                x_train, y_train, x_test, y_test = resample_transform((x_train, y_train, x_test, y_test), resample=False)
                x_train = model.predict(x_train)
                x_test = model.predict(x_test)
        # standarization of the data
        # computing the mean and std on the training data
#        scalar = StandardScaler(with_mean=False)
##         mus = []
#        stds = []
#        trials_no = x_train.shape[0]
#        for i in range(trials_no):
#            scalar.fit(x_train[i])
##             mu = scalar.mean_
#            std = scalar.scale_
##             mus.append(mu)
#            stds.append(std)
#        #scalar.fit(x_train.reshape((x_train.shape[0]*x_train.shape[1], x_train.shape[2])))
#
#        # tranbsforming the training data
##         scalar.mean_ = np.mean(mus, axis=0)
#        scalar.scale_ = np.mean(stds, axis=0)
#        normalized_x_train = np.empty_like(x_train)
#        for i in range(trials_no):
#            temp = scalar.transform(x_train[i])
#            normalized_x_train[i] = temp
#
#        # transforming the test data
#        normalized_x_test = np.empty_like(x_test)
#        trials_no = x_test.shape[0]
#        for i in range(trials_no):
#            temp = scalar.transform(x_test[i])
#            normalized_x_test[i] = temp

#        normalized_x_train = x_train
#        normalized_x_test = x_test

        #standarization
        scalar = StandardScaler(with_mean=True)
        scalar.fit(x_train.reshape(x_train.shape[0],-1))
        x_train = scalar.transform(x_train.reshape(x_train.shape[0], -1)).reshape(x_train.shape)
        x_test = scalar.transform(x_test.reshape(x_test.shape[0], -1)).reshape(x_test.shape)
        if model_name.startswith('deep') and early_stopping:
            x_valid = scalar.transform(x_valid.reshape(x_valid.shape[0], -1)).reshape(x_valid.shape)

#        x_train_reshaped = x_train.reshape(x_train.shape[0],-1)
#        x_test_reshaped = x_test.reshape(x_test.shape[0], -1)
#        mins = np.min(x_train_reshaped , axis=0)
#        maxs = np.max(x_train_reshaped, axis=0)
#        normalized_x_train = 2*(x_train_reshaped-mins)/(maxs-mins)-1
#        normalized_x_test = 2*(x_test_reshaped-mins)/(maxs-mins)-1
#        normalized_x_train = np.reshape(normalized_x_train, x_train.shape)
#        normalized_x_test = np.reshape(normalized_x_test, x_test.shape)
##

                #resampling the data


        if model_name.startswith('deep'):
            n_samples, timepoints, channels = x_train.shape
            x_train = np.reshape(x_train, (n_samples, timepoints * channels))
            ros = RandomOverSampler(random_state=0)
            x_res, y_res = ros.fit_sample(x_train, y_train)
            x_train = np.reshape(x_res, (x_res.shape[0], timepoints, channels))
            y_train = y_res

        x_train = np.expand_dims(x_train, axis=3)
        global x_test
        x_test = np.expand_dims(x_test, axis=3)
        if model_name.startswith('deep') and early_stopping:
            x_valid = np.expand_dims(x_valid, axis=3)

#        c = compute_class_weight('balanced', [0, 1], y)
#        class_weight = {0:c[0],1:c[1]}
#        print class_weight
#         pdb.set_trace()
        #compiling the model
        if model_name.startswith('deep'):
            if 'branched' in model_name:
                if '250' in model_name:
                    path = './models/subjects/'
                    load_model_name = 'deep_intersubjective_branched_250_thesis2_2'
                    files = [f for f in listdir(join(path, subject)) if isfile(join(path, subject, f))]
                    myfiles = [file for file in files if load_model_name in file ]
                    model = load_model(join(path, subject, myfiles[0]))
                    
#                    model = branched2(x.shape, model_config=model_config, f=5)
                else:
                    path = './models/subjects/'
                    load_model_name = 'deep_intersubjective_branched_50_avg_thesis2_2'
                    files = [f for f in listdir(join(path, subject)) if isfile(join(path, subject, f))]
                    myfiles = [file for file in files if load_model_name in file ]
                    model = load_model(join(path, subject, myfiles[0]))
                    
#                    model = branched2(x.shape, model_config=model_config, f=1)
            elif 'eegnet' in model_name:
                if '250' in model_name:
                    model = create_eegnet(x.shape, f=4)
                else:
                    model = create_eegnet(x.shape, f=1)
            elif 'cnn' in model_name:
                if '250' in model_name:
                    model = create_cnn(x.shape, f=5)
                else:
                    model = create_cnn(x.shape, f=1)

#            opt = Adam(lr=lr)
            opt = SGD(lr=1e-4, momentum=0.9)
            lrate = LearningRateScheduler(step_decay)

            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

            m = Metrics()
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=int(patience/2), min_lr=0)
            early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience, verbose=0, mode='auto')
            mod_path = './models/subjects/'+subject
            timestr = time.strftime("%Y%m%d-%H%M")

            checkpointer = ModelCheckpoint(filepath=mod_path+'/best_'+model_name+'_'+timestr,
                                           monitor='val_loss', verbose=1, save_best_only=True)
            if early_stopping:
                history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2 ,
                               validation_data=(x_valid, y_valid), callbacks=[m, early_stop, checkpointer, reduce_lr],
                               )
            else:
                history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2 ,
                               validation_data=(x_test, y_test), callbacks=[m],
                               )

            metrics.append(m)
            histories.append(history)

            probabilities = model.predict(x_test, batch_size=batch_size, verbose=0)
            y_predict = [(round(k)) for k in probabilities]
        else:
            x_train = np.reshape(x_train, (x_train.shape[0], -1))
            x_test = np.reshape(x_test, (x_test.shape[0],-1))
            if 'svm' in model_name:
                clf = svm.LinearSVC(random_state=4)
            elif 'lda' in model_name:
                if 'shrinkage' in model_name:
                    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                else:
                    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
            clf.fit(x_train, y_train)
            y_predict = clf.predict(x_test)
            if 'svm' in model_name:
                probs = clf.decision_function(x_test)
            elif 'lda' in model_name:
                probs = clf.decision_function(x_test)
            metrics['acc'].append(clf.score(x_train, y_train))
            metrics = compute_metrics(metrics, probs, y_predict, y_test)

        cnf_matrix = confusion_matrix(y_test, y_predict)
        cnf_matrices.append(cnf_matrix)
    return metrics, histories, cnf_matrices



def save_resutls(metrics, histories, subject, suffix='', clf=None, early_stopping=True, patience=10):
    res_path = './results/subjects/'+subject
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    mod_path = './models/subjects/'+subject
    if not os.path.exists(mod_path):
        os.makedirs(mod_path)

    timestr = time.strftime("%Y%m%d-%H%M")
    if histories:
        results = []
        for i in range(len(histories)):
            if not early_stopping:
                histories[i].model.save(mod_path+'/model'+str(i+1)+'_'+suffix+'_'+timestr+'.h5')
            dic = histories[i].history
            dic['val_precisions'] = metrics[i].val_precisions
            dic['test_loss'] = metrics[i].test_loss
            dic['test_acc'] = metrics[i].test_acc
            dic['val_recalls'] = metrics[i].val_recalls
            dic['val_f1s'] = metrics[i].val_f1s
            dic['val_aucs'] = metrics[i].val_aucs
            dic['val_balanced_acc'] = metrics[i].val_balanced_acc
            dic['val_recognition_acc'] = [p[0] for p in metrics[i].val_recognition_acc]
            dic['choices'] = [p[1] for p in metrics[i].val_recognition_acc]
            dic['val_bpm'] = metrics[i].val_bpm
#            dic['val_trials_classification'] = [p[1] for p in metrics[i].val_recognition_acc]
            results.append(dic)

        final_results = {key:[] for key in results[0].keys()}
        #print final_results
#        pdb.set_trace()
        for model in results:
#            best_i = np.argmax(model['val_acc'])
            keys = model.keys()
            for key in keys:
                if early_stopping:
                    final_results[key].append(model[key][-(patience+1)])
                else:
                    final_results[key].append(model[key][-1])
    else:
        final_results = metrics
    super_final_results = {key:None for key in final_results.keys() if key != 'val_trials_classification'}
    for key in final_results.keys():
        if key != 'choices':
#            pdb.set_trace()
            mean = np.mean(final_results[key], axis=0)
            std = np.std(final_results[key], axis=0)
            super_final_results[key] = {'mean':mean, 'std':std}
    super_final_results = np.array([super_final_results])

    if histories:
        np.savez(open(res_path+'/all_results_'+suffix+'_'+timestr+'.npz','w'), results=results,
             final_results=final_results, super_final_results=super_final_results)
    else:
        np.savez(open(res_path+'/all_results_'+suffix+'_'+timestr+'.npz','w'), super_final_results=super_final_results)
        joblib.dump(clf, mod_path+'/model'+'_'+suffix+'_'+timestr+'.pkl')
    return super_final_results




def collect_data_intersubjective(subjects, test_subject, mode='eeg', channels=[], all_sub=False, frequency=50):

    x_train = []
    y_train = []
    for subject in subjects:
        if subject == test_subject:
            if not all_sub:
                continue

        x, y, t, tr = load_data(subject, mode=mode, channels=channels, frequency=frequency)
        x = preprocessing(x, frequency=frequency)
#        pdb.set_trace()
        x_train.extend(x)
        y_train.extend(y)
        del x,y,t,tr

    if all_sub:
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train

    x,y,t,tr = load_data(test_subject, mode=mode, channels=channels, frequency=frequency)
    x = preprocessing(x, frequency=frequency)
    x_test = x
    y_test = y
    t_test = t
    tr_test = tr
    del x,y,t,tr


    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    t_test = np.array(t_test)
    tr_test = np.array(tr_test)

#    if len(channels):
#        x_train = x_train[:,:,channels]
#        x_test = x_test[:,:,channels]
    return x_train, y_train, x_test, y_test, t_test, tr_test


def resample_transform(data, resample=True):

    if len(data) > 4:
        x_train, y_train, x_test, y_test, x_valid, y_valid = data
    else:
        x_train, y_train, x_test, y_test = data


    # standarization of the data
    # computing the mean and std on the training data
#    scalar = StandardScaler(with_mean=False)
#    stds = []
#    trials_no = x_train.shape[0]
#    for i in range(trials_no):
#        scalar.fit(x_train[i])
#        std = scalar.scale_
#        stds.append(std)
#
#    scalar.scale_ = np.mean(stds, axis=0)
#    normalized_x_train = np.empty_like(x_train)
#    for i in range(trials_no):
#        temp = scalar.transform(x_train[i])
#        normalized_x_train[i] = temp
#
#    # transforming the test data
#    normalized_x_test = np.empty_like(x_test)
#    trials_no = x_test.shape[0]
#    for i in range(trials_no):
#        temp = scalar.transform(x_test[i])
#        normalized_x_test[i] = temp
#
#    x_train = normalized_x_train
#    x_test = normalized_x_test


    scalar = StandardScaler(with_mean=True)
    scalar.fit(x_train.reshape(x_train.shape[0],-1))
    normalized_x_train = scalar.transform(x_train.reshape(x_train.shape[0], -1)).reshape(x_train.shape)
    normalized_x_test = scalar.transform(x_test.reshape(x_test.shape[0], -1)).reshape(x_test.shape)
    if len(data)>4:
        normalized_x_valid = scalar.transform(x_valid.reshape(x_valid.shape[0], -1)).reshape(x_valid.shape)
    x_train = normalized_x_train
    x_test = normalized_x_test
    if len(data) > 4:
        x_valid = normalized_x_valid

    if resample:
        #resampling the data
        n_samples, timepoints, channels = x_train.shape
        x_train = np.reshape(x_train, (n_samples, timepoints * channels))
        ros = RandomOverSampler(random_state=0)
        x_res, y_res = ros.fit_sample(x_train, y_train)
        x_train = np.reshape(x_res, (x_res.shape[0], timepoints, channels))
        y_train = y_res

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    if len(data) > 4:
        x_valid = np.expand_dims(x_valid, axis=3)
        return x_train, y_train, x_test, y_test, x_valid, y_valid
    return x_train, y_train, x_test, y_test

def transform(data):

    x_train, y_train, x_test, y_test = data


    # standarization of the data
    # computing the mean and std on the training data
    scalar = StandardScaler(with_mean=False)
    stds = []
    trials_no = x_train.shape[0]
    for i in range(trials_no):
        scalar.fit(x_train[i])
        std = scalar.scale_
        stds.append(std)

    scalar.scale_ = np.mean(stds, axis=0)
    normalized_x_train = np.empty_like(x_train)
    for i in range(trials_no):
        temp = scalar.transform(x_train[i])
        normalized_x_train[i] = temp

    # transforming the test data
    normalized_x_test = np.empty_like(x_test)
    trials_no = x_test.shape[0]
    for i in range(trials_no):
        temp = scalar.transform(x_test[i])
        normalized_x_test[i] = temp

    x_train = normalized_x_train
    x_test = normalized_x_test

    return x_train, y_train, x_test, y_test
def intersubjective_shallow(data, model_name):

    x_train, y_train, x_test, y_test, o_t_test, o_tr_test = data

    x_train, y_train, x_test, y_test = resample_transform((x_train, y_train, x_test, y_test), resample=False)

    global t_test
    t_test = o_t_test
    global tr_test
    tr_test = o_tr_test
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    m = ['acc', 'val_acc', 'val_precisions', 'val_recalls', 'val_f1s', 'val_aucs',
             'val_balanced_acc', 'val_recognition_acc', 'val_bpm']
    metrics = {key: [] for key in m }
    history = False
    if 'svm' in model_name:
        clf = svm.LinearSVC(random_state = 0)
    elif 'lda' in model_name:
        if 'shrinkage' in model_name:
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        else:
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    probs = clf.decision_function(x_test)
    metrics['acc'].append(clf.score(x_train, y_train))
    metrics = compute_metrics(metrics, probs, y_predict, y_test)
    cnf_matrix = confusion_matrix(y_test, y_predict)

    return metrics, history, cnf_matrix, clf

def intersubjective_training(data,model_name, subject, epochs=5, lr=0.001,
                             batch_size=128,
                             model_config={'bn':True, 'dropout':True, 'branched':True, 'nonlinear':'tanh'},
                             early_stopping=True, patience=10):

    global y_test
    x_tv, y_tv, x_test, y_test, o_t_test, o_tr_test = data

    if early_stopping:
#        pdb.set_trace()
        x_train, x_valid, y_train, y_valid = train_test_split(x_tv, y_tv,
                                                                  stratify=y_tv,
                                                                  test_size=0.2)
        global x_test
        x_train, y_train, x_test, y_test, x_valid, y_valid = resample_transform((x_train, y_train, x_test, y_test, x_valid, y_valid))
    else:
        x_train = x_tv
        y_train = y_tv
        global x_test
        x_train, y_train, x_test, y_test = resample_transform((x_train, y_train, x_test, y_test))

    global t_test
    t_test = o_t_test
    global tr_test
    tr_test = o_tr_test

    if 'branched' in model_name:
        if '250' in model_name:
            model = branched2(x_train.shape, model_config=model_config, f=5)
        else:
            model = branched2(x_train.shape, model_config=model_config, f=1)
    elif 'eegnet' in model_name:
        if '250' in model_name:
            model = create_eegnet(x_train.shape, f=4)
        else:
            model = create_eegnet(x_train.shape,  f=1)
    elif 'cnn' in model_name:
        if '250' in model_name:
            model = create_cnn(x_train.shape,  f=5)
        else:
            model = create_cnn(x_train.shape,  f=1)

    lrate = LearningRateScheduler(step_decay)
    adam = Adam(lr=lr)
    # lrate = LearningRateScheduler(step_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    m = Metrics()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=int(patience/2), min_lr=0)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience, verbose=0, mode='auto')
    mod_path = './models/subjects/'+subject
    timestr = time.strftime("%Y%m%d-%H%M")


    checkpointer = ModelCheckpoint(filepath=mod_path+'/best_'+model_name+'_'+timestr,
                                           monitor='val_loss', verbose=1, save_best_only=True)
    if early_stopping:
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2,
                       validation_data=(x_valid, y_valid), callbacks=[m, early_stop, checkpointer, reduce_lr])
    else:
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2,
                       validation_data=(x_test, y_test), callbacks=[m])




    probabilities = model.predict(x_test, batch_size=batch_size, verbose=0)
    y_predict = [(round(k)) for k in probabilities]

    cnf_matrix = confusion_matrix(y_test, y_predict)

    return m, history, cnf_matrix



def finetune(model, data, model_name, subject, epochs=10, train_trials=40, mode='all', early_stopping=True, patience=10):

    for layer in model.layers[:26]:
        if mode=='all':
            layer.trainable = True
        elif mode=='top':
            layer.trainable = False
        else:
            print 'wrong keyword argument'
            return
#     print model.summary()

    opt = SGD(lr=1e-4, momentum=0.9)

    model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])

    x_test, y_test, o_t_test, o_tr_test = data

    segments = train_trials * 60

    x_tv = x_test[0:segments]
    y_tv = y_test[0:segments]

    if early_stopping:
        x_train, x_valid, y_train, y_valid = train_test_split(x_tv, y_tv,
                                                              stratify=y_tv,
                                                              test_size=0.2)
        x_test = x_test[segments:]
        global y_test
        y_test = y_test[segments:]
        global x_test
        x_train, y_train, x_test, y_test, x_valid, y_valid = resample_transform((x_train, y_train, x_test, y_test, x_valid, y_valid))
    else:
        x_train = x_tv
        y_train = y_tv
        x_test = x_test[segments:]
        global y_test
        y_test = y_test[segments:]
        global x_test
        x_train, y_train, x_test, y_test = resample_transform((x_train, y_train, x_test, y_test))

#    x_test = x_test[segments:]
#    y_test = y_test[segments:]



    global t_test
    t_test = o_t_test[segments:]
    global tr_test
    tr_test = o_tr_test[segments:]
#    #resampling the data
#    n_samples, timepoints, channels, z = x_train.shape
#    x_train = np.reshape(x_train, (n_samples, timepoints * channels))
#    ros = RandomOverSampler(random_state=0)
#    x_res, y_res = ros.fit_sample(x_train, y_train)
#    x_train = np.reshape(x_res, (x_res.shape[0], timepoints, channels))
#    y_train = y_res
#
#    x_train = np.expand_dims(x_train, axis=3)
#     x_test = np.expand_dims(x_test, axis=3)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=patience, verbose=0, mode='auto')


    m = Metrics()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=int(patience/2), min_lr=0)
    mod_path = './models/subjects/'+subject
    timestr = time.strftime("%Y%m%d-%H%M")

    checkpointer = ModelCheckpoint(filepath=mod_path+'/best_'+model_name+'_'+timestr,
                                           monitor='val_loss', verbose=1, save_best_only=True)
    if early_stopping:
        history = model.fit(x_train, y_train, batch_size=64, epochs=epochs, shuffle=True, verbose=2,
                       validation_data=(x_valid, y_valid), callbacks=[m, early_stop, checkpointer, reduce_lr])
    else:
        history = model.fit(x_train, y_train, batch_size=64, epochs=epochs, shuffle=True, verbose=2,
                       validation_data=(x_test, y_test), callbacks=[m])


    probabilities = model.predict(x_test, batch_size=128, verbose=0)
    y_predict = [(round(k)) for k in probabilities]

    cnf_matrix = confusion_matrix(y_test, y_predict)


    return m, history, cnf_matrix
