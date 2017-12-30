from __future__ import division
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Dropout, Flatten, Reshape
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda
from keras.optimizers import Adam, SGD
from keras.callbacks import Callback, TensorBoard, LearningRateScheduler, EarlyStopping
from keras import regularizers
import keras.backend as K
from imblearn.over_sampling import RandomOverSampler
#from sklearn.model_selection import train_test_split, LeavePGroupsOut, GroupKFold
from sklearn.preprocessing import StandardScaler
#from sklearn.utils.class_weight import compute_class_weight
#from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import plot_confusion_matrix
import pdb
import os
import time
import math


def load_data(subject):
    data_path = 'data/50/subjects/'
    x = np.load(open(data_path+subject+'/segments.npy'))
    y = np.load(open(data_path+subject+'/labels.npy'))
    t = np.load(open(data_path+subject+'/target.npy'))
    tr = np.load(open(data_path+subject+'/triggers.npy'))
    temp_t = []
    for i in t:
        trial_index = np.ones((60,))*i
        temp_t.extend(np.ones((60,))*i)
    t = np.array(temp_t).astype(int)

    return x, y, t, tr
def preprocessing(x):
    x = np.swapaxes(x,1,2)
    ## baseline correction
    corrected = np.empty_like(x)
    for i in range(x.shape[0]):
        baselines = np.mean(x[i,0:5,:], axis=0)
        corrected[i] = x[i] - baselines
    x = corrected
    return x

def branched(data_shape):
    timepoints = data_shape[1]
    channels = data_shape[2]

    input_data = Input(shape=(timepoints, channels, 1))

    spatial_conv = Conv2D(6, (1,channels), activation='relu', padding='valid')(input_data)
    spatial_conv = BatchNormalization()(spatial_conv)
    #spatial_conv = Dropout(0.33)(spatial_conv)
    spatial_conv = Lambda(lambda x: K.dropout(x, level=0.33))(spatial_conv)



    branch1 = Conv2D(4, (21,1), activation='relu', padding='valid')(spatial_conv)
    branch1 = BatchNormalization()(branch1)
    branch1 = MaxPooling2D((3,1))(branch1)
    #branch1 = Dropout(0.33)(branch1)
    branch1 = Lambda(lambda x: K.dropout(x, level=0.33))(branch1)
    branch1 = Flatten()(branch1)

    branch2 = Conv2D(4, (5,1), activation='relu', padding='valid')(spatial_conv)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling2D((3,1))(branch2)
    #branch2 = Dropout(0.33)(branch2)
    branch2 = Lambda(lambda x: K.dropout(x, level=0.33))(branch2)

    branch2 = Conv2D(4, (5,1), activation='relu', padding='valid')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = MaxPooling2D((2,1))(branch2)
    #branch2 = Dropout(0.33)(branch2)
    branch2 = Lambda(lambda x: K.dropout(x, level=0.33))(branch2)


    branch2 = Flatten()(branch2)


    merged = concatenate([branch1, branch2])


    dense = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs = [input_data], outputs=[dense])


    return model

def create_eegnet(data_shape):

    timepoints = data_shape[1]
    channels = data_shape[2]

    model = Sequential()

    model.add(Conv2D(16, (1,30), activation='relu',
                     kernel_regularizer=regularizers.l1_l2(0.0001), input_shape=(timepoints, channels, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Reshape((41,16,1)))

    model.add(Conv2D(4, (16,2), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(4, (2,8), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((4,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

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

def recognition_accuracy(probs):
    trials = int(len(probs)/60)
    n_correct = 0
    matrix = []
    for i in range(trials):
        classes = np.zeros((12,))
        for k in range(i*60,i*60+60):

            classes[tr_test[k]-1] += probs[k]
        choosen = np.argmax(classes)+1
        if choosen == t_test[i*60]:
            n_correct+=1
        confidence = classes/float(np.sum(classes))
        row = np.array([confidence, t_test[i*60]])
        matrix.append(row)
    return np.array([n_correct/float(trials), np.array(matrix)])


def predict_with_uncertainty(f, x, n_iter=10):
    result = []

    for iter in range(n_iter):

        result.append(f([x, 1])[0])
    #pdb.set_trace()
    prediction = np.mean(result,axis=0)
    #uncertainty = result.var(axis=0)
    return prediction
            
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.val_aucs = []
        self.val_balanced_acc = []
        self.val_recognition_acc = []
    def on_epoch_end(self, epoch, logs={}):      
        result = []
        for iter in range(20):
            result.append(self.model.predict(self.validation_data[0]))
        probs = np.mean(result,axis=0)
        y_pred = np.round(probs)
        y_true = self.validation_data[1]
        self.val_precisions.append(precision(y_true, y_pred))
        self.val_recalls.append(recall(y_true, y_pred))
        self.val_f1s.append(f1(y_true, y_pred))
        self.val_aucs.append(auc(y_true, y_pred))
        self.val_balanced_acc.append(balanced_accuracy(y_true, y_pred))
        self.val_recognition_acc.append(recognition_accuracy(probs))
        return


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 40.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
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

def cross_validator(data, n_splits=5, epochs=10):

    metrics = []
    histories = []
    cnf_matrices = []
    x = data[0]
    y = data[1]
    t = data[2]
    tr = data[3]


    for train, test in cv_splitter(x, n_splits=n_splits):

        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]

        global t_test
        t_test = t[test]
        global tr_test
        tr_test = tr[test]

        # standarization of the data
        # computing the mean and std on the training data
        scalar = StandardScaler(with_mean=False)
        stds = []
        trials_no = x_train.shape[0]
        for i in range(trials_no):
            scalar.fit(x_train[i])
            std = scalar.scale_
            stds.append(std)
        # tranbsforming the training data
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


                #resampling the data
        n_samples, timepoints, channels = x_train.shape
        normalized_x_train = np.reshape(normalized_x_train, (n_samples, timepoints * channels))
        ros = RandomOverSampler(random_state=0)
        x_res, y_res = ros.fit_sample(normalized_x_train, y_train)
        normalized_x_train = np.reshape(x_res, (x_res.shape[0], timepoints, channels))
        y_train = y_res

        normalized_x_train = np.expand_dims(normalized_x_train, axis=3)
        normalized_x_test = np.expand_dims(normalized_x_test, axis=3)


        #compiling the model

        model = create_eegnet(x.shape)

#         tb = TensorBoard(log_dir='./Graph', histogram_freq=10, write_graph=False, write_images=True)

#         adam = Adam(lr=0.001)
#         lrate = LearningRateScheduler(step_decay)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        m = Metrics()

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=0, mode='auto')

        history = model.fit(normalized_x_train, y_train, batch_size=64, epochs=epochs, shuffle=True, verbose=0,
                           validation_data=(normalized_x_test, y_test), callbacks=[m, early_stop],
                           )

        metrics.append(m)
        histories.append(history)

        probabilities = model.predict(normalized_x_test, batch_size=64, verbose=0)
        y_predict = [(round(k)) for k in probabilities]

        cnf_matrix = confusion_matrix(y_test, y_predict)
        cnf_matrices.append(cnf_matrix)
    return metrics, histories, cnf_matrices

def save_resutls(metrics, histories, subject, suffix=''):
    res_path = './results/subjects/'+subject
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    mod_path = './models/subjects/'+subject
    if not os.path.exists(mod_path):
        os.makedirs(mod_path)

    timestr = time.strftime("%Y%m%d-%H%M")

    results = []
    for i in range(len(histories)):
        histories[i].model.save(mod_path+'/model'+str(i+1)+'_'+suffix+'_'+timestr+'.h5')
        dic = histories[i].history
        dic['val_precisions'] = metrics[i].val_precisions
        dic['val_recalls'] = metrics[i].val_recalls
        dic['val_f1s'] = metrics[i].val_f1s
        dic['val_aucs'] = metrics[i].val_aucs
        dic['val_balanced_acc'] = metrics[i].val_balanced_acc
        dic['val_recognition_acc'] = [p[0] for p in metrics[i].val_recognition_acc]
        dic['val_trials_classification'] = [p[1] for p in metrics[i].val_recognition_acc]
        results.append(dic)

    final_results = {key:[] for key in results[0].keys()}
    #print final_results
    for model in results:
        keys = model.keys()
        for key in keys:
            final_results[key].append(model[key][-1])

    super_final_results = {key:None for key in final_results.keys() if key != 'val_trials_classification'}
    for key in final_results.keys():
        if key != 'val_trials_classification':
            mean = np.mean(final_results[key])
            std = np.std(final_results[key])
            super_final_results[key] = {'mean':mean, 'std':std}
    super_final_results = np.array([super_final_results])

    np.savez(open(res_path+'/all_results_'+suffix+'_'+timestr+'.npz','w'), results=results,
             final_results=final_results, super_final_results=super_final_results)
    return super_final_results


def collect_data_intersubjective(subjects, test_subject, channels=[]):

    x_train = []
    y_train = []
    for subject in subjects:
        if subject == test_subject:
            pass
        else:
            x, y, t, tr = load_data(subject)
            x = preprocessing(x)
            x_train.extend(x)
            y_train.extend(y)
            del x,y,t,tr

    x,y,t,tr = load_data(test_subject)
    x = preprocessing(x)
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

    if len(channels):
        x_train = x_train[:,:,channels]
        x_test = x_test[:,:,channels]
    return x_train, y_train, x_test, y_test, t_test, tr_test


def resample_transform(data):

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

        #resampling the data
    n_samples, timepoints, channels = x_train.shape
    x_train = np.reshape(x_train, (n_samples, timepoints * channels))
    ros = RandomOverSampler(random_state=0)
    x_res, y_res = ros.fit_sample(x_train, y_train)
    x_train = np.reshape(x_res, (x_res.shape[0], timepoints, channels))
    y_train = y_res

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

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

def intersubjective_training(data, epochs=5):

    x_train, y_train, x_test, y_test, o_t_test, o_tr_test = data

    global t_test
    t_test = o_t_test
    global tr_test
    tr_test = o_tr_test

    model = branched(x_train.shape)


    adam = Adam(lr=0.001)
    # lrate = LearningRateScheduler(step_decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    m = Metrics()

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')


    history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, shuffle=True, verbose=0,
                       validation_data=(x_test, y_test), callbacks=[m, early_stop])



    probabilities = model.predict(x_test, batch_size=128, verbose=0)
    y_predict = [(round(k)) for k in probabilities]

    cnf_matrix = confusion_matrix(y_test, y_predict)

    return m, history, cnf_matrix



def finetune(model, data, epochs=10, train_trials=40, mode='all'):

    for layer in model.layers[:26]:
        if mode=='all':
            layer.trainable = True
        elif mode=='top':
            layer.trainable = False
        else:
            print 'wrong keyword argument'
            return

    opt = SGD(lr=1e-4, momentum=0.9)

    model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])

    x_test, y_test, o_t_test, o_tr_test = data

    segments = train_trials * 60

    x_train = x_test[0:segments]
    y_train = y_test[0:segments]
    x_test = x_test[segments:]
    y_test = y_test[segments:]
    global t_test
    t_test = o_t_test[segments:]
    global tr_test
    tr_test = o_tr_test[segments:]
    #resampling the data
    n_samples, timepoints, channels, z = x_train.shape
    x_train = np.reshape(x_train, (n_samples, timepoints * channels))
    ros = RandomOverSampler(random_state=0)
    x_res, y_res = ros.fit_sample(x_train, y_train)
    x_train = np.reshape(x_res, (x_res.shape[0], timepoints, channels))
    y_train = y_res

    x_train = np.expand_dims(x_train, axis=3)

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')


    m = Metrics()

    history = model.fit(x_train, y_train, batch_size=16, epochs=epochs, shuffle=True, verbose=0,
                       validation_data=(x_test, y_test), callbacks=[m, early_stop])


    probabilities = model.predict(x_test, batch_size=128, verbose=0)
    y_predict = [(round(k)) for k in probabilities]

    cnf_matrix = confusion_matrix(y_test, y_predict)


    return m, history, cnf_matrix
