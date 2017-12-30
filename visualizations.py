from src import collect_data_intersubjective, transform
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
from keras.models import load_model
import numpy as np
from os import listdir
from os.path import isfile, join
from vis.visualization import get_num_filters
from deepviz.saliency import GradientSaliency
from deepviz.guided_backprop import GuidedBackprop

## this script visualize the saliency maps and activation maximization maps for thr trained models.

# define a function to show and/or save the maps.
def show_save_image(matrix, prefix='', actions=['save']):
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(20,12)
    if prefix.startswith('saliency'):
        cax = ax.imshow(np.swapaxes(matrix, 0, 1)[:,:,0], cmap='jet', vmin=-0.015, vmax=0.015)
    elif prefix.startswith('activation'):
        cax = ax.imshow(np.swapaxes(matrix, 0, 1)[:,:,0], cmap='jet', vmin=-4, vmax=4)
    ax.set_yticklabels(eeglabel)
    ax.set_yticks(range(0,len(eeglabel)))
    ax.set_title(prefix)
    ax.set_xticklabels(range(-200,801, 100))
    cbar = fig.colorbar(cax)
    ax.set_aspect('auto')
    ax.set_xlabel('time')
    ax.set_ylabel('channel')
    if 'save' in actions:
        fig.savefig('plots/'+prefix+'.png')
    if 'show' in actions:
        plt.show()
    plt.close()




subjects = [name for name in listdir("./data/50/subjects/")]
path = './models/subjects/'
# defining channels names
eeglabel =  ['Fz','Cz','Pz','Oz','Iz','Fp1','Fp2','F3','F4','F7','F8','T7','T8','C3','C4','P3','P4','O9','O10','P7',
             'P8','FC1','FC2','CP1','CP2','PO3','PO4','PO7','PO8','LMAST']
# defining the experiments that need to be visualized to get their models from the models directory.
trial_names = ['intersubjective_eegnet', 'intersubjective_ft_10eegnet', 'intersubjective_ft_20eegnet', 'intersubjective_ft_30eegnet',
              'intersubjective_ft_50eegnet', 'intersubjective_ft_70eegnet']

all_grads = np.empty((len(subjects),41,30,1))

for n, test_subject in enumerate(subjects):
    print 'WORKING ON', test_subject
    
    # loading the subject data
    x_train, y_train, x_test, y_test, o_t_test, o_tr_test = collect_data_intersubjective(subjects, test_subject)
    x_train, y_train, x_test, y_test = transform((x_train, y_train, x_test, y_test))
    
    # calculating the minimum and maximum values for activation maximization calculations
    min_values = np.amin(np.min(x_train, axis=1), axis=1)
    max_values = np.amax(np.amax(x_train, axis=1), axis=1)

    min_value = np.mean(min_values)
    max_value = np.mean(max_values)

    # create the maps fpr each model/experiment
    for trial_name in trial_names:

        files = [f for f in listdir(join(path, test_subject)) if isfile(join(path, test_subject, f))]
        myfile = [file for file in files if trial_name in file ][0]
        
        model = load_model(join(path, test_subject, myfile))
        # creating the saliency maps only for the models that are not finetuned. (just a personal choice)
         if '_ft_' not in trial_name:
             attended_examples = x_test[y_test==1]
             subj_grads = np.empty(attended_examples.shape+(1,))
             vanilla = GradientSaliency(model)
             # creating a saliency map for each attended example and then averag them.
             for ex in range(attended_examples.shape[0]):
                 example = attended_examples[ex]
                 example = np.expand_dims(example, axis=2)
                 grads = vanilla.get_smoothed_mask(example)
                 subj_grads[ex] = grads
             subj_grads = np.mean(subj_grads, axis=0)
             all_grads[n] = subj_grads
             show_save_image(subj_grads, prefix='saliency_maps/'+test_subject+'_'+trial_name, actions=['save'])


        # computing the activation maximization maps
        layer_idx = -1
        filters = np.arange(get_num_filters(model.layers[layer_idx]))
        for filter_idx in filters:
            # changing the sigmoid activation to linear.
            model.layers[layer_idx].activation = activations.linear
            model = utils.apply_modifications(model)
            img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=(min_value,max_value))
            show_save_image(img, prefix='activation_maximization/'+test_subject+'_'+trial_name, actions=['save'])



# saving the numerical values of the saliency maps of all subjects as it will be used later to decide the most significant
#channels

np.save(open('plots/saliency_maps/all_grads_eegnet.npy', 'w'), all_grads)
