import sys
import time
from PIL import Image
import numpy as np
import mat73
import mxnet as mx
from mxnet import nd
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
import os
import scipy.stats as sst

sys.path.insert(0, '../run')
import run_exact_samedata as samedata
class AFRT(samedata.HybridSequential):
    """Implementation of the overarching AFRT model 
      consists of three layers: affine layer -> feature model -> response layer
      affine and response layers are learned, feature model is fixed.

    Attributes:
        fov: field of view, the size of the receptive field. Is passed to the 
        Affine layer
        _feature: the feature model, a convolutional neural network that 
        generates features. Fixed model, parameters are not learned. Easiest 
        (but slightly dirty) way is to put it in a list and make it an attribute
    """

    def __init__(self, layer: int, ctx) -> None:
        """Initializes fov and the different model layers
        
        Parameters:
            layer: starting layer of the feature model to be analysed.
        """
        super().__init__()
        self.fov = [16, 32, 32, 32, 64, 224, 224, 224]
        self.selected_fov = self.fov[layer - 1]
        self._feature = [samedata.Alexnet(layer, ctx)]
        # self._feature = [feature_model]
        with self.name_scope():
            self.add(samedata.Affine(self.selected_fov))
            self.add(samedata.Response())
        
    def hybrid_forward(self, F: samedata.ModuleType, x: samedata.nd.NDArray) -> samedata.nd.NDArray:
        """applies Affine layer, feature model, and response layer"""
        affine_out = self[0](x)
        feature_out=self._feature[0](affine_out)
        y = self[1](feature_out) 
        return y, feature_out, affine_out   # original


def check_run(runname, threshold=0.4, roi=None):
    data = mat73.loadmat("/home/lynn/projects/bin/monkey_b2p/THINGS_normMUA.mat")
    indices = [i for i, d in enumerate(data['reliab'].mean(1)) if d > threshold]
    roi_channels = samedata.electrodes_to_nan_channels([str(i) for i in samedata.roi_to_nan_electrodes([roi])])
    roi_indices = [x for x in indices if x in roi_channels]
    list_roi = [int(i[len(i)-11:-10]) for i in glob(f'../run/models/{runname}/{roi}*')]
    print(f'\n{runname.upper()}: ')
    print(f'{roi}: {len(list_roi)} out of {len(roi_channels)*5} finished..\n')
    return int(len(list_roi)/5)

runname = 'AFRT'
threshold = 0.4
reliab_to_check = 0.4
ctx = mx.gpu(1)
batch_size = 100

test_path = ''
test_files = samedata.load_fname(test_path)
size = (96, 96)
# Load test data
stimuli_test = []
for i, filename in enumerate(test_files):
    img = Image.open(filename)
    img.thumbnail(size)
    stimuli_test.append(samedata.preprocess(np.array(img).astype(np.float16).transpose(2, 0, 1)))


data = mat73.loadmat("../../projects/bin/monkey_b2p/THINGS_normMUA.mat")

indices = [i for i, d in enumerate(data['reliab'].mean(1)) if d > threshold]

V1_channels = samedata.electrodes_to_nan_channels([str(i) for i in samedata.roi_to_nan_electrodes(['V1'])])
V4_channels = samedata.electrodes_to_nan_channels([str(i) for i in samedata.roi_to_nan_electrodes(['V4'])])
IT_channels = samedata.electrodes_to_nan_channels([str(i) for i in samedata.roi_to_nan_electrodes(['IT'])])

v1_indices = [x for x in indices if x in V1_channels]
v4_indices = [x for x in indices if x in V4_channels]
it_indices = [x for x in indices if x in IT_channels]

indices_of_interest = [i for i, d in enumerate(data['reliab'].mean(1)) if d > 0.4]

for roi in [
    'V1',
    'V4',
    'IT'
]:
    ran_until = check_run(f'{runname}', threshold, roi)

    if roi == 'V1':
        ori_ind = v1_indices
        new_ind = np.arange(len(v1_indices))
    if roi == 'V4':
        ori_ind = v4_indices
        new_ind = np.arange(len(v1_indices), len(v1_indices) + len(v4_indices))
    if roi == 'IT':
        ori_ind = it_indices
        new_ind = np.arange(len(v1_indices) + len(v4_indices), len(v1_indices) + len(v4_indices) + len(it_indices))
        
        
    roi_channels_test = data['test_MUA'][ori_ind]
    
    best = np.zeros(len(ori_ind))
        
    all_cors = []
    reliab = []
            
    best_features_l1 = []
    best_features_l2 = []
    best_features_l3 = []
    best_features_l4 = []
    best_features_l5 = []

    best_affine_features_l1 = []
    best_affine_features_l2 = []
    best_affine_features_l3 = []
    best_affine_features_l4 = []
    best_affine_features_l5 = []

    best_response_weights_l1 = []
    best_response_weights_l2 = []
    best_response_weights_l3 = []
    best_response_weights_l4 = []
    best_response_weights_l5 = []

    best_affine_weights_l1 = []
    best_affine_weights_l2 = []
    best_affine_weights_l3 = []
    best_affine_weights_l4 = []
    best_affine_weights_l5 = []
    
    for i, c in tqdm(enumerate(new_ind[:ran_until]), total= len(new_ind)):
        

        brain_test = roi_channels_test[i]
        
        testing_iterator = samedata.load_dataset(nd.array(stimuli_test, ctx=ctx), nd.array(brain_test, ctx=ctx), batch_size, shuffle = False) # perhaps change it later to shuffle?
        
        cors = []
        
        old_reliab_indice = indices_of_interest[i]

        r = data['reliab'].mean(1)[old_reliab_indice]
        if r > reliab_to_check:

            
            affine_features = []
            all_features = []
            response_weights = []
            affine_weights = []
            for layer in [
                1, 
                2, 
                3, 
                4, 
                5,
                ]:

                model = AFRT(layer, ctx)
                model.initialize(init=samedata.AFRTInit(), ctx=ctx, force_reinit = True)
                model.load_parameters(f'../run/models/{runname}/{roi}_v{c}_l{layer}.params', ctx = ctx)
                response_weights.append(model[1][0].weight.data().asnumpy())
                affine_weights.append(model.collect_params()[model[0].name + '_theta'].data().asnumpy())
                y_test_layer = []
                t_test_layer = []
                for batch in testing_iterator:

                    x_test = batch.data[0]
                    t_test = batch.label[0]
                    y_test, features, affine_feature = model(x_test)

                    y_test_layer.append(y_test.asnumpy()[:,0])
                    t_test_layer.append(t_test.asnumpy())
                    # break
                testing_iterator.reset()

                all_features.append(features.asnumpy())
                affine_features.append(affine_feature.asnumpy())
                pearsonr, _ = sst.pearsonr(np.concatenate(y_test_layer), np.concatenate(t_test_layer))
                cors.append(pearsonr)

            best[i] = np.argmax(cors)
            all_cors.append(cors)

        if np.argmax(cors) == 0:
            best_features_l1.append(all_features[np.argmax(cors)])
            best_affine_features_l1.append(affine_features[np.argmax(cors)])
            best_response_weights_l1.append(response_weights[np.argmax(cors)])
            best_affine_weights_l1.append(affine_weights[np.argmax(cors)])
        if np.argmax(cors) == 1:
            best_features_l2.append(all_features[np.argmax(cors)])
            best_affine_features_l2.append(affine_features[np.argmax(cors)])
            best_response_weights_l2.append(response_weights[np.argmax(cors)])
            best_affine_weights_l2.append(affine_weights[np.argmax(cors)])
    
        if np.argmax(cors) == 2:
            best_features_l3.append(all_features[np.argmax(cors)])
            best_affine_features_l3.append(affine_features[np.argmax(cors)])
            best_response_weights_l3.append(response_weights[np.argmax(cors)])
            best_affine_weights_l3.append(affine_weights[np.argmax(cors)])
    
        if np.argmax(cors) == 3:
            best_features_l4.append(all_features[np.argmax(cors)])
            best_affine_features_l4.append(affine_features[np.argmax(cors)])
            best_response_weights_l4.append(response_weights[np.argmax(cors)])
            best_affine_weights_l4.append(affine_weights[np.argmax(cors)])
    
        if np.argmax(cors) == 4:
            best_features_l5.append(all_features[np.argmax(cors)])
            best_affine_features_l5.append(affine_features[np.argmax(cors)])
            best_response_weights_l5.append(response_weights[np.argmax(cors)])
            best_affine_weights_l5.append(affine_weights[np.argmax(cors)])


    os.makedirs(f'correlations/{runname}', exist_ok = True)
    os.makedirs(f'features/{runname}', exist_ok = True)
    os.makedirs(f'weights/{runname}', exist_ok = True)

    np.save(f'correlations/{runname}/best_{roi}.npy', best)
    np.save(f'correlations/{runname}/corrs_{roi}.npy', np.stack(all_cors))

    np.save(f'features/{runname}/best_features_{roi}_l1.npy', np.array(best_features_l1))
    np.save(f'features/{runname}/best_features_{roi}_l2.npy', np.array(best_features_l2))
    np.save(f'features/{runname}/best_features_{roi}_l3.npy', np.array(best_features_l3))
    np.save(f'features/{runname}/best_features_{roi}_l4.npy', np.array(best_features_l4))
    np.save(f'features/{runname}/best_features_{roi}_l5.npy', np.array(best_features_l5))

    np.save(f'features/{runname}/best_affine_features_{roi}_l1.npy', np.array(best_affine_features_l1))
    np.save(f'features/{runname}/best_affine_features_{roi}_l2.npy', np.array(best_affine_features_l2))
    np.save(f'features/{runname}/best_affine_features_{roi}_l3.npy', np.array(best_affine_features_l3))
    np.save(f'features/{runname}/best_affine_features_{roi}_l4.npy', np.array(best_affine_features_l4))
    np.save(f'features/{runname}/best_affine_features_{roi}_l5.npy', np.array(best_affine_features_l5))

    np.save(f'weights/{runname}/best_response_weights_{roi}_l1.npy', np.array(best_response_weights_l1))
    np.save(f'weights/{runname}/best_response_weights_{roi}_l2.npy', np.array(best_response_weights_l2))
    np.save(f'weights/{runname}/best_response_weights_{roi}_l3.npy', np.array(best_response_weights_l3))
    np.save(f'weights/{runname}/best_response_weights_{roi}_l4.npy', np.array(best_response_weights_l4))
    np.save(f'weights/{runname}/best_response_weights_{roi}_l5.npy', np.array(best_response_weights_l5))
    
    np.save(f'weights/{runname}/best_affine_weights_{roi}_l1.npy', np.array(best_affine_weights_l1))
    np.save(f'weights/{runname}/best_affine_weights_{roi}_l2.npy', np.array(best_affine_weights_l2))
    np.save(f'weights/{runname}/best_affine_weights_{roi}_l3.npy', np.array(best_affine_weights_l3))
    np.save(f'weights/{runname}/best_affine_weights_{roi}_l4.npy', np.array(best_affine_weights_l4))
    np.save(f'weights/{runname}/best_affine_weights_{roi}_l5.npy', np.array(best_affine_weights_l5))

