

import os
import time
import mat73
import re
from types import ModuleType

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst
from scipy import io
from PIL import Image

import mxnet as mx
from mxnet import autograd, gluon, nd, init
from mxnet.gluon import loss, nn
from mxnet.gluon.nn import Dense, HybridBlock, HybridSequential
from mxnet.io import NDArrayIter
from mxnet.ndarray import NDArray

from matplotlib.patches import Rectangle, Circle
import math

import e_modules
from tqdm import tqdm

import gc




np.random.seed(37)




class Affine(HybridBlock):
    """Affine layer, applies an affine transformation based on parameters theta
    
    Constrains the affine parameters and calculates the transformation matrix. 
    Applies the transformation matrix on the image and downsamples it.Generates 
    a grid with size based on self.fov, and uses a billinear sampler to 
    interpolate and thus downscale the image while retaining as much information 
    as possible.

    attributes:
        fov: field of view, the size of the receptive field. Depends on the
          layer of the convolutional neural network (must result in a scalar).
        theta: affine parameters, consists of 3 values: scale, translation of x,
          and translation of the y-axis.
    """

    def __init__(self, fov) -> None:
        """Initializes theta as empty and fov by the provided parameter."""
        super().__init__()
        self.fov = fov
        with self.name_scope():
            self.theta = self.params.get("theta", shape=(1, 3))

    def hybrid_forward(self, F: ModuleType, x: NDArray, theta: NDArray) -> NDArray:
        """Forward pass through the layer.
        
        Constraints the affine parameters by applying appropriate activation 
        functions. Scaling is only possible in the range (0, 1], and translation
        depends on the scaling (the further you zoom in the more you can shift).

        return: downsampled image that is transformed according to theta
        """
        theta = theta[0]
        s = F.Activation(theta[0], "sigmoid")
        t = F.Activation(F.Concat(theta[1], theta[2], dim=0), "tanh") * (1 - s)
        _theta = F.Concat(s, nd.zeros(1), t[0], nd.zeros(1), s, t[1], dim=0)[None]

        _image = F.BilinearSampler(x, F.GridGenerator(F.broadcast_to(_theta, (x.shape[0], 6)), "affine", (self.fov, self.fov)))
        return _image


class Alexnet(HybridBlock):
    """Implementation of Alexnet with pre-trained parameters,
        as taken from mxnet's model_zoo"""
    def __init__(self, layer: int, ctx) -> None:
        super(Alexnet, self).__init__()
        self.fov = [16, 32, 32, 32, 64, 224, 224, 224]
        self.layer = layer  # starting layer
        with self.name_scope():
            self.features = nn.HybridSequential("")
            with self.features.name_scope():
                if layer >= 1:
                    self.features.add(nn.Conv2D(64, 11, 4, 2, activation="relu"))
                    self.features.add(nn.MaxPool2D(3, 2))
                if layer >= 2:
                    self.features.add(nn.Conv2D(192, 5, padding=2, activation="relu"))
                    self.features.add(nn.MaxPool2D(3, 2))
                if layer >= 3:
                    self.features.add(nn.Conv2D(384, 3, padding=1, activation="relu"))
                if layer >= 4:
                    self.features.add(nn.Conv2D(256, 3, padding=1, activation="relu"))
                if layer >= 5:
                    self.features.add(nn.Conv2D(256, 3, padding=1, activation="relu"))
                    self.features.add(nn.MaxPool2D(3, 2))
                if layer >= 6:
                    self.features.add(nn.Flatten())
                    self.features.add(nn.Dense(4096, activation="relu"))
                    self.features.add(nn.Dropout(0.5))
                if layer >= 7:
                    self.features.add(nn.Dense(4096, activation="relu"))
                    self.features.add(nn.Dropout(0.5))
            if layer == 8:
                self.output = nn.Dense(1000)
        self.load_parameters("/home/lynn/.mxnet/models/alexnet-44335d1f.params", ctx=ctx, ignore_extra=True)

    def hybrid_forward(self, F: ModuleType, x: nd.NDArray) -> nd.NDArray:
        """Forward pass through the model, outputs the computed features."""
        return self.output(self.features(x)) if self.layer == 8 else self.features(x)


class Response(HybridSequential):
    """Fully connected linear layer to go from feature space to response space.
        Predicts the output of a single voxel."""
    def __init__(self) -> None:
        super().__init__()

        with self.name_scope():
            self.add(Dense(1))


class AFRT(HybridSequential):
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
        self._feature = [Alexnet(layer, ctx)]

        with self.name_scope():
            self.add(Affine(self.fov[layer - 1]))
            self.add(Response())
        
    def hybrid_forward(self, F: ModuleType, x: nd.NDArray) -> nd.NDArray:
        """applies Affine layer, feature model, and response layer"""
        return self[1](self._feature[0](self[0](x)))  # original


class AFRTInit(init.Initializer):
    """Initializer for the AFRT model.
    
    Checks which layer is called using regular expressions.
    Initialisation of various layers:
      -Affine layer:
         -theta: [5, 0, 0], will result in scale = 0.9933, translation = 0, 0
      -Response layer:
         -weight: 1
         -bias: 0
    Response layer might later be restored to default: uniform [-0.07, 0.07]
    """
    
    def _init_weight(self, name, data):
        """Overridden initialisation function."""
        if re.search("afrt\d+_affine0_theta", name):
            data[:] = nd.array([1, 0, 0])

def preprocess(x_list):
    """Preprocess images for Alexnet by performing the following operations:
        -divide by 255.
        -zscore according to Alexnet's expected distribution.
        
    parameters:
        x_list: list containing images of shape (width, height, channel).
    
    returns: pre-processed images as numpy arrays.
    """    

    mean = np.array((0.485, 0.456, 0.406))[None, None].transpose(2, 0, 1)
    std = np.array((0.229, 0.224, 0.225))[None, None].transpose(2, 0, 1)

    return ((x_list / 255) - mean) / std

def ssecorperp(x_list):
    """Reverse function of preprocess, performs the following operations:
        -reverse zscore (multiply by std and add mean)
        -multiply by 255
        -clip values between 0 and 255
        -convert to integer

    parameters:
        x_list: numpy array containing images of shape (width, height, channel)
        
    returns: reverse pre-processed images as numpy arrays in a list
    """
    mean = np.array((0.485, 0.456, 0.406))[None, None].transpose(2, 0, 1)
    std = np.array((0.229, 0.224, 0.225))[None, None].transpose(2, 0, 1)
    
    return [np.uint8(np.clip(((x * std + mean) * 255), 0, 255)) for x in x_list]

def load_fname(path):
    """os.scandir does not sort by name, need to sort manually
    -regex check to prevent non-image files from messing things up"""
    files = sorted(os.scandir(path), key = lambda x: x.name)
    return [os.path.join(path, x.name) for x in files if re.search(".+\.bmp", x.name)]

def load_dataset(x, t, batch_size, shuffle = True):
    return NDArrayIter({ "x": x }, { "t": t }, batch_size, shuffle)


def roi_to_nan_electrodes(roi):
        
    '''
    
    Keep in mind its different than the ones without nan
    
    '''

    electrodes_v1         = [*(range(1,9))] # electrode #6 doesnt exist
    electrodes_v4         = [*(range(9,13))]
    electrodes_IT         = [*(range(13,17))]

    roi_dic               = {'V1':electrodes_v1, 'V4':electrodes_v4, 'IT': electrodes_IT }

    roi_electrodes            = []

    for r in roi:
        roi_electrodes.extend(roi_dic[r])
    
    return roi_electrodes




def electrodes_to_nan_channels(electrodes):
    
    '''
    
    Must be a list, for the monkey data, you can choose between V1, V4 or IT
    
    '''
    
    amount_of_elecs       = [64] * 16

    amount_of_elecs_0     = [0] + amount_of_elecs
    
    list_of_channels_e    = [[*range(np.cumsum(amount_of_elecs_0)[x], np.cumsum(amount_of_elecs)[x])] for x in range(16)]

    elecs_dic             = {
        '1':list_of_channels_e[0],
        '2':list_of_channels_e[1],
        '3':list_of_channels_e[2],
        '4':list_of_channels_e[3],
        '5':list_of_channels_e[4],
        '6':list_of_channels_e[5],
        '7':list_of_channels_e[6],
        '8':list_of_channels_e[7],
        '9':list_of_channels_e[8],
        '10':list_of_channels_e[9],
        '11':list_of_channels_e[10],
        '12':list_of_channels_e[11],
        '13':list_of_channels_e[12],
        '14':list_of_channels_e[13],
        '15':list_of_channels_e[14],
        '16':list_of_channels_e[15]
    }

    electrodes_channels   = []

    for e in electrodes:
        electrodes_channels.extend(elecs_dic[e])
    
    return electrodes_channels


def load_fname(path):
    """os.scandir does not sort by name, need to sort manually
    -regex check to prevent non-image files from messing things up"""
    files = sorted(os.scandir(path), key = lambda x: x.name)
    return [os.path.join(path, x.name) for x in files if re.search(".+\.bmp", x.name)]



if __name__ == '__main__':
    train_path = '/home/lynn/projects/monkey_b2p/THINGS_train'
    test_path = '/home/lynn/projects/monkey_b2p/THINGS_test'
    train_files = load_fname(train_path)
    test_files = load_fname(test_path)

    size = (96, 96)
    stimuli_train = []
    stimuli_test = []
    start = time.time()

    for i, filename in enumerate(train_files):
        img = Image.open(filename)
        img.thumbnail(size)
        stimuli_train.append(preprocess(np.array(img).astype(np.float16).transpose(2, 0, 1)))

    # Load test data
    for i, filename in enumerate(test_files):
        img = Image.open(filename)
        img.thumbnail(size)
        stimuli_test.append(preprocess(np.array(img).astype(np.float16).transpose(2, 0, 1)))

    ctx = mx.gpu(2)
    
    # Load response data and receptive field correlations
    data = mat73.loadmat("../THINGS_normMUA.mat")

    indices = [i for i, d in enumerate(data['reliab'].mean(1)) if d > 0.4]

    V1_channels = electrodes_to_nan_channels([str(i) for i in roi_to_nan_electrodes(['V1'])])
    V4_channels = electrodes_to_nan_channels([str(i) for i in roi_to_nan_electrodes(['V4'])])
    IT_channels = electrodes_to_nan_channels([str(i) for i in roi_to_nan_electrodes(['IT'])])

    # separate into brain areas; still sorted from before
    v1_indices = [x for x in indices if x in V1_channels]
    v4_indices = [x for x in indices if x in V4_channels]
    it_indices = [x for x in indices if x in IT_channels]

    n_epochs = 100
    runname = 'lynn_run_samedata_repeat_deeper_layers'
    os.makedirs(f'models/{runname}', exist_ok = True)
    batch_size = 20

    for roi in [
        'V1',
        'V4',
        'IT'
    ]:
        # old and new indices
        if roi == 'V1':
            print(roi.upper())
            ori_ind = v1_indices
            new_ind = np.arange(len(v1_indices))
        elif roi == 'V4':
            print(roi.upper())
            ori_ind = v4_indices
            new_ind = np.arange(len(v1_indices), len(v1_indices) + len(v4_indices))
        elif roi == 'IT':
            print(roi.upper())
            ori_ind = it_indices
            new_ind = np.arange(len(v1_indices) + len(v4_indices), len(v1_indices) + len(v4_indices) + len(it_indices))
        else:
            print('no matching ROI')
            new_ind = False
            
            
        roi_channels_train = data['train_MUA'][ori_ind]
        roi_channels_test = data['test_MUA'][ori_ind]

        all_cors = []
        best_corrs = np.zeros(len(ori_ind))
        for i,c in enumerate(new_ind):

            brain_train = roi_channels_train[i]
            brain_test = roi_channels_test[i]

            training_iterator = load_dataset(nd.array(stimuli_train, ctx=ctx), nd.array(brain_train, ctx=ctx), batch_size, shuffle=True) # perhaps change it later to shuffle?
            testing_iterator = load_dataset(nd.array(stimuli_test, ctx=ctx), nd.array(brain_test, ctx=ctx), batch_size) # perhaps change it later to shuffle?
            len_train_iterator = len(stimuli_train)/batch_size
            len_test_iterator = len(stimuli_test)/batch_size

            cors = []
            # Train a model per layer
            for layer in [
                1, 
                2,
                3, 
                4, 
                5
            ]:
                # ----------------------- TRAINING LOOP FOR EACH CHANNEL HEREEEE -----------------------

                # Define model
                mse = gluon.loss.L2Loss()
                model = AFRT(layer, ctx)
                model.initialize(init=AFRTInit(), ctx=ctx)
                trainer = gluon.Trainer(model.collect_params(), "Adam", {"learning_rate": 0.002})

                test_losses = []
                train_losses = []
                for epoch in tqdm(range(n_epochs), total = n_epochs):
                    train_loss_mean, test_loss_mean = 0., 0.

                    # TRAINING LOOP # 
                    for batch in training_iterator:
                        
                        with autograd.record():
                            x_train = batch.data[0]
                            t_train = batch.label[0]
                            y_train = model(x_train)
                            train_loss = mse(y_train, t_train)

                        train_loss.backward()
                        trainer.step(batch_size)
                        train_loss_mean += train_loss.mean().asscalar()
                        
                    _test_losses = []
                    # TESTING LOOP #
                    for batch in testing_iterator:
                        x_test = batch.data[0]
                        t_test = batch.label[0]
                        y_test = model(x_test)
                        test_loss = mse(y_test, t_test)
                        test_loss_mean += test_loss.mean().asscalar()
                    
                        
                    train_losses.append(train_loss_mean/ len_train_iterator)
                    test_losses.append(test_loss_mean/ len_test_iterator)
                    
                    training_iterator.reset()
                    testing_iterator.reset()
                    
                # Save model, specify brain area in the name
                model.save_parameters(f"models/{runname}/{roi}_v{c}_l{layer}.params")
                os.makedirs(f"models/losses/{runname}/", exist_ok = True)
                np.save(f"models/losses/{runname}/{roi}_v{c}_l{layer}_test_loss.npy", np.array(test_losses))
                np.save(f"models/losses/{runname}/{roi}_v{c}_l{layer}_train_loss.npy", np.array(train_losses))
                
                print(f'end of channel {c}, layer {layer}')
                del model
                gc.collect()
 