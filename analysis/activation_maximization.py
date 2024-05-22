
from mxnet.gluon.nn import Dense, HybridBlock, HybridSequential
import mxnet
from types import ModuleType
from mxnet.ndarray import NDArray
from mxnet import nd, init, autograd, gluon
from mxnet.gluon import nn
from types import ModuleType
import re
import mxnet as mx
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os

class Alexnet(HybridBlock):
    """Implementation of Alexnet with pre-trained parameters,
        as taken from mxnet's model_zoo"""
    def __init__(self, layer: int, ctx: None) -> None:
        super(Alexnet, self).__init__()
        
        self.fov = [224, 224, 224, 224, 224, 224, 224, 224]
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
    
def create_initial_image(shape, ctx):
    return nd.random.uniform(shape=shape, ctx=ctx)
    
def optimize_image(image, model, neuron_index, iterations=10000, lr=0.03, maximize=True):
    for i in tqdm(range(iterations)):
        image.attach_grad()
        with autograd.record():
            output = model(image)[0, neuron_index]
            activation_loss = output.mean() if maximize else -output.mean()
            loss = activation_loss 

        loss.backward()

        grad_norm = nd.norm(image.grad) + 3**-10
        image += lr * image.grad / grad_norm
    return image
os.makedirs('alexnet_neurons', exist_ok=True)

ctx = mx.gpu(0)  
for layer in [1,2,3,4,5]:
    
    initial_image = create_initial_image((1, 3, 224, 224), ctx = ctx)

    model = Alexnet(layer, ctx)    
    output = model(initial_image)
    layer_images = []
    for neuron_index in range(output.shape[1]):
        initial_image = create_initial_image((1, 3, 224, 224), ctx = ctx)
        model = Alexnet(layer, ctx)    
        optimized_image = optimize_image(initial_image,  model, neuron_index, maximize=True)  # Set maximize to False for negative optimization
        layer_images.append(optimized_image.asnumpy())
        np.save(f'alexnet_neurons/layer_image_latest.npy', optimized_image.asnumpy())
    layer_images = np.concatenate(layer_images)
    np.save(f'alexnet_neurons/layer{layer}_images.npy', layer_images)

