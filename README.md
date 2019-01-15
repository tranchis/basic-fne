# basic-FNE

Basic code for generating the Full-Network Embedding of a dataset using a pre-trained convolutional neural network (CNN) model, as introduced in [1].

Code is esssentially the "full_network_embedding" method, which extracts and postprocesses the activations of a list of images, as these are feed-forwarded through a given pre-trained CNN.
The code contains an example of use, loading a provided by the tensorflow.lucid package.

A straight-forward application of this code is image classification, often through the training of a linear SVM [1]. The FNE is particularly competitive in contexts with little data availability, working with as little as 10 examples per class. The FNE is also robust to dissimilarities between the pre-trained task and the target task, making it apropriate for unsual or highly specific problems.

[1] https://arxiv.org/abs/1705.07706


