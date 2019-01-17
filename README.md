# basic-FNE

Basic code for generating the Full-Network Embedding of a dataset using a pre-trained convolutional neural network (CNN) model, as introduced in [1].

The Full-Network embedding is based on two main points. First, the use of all layers of the network, integrating activations from different levels of information and from different types of layers (.ie convolutional and fully connected). Second, the contextualisation and leverage of information based on a novel three-valued discretisation method. The former provides extra information useful to extend the characterisation of data, while the later reduces noise and regularises the embedding space. Significantly, this also reduces the computational cost of processing the resultant representations. The proposed method is shown to outperform single layer embeddings on several image classification tasks, while also being more robust to the choice of the pre-trained model used as the transfer source.
The present code is esssentially the "full_network_embedding" method, which extracts and postprocesses the activations of a list of images, as these are feed-forwarded through a given pre-trained CNN.
The code contains an example of use, loading a pre-trained model provided by the tensorflow.lucid package.

A straight-forward application of this code is image classification, often through the training of a linear SVM [1]. The FNE is particularly competitive in contexts with little data availability, working with as little as 10 examples per class. The FNE is also robust to dissimilarities between the pre-trained task and the target task, making it apropriate for unsual or highly specific problems.

[1] https://arxiv.org/abs/1705.07706

If you find this code usefull, please reference it as:
D. Garcia-Gasulla et al., "An Out-of-the-box Full-Network Embedding for Convolutional Neural Networks," 2018 IEEE International Conference on Big Knowledge (ICBK), Singapore, 2018, pp. 168-175.
doi: 10.1109/ICBK.2018.00030

```
@INPROCEEDINGS{8588789,
author={D. Garcia-Gasulla and A. Vilalta and F. Parés and E. Ayguadé and J. Labarta and U. Cortés and T. Suzumura},
booktitle={2018 IEEE International Conference on Big Knowledge (ICBK)},
title={An Out-of-the-box Full-Network Embedding for Convolutional Neural Networks},
year={2018},
volume={},
number={},
pages={168-175},
keywords={Feature extraction;Training;Computational modeling;Task analysis;Space exploration;Tuning;Transfer Learning, Feature Extraction, Embedding Spaces},
doi={10.1109/ICBK.2018.00030},
ISSN={},
month={Nov},}
```

