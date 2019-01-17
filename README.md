# basic-FNE

Simple code for generating the Full-Network Embedding (FNE) of a dataset using a pre-trained convolutional neural network (CNN) model, as defined in [1].

The Full-Network Embedding is composed by two main steps. First, it extracts the neural activations of a given input using all convolutional or dense layers of the network, thus integrating information at different levels of abstraction (depending on layer depth) and combining different types of neural representations (convolutional and dense layers). Secondly, the FNE contextualizes and leverages this information through a three-valued discretisation step. While the contextualization provides a problem-specific characterisation of the data, the discretization reduces noise and regularises the embedding space. Significantly, this may also reduce the computational cost of processing the resultant representations through methods like SVMs. The FNE is shown to outperform single layer embeddings on several image classification tasks, while also being more robust to the choice of the pre-trained model used as the transfer source. See [1] for more details. The FNE can also be integrated into many multimodal embedding schemes, providing a boost in performance [2].

All FNE functionalities are coded within the "full_network_embedding" method, which extracts and postprocesses the activations of a list of images, as these are feed-forwarded through a given pre-trained CNN. The code contains an example of use, loading a pre-trained model provided by the tensorflow.lucid package.

A straight-forward application of this code is image classification, often through the training of a linear SVM [1]. The FNE is particularly competitive in contexts with little data availability, working with as little as 10 examples per class. The FNE is also robust to dissimilarities between the pre-trained task and the target task, making it apropriate for unusual or highly specific problems.

## Requirements?
* Python. Tested in 2.7.12 and 3.5.2. Should work on most versions.
* Numpy. Tested in 1.14.2. Should work on most versions.
* Tensorflow. Tested in 1.4.0. Should work on most later versions.
* Lucid. Needed for the example case only.
* OpenCV. Tested in 3.2.0.

## Reference

If you find this code useful, please reference it as:

[1] D. Garcia-Gasulla et al., "An Out-of-the-box Full-Network Embedding for Convolutional Neural Networks," 2018 IEEE International Conference on Big Knowledge (ICBK), Singapore, 2018, pp. 168-175.
doi: 10.1109/ICBK.2018.00030

```
@INPROCEEDINGS{8588789,
author={D. Garcia-Gasulla and A. Vilalta and F. Parés and E. Ayguadé and J. Labarta and U. Cortés and T. Suzumura},
booktitle={2018 IEEE International Conference on Big Knowledge (ICBK)},
title={An Out-of-the-box Full-Network Embedding for Convolutional Neural Networks},
year={2018},
pages={168-175},
keywords={Feature extraction;Training;Computational modeling;Task analysis;Space exploration;Tuning;Transfer Learning, Feature Extraction, Embedding Spaces},
doi={10.1109/ICBK.2018.00030},
month={Nov},}
```
[2] Vilalta, Armand, et al. "Studying the impact of the Full-Network embedding on multimodal pipelines." Semantic Web Preprint: 1-15.

## License
GNU General Public License v2.0

