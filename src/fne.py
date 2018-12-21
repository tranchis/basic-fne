import os
import numpy as np
import tensorflow as tf
import lucid.modelzoo.vision_models as models
import cv2


def full_network_embedding(model, image_paths, input_tensor, target_tensors, stats=np.empty((0, 0))):
    ''' 
    Generates the Full-Network embedding[1] of a list of images using a pre-trained
    model (input parameter model) with its computational graph loaded. Tensors used 
    to compose the FNE are defined by target_tensors input parameter. The input_tensor
    input parameter defines where the input is fed to the model.

    By default, the statistics used to standardize are the ones provided by the same 
    dataset we wish to compute the FNE for. Alternatively these can be passed through
    the stats input parameter.

    This function aims to generate the Full-Network embedding in an illustrative way.
    We are aware that it is possible to integrate everything in a tensorflow operation,
    however this is not our current goal.

    [1] https://arxiv.org/abs/1705.07706
   
    Args:
        model (tf.GraphDef): Serialized TensorFlow protocol buffer (GraphDef) containing the pre-trained model graph
                             from where to extract the FNE. You can get corresponding tf.GraphDef from default Graph
                             using `tf.Graph.as_graph_def`.
        image_paths (list(str)): List of images to generate the FNE for.
        input_tensor (str): Name of tensor from model where the input is fed to
        target_tensors (list(str)): List of tensor names from model to extract features from.
        stats (2D ndarray): Array of feature-wise means and stddevs for standardization.

    Returns:
       2D ndarray : List of features per image. Of shape <num_imgs,num_feats>
       2D ndarry: Mean and stddev per feature. Of shape <2,num_feats>
    '''
    # Just in case, we reset the graph.
    tf.reset_default_graph()
    tf.import_graph_def(model)

    # Prepare output variable
    len_features = 0
    for tensor_name in target_tensors:
        t_tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
        len_features += t_tensor.get_shape().as_list()[-1]
    features = np.empty((len(image_paths), len_features))
    # Prepare tensors to capture
    x0 = tf.get_default_graph().get_tensor_by_name(input_tensor)
    tensorOutputs = []
    for tname in target_tensors:
        t = tf.get_default_graph().get_tensor_by_name(tname)
        tensorOutputs.append(t)
    # Extract features
    with tf.Session() as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])
        for idx, img_path in enumerate(image_paths):
            # Load image and swap channels from cv2's BGR to tf's RGB
            img = np.asarray(cv2.imread(img_path), dtype=np.float32)[:, :, ::-1]
            # TODO: images are resized? how is it matched to the input?
            # Extract features, keep only first instance (current image)
            # TODO: extract by batches for minimal efficiency?
            feature_vals = sess.run(tensorOutputs, feed_dict={x0: np.expand_dims(img, 0)})
            features_current = np.empty((1, 0))
            for feat in feature_vals:
                # SPATIAL AVERAGE POOLING
                pooled_vals = np.mean(np.mean(feat, axis=2), axis=1)
                features_current = np.concatenate((features_current, pooled_vals), axis=1)
            # Store in position
            features[idx] = features_current.copy()
    # STANDARDIZATION STEP
    # Compute statistics if needed
    if len(stats) == 0:
        stats = np.zeros((2, len_features))
        stats[0, :] = np.mean(features, axis=0)
        stats[1, :] = np.std(features, axis=0)
    # Apply statistics, avoiding nans after division by zero
    features = np.divide(features - stats[0], stats[1], out=np.zeros_like(features), where=stats[1] != 0)
    if len(np.argwhere(np.isnan(features))) != 0:
        raise Exception('There are nan values after standardization!')
    # DISCRETIZATION STEP
    th_pos = 0.15
    th_neg = -0.25
    features[features > th_pos] = 1
    features[features < th_neg] = -1
    features[[(features >= th_neg) & (features <= th_pos)][0]] = 0

    # Store output
    outputs_path = '../outputs'
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)
    np.savez(open(os.path.join(outputs_path, 'fne.pkl'), 'wb'), features)
    np.savez(open(os.path.join(outputs_path, 'stats.pkl'), 'wb'), stats)

    # Load output
    # fne = np.load('fne.pkl')['arr_0']
    # fne_stats = np.load('stats.pkl')['arr_0']

    # Return
    return features, stats


if __name__ == '__main__':
    # This shows an example of calling the full_network_embedding method using
    # the InceptionV1 architecture pretrained on ILSVRC2012 (aka ImageNet), as
    # provided by the tensorflow lucid package. Using any other pretrained CNN
    # model is straightforward.

    # Load model
    model = models.InceptionV1()
    model.load_graphdef()

    # Define input and target tensors
    input_tensor = 'import/input:0'
    target_tensors = ['import/mixed3a:0', 'import/mixed3b:0', 'import/mixed4a:0', 'import/mixed4b:0',
                      'import/mixed4c:0', 'import/mixed4d:0', 'import/mixed4e:0', 'import/mixed5a:0',
                      'import/mixed5b:0']

    # Define images to process
    image_paths = ['../images/img1.jpg', '../images/img2.jpg', '../images/img3.jpg']

    # Call FNE method
    fne_features, fne_stats = full_network_embedding(model.graph_def, image_paths, input_tensor, target_tensors)
