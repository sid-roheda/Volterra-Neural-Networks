
Code release for [Volterra Neural Networks (VNNs)](https://arxiv.org/abs/1910.09616) and [Conquering the cnn over-parameterization dilemma: A volterra filtering approach for action recognition](https://ojs.aaai.org/index.php/AAAI/article/view/6870).  

# Training
Configure your dataset path in mypath.py.

You can choose different models and datasets in train_VNN_fusion_highQ.py. run 'python3 train_VNN_fusion_highQ.py' to start VNN training from scratch.

Use 'networks/vnn_rgb_of_complex.py' as model architecture to train complex model of VNN from scratch.

Pre-processing of video frames will be done the first time the code is run and will most likely take a long time but is a one time process.

The current implementation performs computation of Optical Flow in real time. Another option of potentially improving performance is to store pre-calculated dense optical flow on the hard drive. This will save computation time.

VNN model is always trained from scratch in this implementation.

################################################## <br />

Pre-Trained Model to be added soon.
