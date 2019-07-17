## Introduction
**Cervical Cancer:**
<br />
Approximately 5000 women will die due to cervical cancer in North America this year, and while the 5-year survival rate for women with invasive cervical cancer that has been detected early is 91%, this rate drops to 17% detection when detection is delayed. 
Cervical cancer, which continues to be the second most common cancer affecting women worldwide (and most common in developing countries) can be properly addressed if detected by high quality screening. Specifically, identifying the type of transformation zones within the cervix (Type 1, Type 2, Type 3) can help identify which patient requires further testing.  In this paper we will compare two types of neural networks (NNs) used in image classification and measure their performance in classifying three distinct types of cervical transformation zones. The goal of this study is to aid in the early detection process of cervical cancer. 


*Neural Networks for Image Classification*

In a traditional neural network (NN), there are n-number of artificial neurons that are linearly interconnected with each other. An input image is passed from the input end and the NN decides the class of the output based on the weights of the input. Training of an NN means feeding many images of various classes as inputs where these images are all labeled. We can also think of a neural network as a simple mathematical formula x*w=y, where x is your input image, y is the network defined class and w refers to the weight of a single neuron layer. The training process of an NN consists of two parts, forward pass and back propagation. Forward pass is when we give images to the network as input y and the network generates some y' output labels. The difference between y and y', for some loss function, is the error of the NN. In backpropagation, the NN tries to minimize the error by tweaking w. 
However, fully connected neural networks such as these are incredibly expensive to run. Convolutional neural network (CNN) models on the other hand, have been shown to be just as effective in image classification and less computationally expensive . 
While traditional neural networks use fully connected layers, wherein each neuron takes the full layer (the entire dataset) as an input, CNNs use neurons that limit their input to a specific subset of the neuron in the layer above it. 
These partially connected layers reduce the computational complexity of the forward pass, while also significantly reducing the total number of parameters in the network. CNNs are typically used in image processing domains due to their ability 
to provide context to a region of interest. This is in contrast to fully connected NNs, in which the context of an image feature can be lost in the noise of information coming from the entire image. 
Despite this, CNNs are not without limitations. The partially connected aspect of the CNN which gives rise to the sub-sampling feature of this model (e.g a picture of a dog is put together by identifying 
smaller features such as a tail and paws), will eventually lose the precise spatial relationships between higher-order correlations (e.g the tail can’t be in front of the paws to be classified a dog). 
Capsule networks, as described by Hinton et. al (2017), was recently introduced as a network that is more robust to detecting how images and objects are positioned and more easily able to generalize these 
spatial relationships. By nesting another layer of neurons within a layer in an NN (a capsule), lower and higher-level capsules can connect to provide better feature detection in lower layers of the image,
and is a much more effective method than the max-pooling mechanism in CNNs (which essentially reduces the spatial size of the image in order to pick the largest feature, considered to be crude). 
While this new methodology has shown to outperform CNNs on the classical MNIST data set of images, we believe that we are the first to apply a capsule network to an extensive medical imaging dataset.  


## Methodology <br />
*Image Dataset* <br />
5000 medical images of cervix types in .jpg format are provided by Intel/Mobile-ODT, with a dataset provided specifically for training with correct class labels, and an additional 511 images for testing. Many raw images include unnecessary items such as surrounding clinical equipment in the frame. Therefore, preprocessing of the images to extract our region-of-interest (ROI) was done for all images. Additionally, some images had a strong green tinge (576 images in total), which could adversely affect training. We trained on both the full datasets and non-green datasets to determine if the greened image influenced accuracy or loss. 
<br />

*Preprocessing Stage with Image Cropping*  <br />
In order to extract our ROI from each cervical image, we employ an image segmentation algorithm described by Greenspan et al. (2009) in cervical imaging research for the National Cancer Institute (NCI). We first remove any circular frames that might be present in the image that are originally captured through the clinical capture methods used by physicians (Figure 1). This is done by utilizing the algorithm for finding the largest rectangle in a histogram, which allows us to crop the largest inscribed rectangle in the frame 3. We then begin the initial delineation of the cervix by using two features: how red a pixel in the image is, and how far that pixel is from the center of the image. The image is then separated into two clusters using K-means procedure, which allows us to approximate the actual region of the cervix within the image. 
<br />
![Fig 1](https://github.com/matthiaslmz/MiscalleanousProj/blob/master/MedicalImageClassifier/fig1.png)
<br />
**Figure 1.** A) A raw image of a Type 1 cervix taken by a clinician. B) Red rectangle denotes the ROI after finding largest inscribed rectangle and using K-means procedure. Original image is resized to 256 x 256 px. C) Final image used for training model
<br />

*Convolutional Neural Network* <br />
There are five main operations in our CNN architecture including the convolution operator, batch normalization, a rectified linear unit (ReLU) operation, a pooling step, and a fully connected layer 5. We have two convolutional layers that are based utilized as part of the PyTorch tutorial with modified parameters 4. The first layer has 3 input image channels, 6 output channels, a 5*5 kernel/filter, and padding of size 2. In addition, the Rectified Linear Unit (ReLU) operation is used after every convolution operation. This is done to reconcile the linear output we receive from the convolution with notion that out image classification function will usually be nonlinear, and we do so by introducing a nonlinear function like ReLU. Furthermore, we also added a 2D-max pooling step with the purpose of reducing the spatial size of the input images that reduces the features and computational complexity of the CNN. In addition, 
the pooling layer is added to prevent the model of overfitting. The training process of a CNN is also complicated by the fact that the distribution of each layer’s input changes due to the change of the layer’s predecessor’s changing weights. This results in a changing distributions of output activations, hence slowing down the training process and requires careful initialization. Thus, batch normalization addresses this problem by continuously taking the output of a particular layer and normalizing it before sending it across to the next layer as input resulting in a faster learning rate and higher accuracy. Lastly, we have a fully 
<br />
![Fig 2](https://github.com/matthiaslmz/MiscalleanousProj/blob/master/MedicalImageClassifier/fig2.png)
<br />
**Figure 2.** A model representation of our convolutional neural network developed with PyTorch.

*Capsule Network*
Our capsule network implementation is primarily based on the barebones CUDA-enabled PyTorch implementation by Kenta Iwasaki of Gram.Ai 6, which is in turn directly based on the CapsNet architecture described by Hinton et. al (2017). We utilize the exact same hyperparameters that are also used by Hinton et. al (2017) to describe a simple and relatively shallow capsule network to understand the scalability of such a model to more complex image data sets.
<br />
![Fig 3](https://github.com/matthiaslmz/MiscalleanousProj/blob/master/MedicalImageClassifier/fig3.png)
<br />
**Figure 3.** A model representation of Hinton’s capsule network developed for MNIST and adapted for out purposes. This shallow capsule network utilizes an initial 256, 9 x 9 convolutional kernels with a second layer containing 32 channels of capsules which each contain 8 convolutional units with 9 x 9 kernels.

*Training Strategy*
We chose to focus on optimizing two hyperparameters in relation to training our models. An epoch defines the number of times a model updates the weights using all of the training vectors, or in our specific case, when one forward pass and one backward pass has occurred on all training examples. A batch size is the number of training examples utilizes in one forward/backward pass. We chose to keep all of our models at batch size = 4 in order to isolate and observe the effect of increasing epochs on our training and validation error, as well as keep our batch size low enough so as to not run into issues regarding memory allocation and space.
We also used a cross-entropy loss function, ![eq1](https://latex.codecogs.com/gif.latex?-%28y%20log%28p%29%20&plus;%20%281%20-%20y%29log%281-p%29%29), for both our CNN and capsule network. This loss function allows us to understand the distance between two probability vectors (the cross-entropy). 
We also used the logloss function ![eq2](https://latex.codecogs.com/gif.latex?-%5Cfrac%7B1%7D%7BN%7D%28%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bj%3D1%7D%5EM%20%5Ccdot%20y_%7Bij%7Dlog%28p_%7Bij%7D%29%29) to evaluate our model on the test-set (a variation on the multi-class log loss function where N is the number of images in the test set, M is the number of categories, y_ij is 1 if the observation i belongs to class j and 0 otherwise, and p_ij is the predicted probability that observation i belongs to class.

**Results**
We first evaluated whether the greened images contributed significantly in training our deep CNN models. As figure X shows, they did not have a substantial effect. Thus, we chose to move forward without the non-green images as any potential benefit in training did not outweigh the computational cost increase. Moving forwards with the non-green data, we compared a CNN trained for 50 epochs with a Capsule Net trained for 50 epochs. The capsule net was trained with a different loss function for compatibility reasons, however even with that caveat it was clearly performing better in accuracy, and the loss was getting better consistently, 
where no such behavior was observed in the CNN. Given that this was a relatively shallow Capsule Net, we were confident that it wouldn’t be overfitting more than a CNN after the same number of epochs, and moved forward with a full test of Capsule Net. Interestingly, the early promising results for CNN’s were a result of it always guessing the dominant class early on, class two. We ran Capsule Net with more epochs until the error rate flattened out, which happened at epoch 200. At that point, it had an impressive validation accuracy of 83.6%. 

![Fig 4](https://github.com/matthiaslmz/MiscalleanousProj/blob/master/MedicalImageClassifier/fig4.png)
<br />
**Figure 4.** Comparison of CNN and Capsule models trained on stratified levels of cleaned training data to evaluate loss value and prediction accuracy. Green, Red and Yellow all depict accuracy, while Blue, Navy Blue, and Grey depict loss. At 50 epochs, capsule network achieved a loss of 0.3072 with a prediction accuracy 59.36% on the test set. The capsule outperformed all three CNN models
<br />
![Fig 5](https://github.com/matthiaslmz/MiscalleanousProj/blob/master/MedicalImageClassifier/fig5.png)
<br />
**Figure 5.** Running 200 epochs on the capsule network on the test set. We found that our loss function continued to be substantially minimized until epoch 200, where we saw a loss value of 0.2026 and an accuracy rate of 83.60%.








