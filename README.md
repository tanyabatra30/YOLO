OBJECT DETECTION USING YOLO

You Only Look Once (YOLO) is a real-time object detection algorithm that detects and locates objects in images or video frames. It uses a single convolutional neural network (CNN) architecture called Darknet to predict the bounding boxes and class probabilities of objects present in an input image.

The Darknet architecture consists of a series of convolutional layers, with skip connections to preserve spatial information. The network begins with an input image and passes it through convolutional layers to extract features. These features are then processed by several fully connected layers to make predictions about the objects present in the image.

ADVANTAGES OF YOLO

Speed: It is one of the fastest object detection systems, capable of processing images in real time. This makes it ideal for applications such as self-driving cars, surveillance cameras, and robotics.

Accuracy: It achieves high accuracy in object detection tasks, especially for small objects, which are challenging for other object detection systems.

Simplicity: It uses a single neural network for object detection and classification, which makes it simpler than other systems that use multiple networks.

Flexibility: It can be trained to detect a wide variety of objects, including people, vehicles, animals, and more.

Open-source: It is an open-source project, which means that its source code is freely available and can be modified to suit specific applications.

YOLO BACKGROUND WORKING
Preprocessing::The input image is resized and normalized to a fixed size.

CNN Architecture: YOLO uses a deep convolutional neural network to process the image and predict the object bounding boxes and class probabilities. The network consists of 24 convolutional layers followed by 2 fully connected layers.

Feature Extraction: The network processes the input image and extracts features at different scales and resolutions. The output of the last convolutional layer is a feature map that captures the spatial information of the image.

Object Detection: The network divides the feature map into a grid of cells and predicts the bounding boxes and class probabilities for each cell. Each cell predicts a fixed number of bounding boxes and confidence scores for each class.

Non-max Suppression: The predicted bounding boxes are filtered using non-max suppression to remove overlapping boxes with low confidence scores.

Post-processing: The remaining boxes are then thresholded to remove boxes with low confidence scores and the final boxes are returned as the predicted object detections.

CONFIDENCE SCORE CALCULATION

The confidence score is calculated using a combination of the objectness score and the class probability scores. The objectness score represents the likelihood that the bounding box contains an object of any class. In contrast, the class probability scores represent the likelihood that the object in the bounding box belongs to each possible class.

The confidence score is calculated by multiplying the objectness score and the maximum class probability score for the bounding box. This means that the confidence score will be high if the objectness score is high and the predicted class is also highly probable.

During the detection process, the algorithm filters out all bounding boxes with a confidence score below a certain threshold. This threshold can be adjusted depending on the desired trade-off between precision and recall. A high threshold will result in a more precise detection, but it may miss some objects, while a lower threshold will detect more objects, but may also result in more false positives.

CNN
Convolutional Neural Network or CNN is a type of neural network that is commonly used in image and video processing tasks. It is designed to automatically and adaptively learn spatial hierarchies of features from input images, without requiring manual feature extraction.

CONVOLUTIONAL LAYER

A convolutional layer applies a set of learnable filters or kernels to the input image to extract features.
The filters slide or convolve over the input image and perform element-wise multiplication and summation to produce a feature map, which represents the activation of the filter at each spatial location of the input image.
Convolutional layers learn to detect different low-level and high-level features, such as edges, corners, and object parts, by adjusting the weights of the filters during training.
POOLING LAYER

A pooling layer reduces the spatial size of the feature maps by downsampling or subsampling them.
The most common type of pooling is max pooling, which selects the maximum value within a fixed window or kernel size and discards the rest of the values.
Max pooling helps to reduce the number of parameters and computations in the network, improve translation invariance, and increase the robustness to small spatial translations and distortions in the input images.
Other types of pooling include average pooling, which computes the average value within the window, and global pooling, which aggregates the feature maps into a single value per channel.
FULLY CONNECTED LAYER

The fully connected layer takes the flattened output of the previous layers and performs a matrix multiplication with a set of learnable weights. This layer is typically used to perform the final classification or regression task.

DROPOUT LAYER

The dropout layer is a regularization technique that randomly drops out a certain percentage of the neurons in the network during training, preventing overfitting.

BATCH NORMALIZATION LAYER

The batch normalization layer is used to normalize the input to each layer, reducing the dependence of the network on the scale and distribution of the input.



