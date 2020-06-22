# Traffic-Sign-Recognition
Traffic Sign Recognition 

## Writeup

--

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_result/histogram_y_train.png "Visualization Train Images [Qty Vs Labels]"
[image2]: ./images_result/histogram_y_valid.png "Visualization Valid Images [Qty Vs Labels]"
[image3]: ./images_result/histogram_y_test.png "Visualization Test Images [Qty Vs Labels]"
[image4]: ./images_result/visualizationImages.png "Original Images"
[image5]: ./images_result/originalImageGrayNormalized.png "Gray scaled image"
[image6]: ./images_result/augmentedGrayImages.png "Augmented images"
[image7]: ./images_result/histogram_y_trainBefore.png "New images"
[image8]: ./images_result/histogram_y_trainAfter.png "New images"
[image9]: ./images_result/visualizationNewImages.png "New images"
[image10]: ./images_result/Modelused.png "Model used"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/dannofield/Traffic-Sign-Recognition)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(labels)
```

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images we have per label.
We can see that they are not the same number, this could lead us to problems when we start to train our model.

```python
# histogram of label frequency
hist, bins = np.histogram(y_train, bins=n_classes)

fig, ax = plt.subplots(figsize=(20,20))    
width = 0.7 # the width of the bars 
ind = np.arange(n_classes)  # the x locations for the groups
ax.barh(ind, hist, width, align='edge',color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(labels.index, minor=False)
for i, v in enumerate(hist):
    ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
plt.title(str(n_train) + ' Training Images')
plt.xlabel('# of Training Images')
plt.ylabel('Classes')
plt.show()
```

##### Number of Training images per label
![alt text][image1]

##### Number of Valid images per label
![alt text][image2]

##### Number of Test images per label
![alt text][image3]

### Design and Test a Model Architecture

Here we can see an exploratory visualization of the images and their labels according to their indexes before grayscaling.
![alt text][image4]

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tryed first to test the model using full color images but I could not reach more than 91% accuracy. So I decided to convert
the images to grayscale first before normalizing them.

Here is an example of a traffic sign image after grayscaling.

![alt text][image5]

As a last step, I normalized the image data because Given the use of small weights in the model and the use of error between predictions and expected values, the scale of inputs and outputs used to train the model are an important factor. Unscaled input variables can result in a slow or unstable learning process, whereas unscaled target variables on regression problems can result in exploding gradients causing the learning process to fail.

Data preparation involves using techniques such as the normalization and standardization to rescale input and output variables [prior to training a neural network model](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)

#### I generated additional data for training

I decided to generate additional data because as you can see on the histograms, we have few training images from some labels
To add more data to the the data set, I used the following techniques to increase the number of images with labels lower than 800 units.


```python
def random_translate(img):
    rows,cols,_ = img.shape
    
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

def random_scaling(img):   
    rows,cols,_ = img.shape

    # transform limits
    px = np.random.randint(-2,2)

    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])

    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(rows,cols))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

def random_warp(img):
    
    rows,cols,_ = img.shape

    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06

    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

def random_brightness(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    return dst

```

So I took one image as an example and generated images base on that one.

```python
for ...
    new_img = random_translate( random_scaling( random_warp( random_brightness(image_from_original_dataset))))
    X_train = np.concatenate((X_train, [new_img]), axis=0)
```
##### Here is an example of an original image and an augmented images:

![alt text][image5]

![alt text][image6]

You can see the difference between the original data set before creating the images and the augmented data set after. 
```
creating images...
Creating 620 random images for label Speed limit (20km/h)
Creating 440 random images for label End of speed limit (80km/h)
Creating 110 random images for label Stop
Creating 260 random images for label No vehicles
Creating 440 random images for label Vehicles over 3.5 metric tons prohibited
Creating 620 random images for label Dangerous curve to the left
Creating 500 random images for label Dangerous curve to the right
Creating 530 random images for label Double curve
Creating 470 random images for label Bumpy road
Creating 350 random images for label Slippery road
Creating 560 random images for label Road narrows on the right
Creating 260 random images for label Traffic signals
Creating 590 random images for label Pedestrians
Creating 320 random images for label Children crossing
Creating 560 random images for label Bicycles crossing
Creating 410 random images for label Beware of ice/snow
Creating 110 random images for label Wild animals crossing
Creating 590 random images for label End of all speed and passing limits
Creating 201 random images for label Turn right ahead
Creating 440 random images for label Turn left ahead
Creating 470 random images for label Go straight or right
Creating 620 random images for label Go straight or left
Creating 530 random images for label Keep left
Creating 500 random images for label Roundabout mandatory
Creating 590 random images for label End of no passing
Creating 590 random images for label End of no passing by vehicles over 3.5 metric tons
```
Original Dataset | Augmented data set
------------ | -------------
![alt text][image7] | ![alt text][image8]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

![alt text][image10]

(image took it from https://towardsdatascience.com/beginning-my-journey-in-self-driving-car-udacity-nano-degree-a39d898658a2)

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray & normalized image   			| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3	    | outputs 10x10x16   							|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16     				|
| Convolution 3x3	    | outputs 1x1x400   							|
| RELU					|												|
| Fully connected		|            									|
| Softmax				|           									|
|						|												|
|						|												|
 


#### 3. How I trained my model. Batch size, number of epochs and learning rate.
The hyperparameters i ended up using were
```python
EPOCHS = 60
BATCH_SIZE = 100

rate = 0.0009
```

#### 4. Approach for finding a solution

I opted for converting all the images to gray scales since the article [Sermanet/LeCunn traffic sign classification](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) mentions that ignoring color information could  increase network’s capacity and depth. I also found a better perfonace than using color images.

The architecture I have used looks like:

```python
def LeNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma), name="W1")
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6), name="b1")
    x = tf.nn.bias_add(x, b1)
    print("layer 1 shape:",x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer1 = x
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name="W2")
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x = tf.nn.bias_add(x, b2)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer2 = x
    
    # TODO: Layer 3: Convolutional. Output = 1x1x400.
    W3 = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean = mu, stddev = sigma), name="W3")
    x = tf.nn.conv2d(x, W3, strides=[1, 1, 1, 1], padding='VALID')
    b3 = tf.Variable(tf.zeros(400), name="b3")
    x = tf.nn.bias_add(x, b3)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)
    layer3 = x

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    layer2flat = flatten(layer2)
    print("layer2flat shape:",layer2flat.get_shape())
    
    # Flatten x. Input = 1x1x400. Output = 400.
    xflat = flatten(x)
    print("xflat shape:",xflat.get_shape())
```


#### My final model results were:
#### training set  & validation accuracy
Training...EPOCHS: 60 BATCH_SIZE: 100 rate: 0.0009 Iterations: 464.8

EPOCH 1 ...
Validation Accuracy = 0.775

EPOCH 2 ...
Validation Accuracy = 0.846

EPOCH 3 ...
Validation Accuracy = 0.883

EPOCH 4 ...
Validation Accuracy = 0.893

EPOCH 5 ...
Validation Accuracy = 0.905

EPOCH 6 ...
Validation Accuracy = 0.907

EPOCH 7 ...
Validation Accuracy = 0.905

EPOCH 8 ...
Validation Accuracy = 0.910

EPOCH 9 ...
Validation Accuracy = 0.912

EPOCH 10 ...
Validation Accuracy = 0.920

EPOCH 11 ...
Validation Accuracy = 0.922

EPOCH 12 ...
Validation Accuracy = 0.916

EPOCH 13 ...
Validation Accuracy = 0.915

EPOCH 14 ...
Validation Accuracy = 0.920

EPOCH 15 ...
Validation Accuracy = 0.929

EPOCH 16 ...
Validation Accuracy = 0.930

EPOCH 17 ...
Validation Accuracy = 0.936

EPOCH 18 ...
Validation Accuracy = 0.933

EPOCH 19 ...
Validation Accuracy = 0.925

EPOCH 20 ...
Validation Accuracy = 0.930

EPOCH 21 ...
Validation Accuracy = 0.934

EPOCH 22 ...
Validation Accuracy = 0.931

EPOCH 23 ...
Validation Accuracy = 0.927

EPOCH 24 ...
Validation Accuracy = 0.944

EPOCH 25 ...
Validation Accuracy = 0.936

EPOCH 26 ...
Validation Accuracy = 0.938

EPOCH 27 ...
Validation Accuracy = 0.936

EPOCH 28 ...
Validation Accuracy = 0.942

EPOCH 29 ...
Validation Accuracy = 0.935

EPOCH 30 ...
Validation Accuracy = 0.936

EPOCH 31 ...
Validation Accuracy = 0.935

EPOCH 32 ...
Validation Accuracy = 0.935

EPOCH 33 ...
Validation Accuracy = 0.937

EPOCH 34 ...
Validation Accuracy = 0.937

EPOCH 35 ...
Validation Accuracy = 0.939

EPOCH 36 ...
Validation Accuracy = 0.933

EPOCH 37 ...
Validation Accuracy = 0.945

EPOCH 38 ...
Validation Accuracy = 0.946

EPOCH 39 ...
Validation Accuracy = 0.939

EPOCH 40 ...
Validation Accuracy = 0.938

EPOCH 41 ...
Validation Accuracy = 0.933

EPOCH 42 ...
Validation Accuracy = 0.943

EPOCH 43 ...
Validation Accuracy = 0.945

EPOCH 44 ...
Validation Accuracy = 0.948

EPOCH 45 ...
Validation Accuracy = 0.944

EPOCH 46 ...
Validation Accuracy = 0.941

EPOCH 47 ...
Validation Accuracy = 0.941

EPOCH 48 ...
Validation Accuracy = 0.943

EPOCH 49 ...
Validation Accuracy = 0.950

EPOCH 50 ...
Validation Accuracy = 0.945

EPOCH 51 ...
Validation Accuracy = 0.942

EPOCH 52 ...
Validation Accuracy = 0.943

EPOCH 53 ...
Validation Accuracy = 0.945

EPOCH 54 ...
Validation Accuracy = 0.948

EPOCH 55 ...
Validation Accuracy = 0.947

EPOCH 56 ...
Validation Accuracy = 0.932

EPOCH 57 ...
Validation Accuracy = 0.946

EPOCH 58 ...
Validation Accuracy = 0.946

EPOCH 59 ...
Validation Accuracy = 0.949

EPOCH 60 ...
Validation Accuracy = 0.951

Model saved


#### test set accuracy of
INFO:tensorflow:Restoring parameters from ./lenet
Test Accuracy = 0.934


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

I tool pictures from the [German traffic signs website](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) to predict the traffic sign type.
All images are down-sampled or upsampled to 32x32 (dataset samples sizes vary from 15x15 to 250x250)
```python
german_images = []
german_labels = [20,7,3,3,14,35,7,10,9,3]

size = (32, 32)

for i, img in enumerate(glob('GERMAN_IMG_DATABASE/0000*.ppm')):
    image = cv2.imread(img)    
    
    #Not all images are 32x32
    image = cv2.resize(image,size)
    #print('Resized Dimensions : ',image.shape)    
    german_images.append(image)
```

Here are some German traffic signs that I took from their website the web:

![alt text][image9]

```python
import tensorflow as tf

#Gray scale
german_images = np.sum(german_images/3, axis=3, keepdims=True)
#Normalize them
german_images = (german_images - 128)/128 

### Calculate the accuracy for these 5 new images. 
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(german_images, german_labels)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
            
```

And the result was 100% accurate
```
INFO:tensorflow:Restoring parameters from ./lenet
Test Accuracy = 1.000

```

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

# References
Rescale input and output variables [prior to training a neural network model](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/) 
https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/

https://towardsdatascience.com/beginning-my-journey-in-self-driving-car-udacity-nano-degree-a39d898658a2

https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb
