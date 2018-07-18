#importing all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import tflearn
from PIL import Image
from skimage.util.montage import montage2d

#importing the dataset
from tensorflow.examples.tutorials.mnist import input_data
#one hot encoding returns an array of zeros and a single one, One corresponds to the class
data = input_data.read_data_sets("data/MNIST/",one_hot=True)

#checking the shape of the input data the datasetis divided into the training dataset , test dataset,validation dataset
#It tells the total numbers of data in a respective datasets along with there size
print ("shape of the images in training dataset {}".format(data.train.images.shape))
print("shape of the class in training dataset {}".format(data.train.labels.shape))
print("shape of the images in test dataset {}".format(data.test.images.shape))
print("shape of the class in test dataset {}".format(data.test.labels.shape))
print("shape of the images in validation dataset {}".format(data.validation.images.shape))
print("shape of the class in validation dataset {}".format(data.validation.labels.shape))

#sample data
sample_image = data.train.images[5].reshape(28,28)
plt.imshow(sample_image,cmap='gray')
plt.title('sample image')
plt.axis('off')
plt.show()

#function to create a montage of input image
montage_image = data.train.images[0:100]
new_montage_image = np.zeros([100,28,28])
for i in range(len(montage_image)):
    new_montage_image[i] = montage_image[i].reshape(28,28)
plt.imshow(montage2d(new_montage_image),cmap='gray')
plt.title('montage image of the input data')
plt.axis('off')
plt.show()

#for data visualization we will calculate the mean and standard daviation
images = data.train.images
images = np.reshape(images,[images.shape[0],28,28])
mean_image = np.mean(images,axis=0)
std_image = np.std(images,axis=0)

#for plotting the mean
plt.imshow(mean_image)
plt.title('mean of the input data')
plt.colorbar()
plt.axis('off')
plt.show()

#for plotting the standard deviation
plt.imshow(std_image)
plt.title('standard daviation of the input data')
plt.colorbar()
plt.axis('off')
plt.show()

#Its time to define our placeholder from were our input is fed to the model
#input - shape 'None' states that, the value can be anything, i.e we can feed in any number of images
x = tf.placeholder(tf.float32,shape=[None,784])#input image
y =  tf.placeholder(tf.float32,shape=[None,10])#input class


#model layout will be as fllows input_layer --> convolutional_layer1 --> convolutinal_layer2 -->fully_connected_layer --> softmax_layer
#input_layer
#reshaping input for convolutional  operation in tensorflow
# '-1' states  that there is no fixed batch dimension, 28x28(=784) is reshaped from 784 pixels and '1' for a single 
#channel, i.e a gray scale image
x_input = tf.reshape(x,[-1,28,28,1], name = 'input')
# The first convolutional layer with 32 output filters, filter  size is 5x5, strides of 2,same padding, and RELU activation.
#biasing is not added but one can add bias. Optionally you can add max pooling layer as well
conv_layer1 = tflearn.layers.conv.conv_2d(x_input,nb_filter=32,filter_size=5,strides=[1,1,1,1],padding = 'same',activation='relu',
                                       regularizer='L2',name='conv_layer_1')
#2x2 max pool layer
out_layer1 = tflearn.layers.conv.max_pool_2d(conv_layer1,2)

#convolutonal layer 2
conv_layer2 = tflearn.layers.conv.conv_2d(out_layer1,nb_filter=23,filter_size=5,strides=[1,1,1,1],padding='same',activation='relu',
                                        regularizer='L2',name='conv_layer_2')
#2x2 max pool layer
out_layer2 = tflearn.layers.conv.max_pool_2d(conv_layer2,2)
#fully connected layer
fc1 = tflearn.layers.core.fully_connected(out_layer2,1024,activation='relu')
fc1_dropout = tflearn.layers.core.dropout(fc1,0.8)
y_predict = tflearn.layers.core.fully_connected(fc1_dropout,10,activation = 'softmax',name='dropout')


print("shape of the input {}".format(x_input.get_shape().as_list()))
print("shape of the convolutional layer 1 {}".format(out_layer1.get_shape().as_list()))
print("shape of the convolutional layer 2 {}".format(out_layer2.get_shape().as_list()))
print("shape of the fully connected layer {}".format(fc1.get_shape().as_list()))
print("shape of the output layer 1 {}".format(y_predict.get_shape().as_list()))

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_predict),reduction_indices = [1]))
#optamizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#calculating acuracy of our model
correct_prediction = tf.equal(tf.argmax(y_predict,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#session parameter
sess = tf.InteractiveSession()
#Initializing variables
init = tf.global_variables_initializer()
sess.run(init)

#get the graph
g = tf.get_default_graph()

#every operation in our graph
[op.name for op in g.get_operations()]


#number of interations
epoch=15000
batch_size=50
for i in range(epoch):
    #batch wise training 
    x_batch, y_batch = data.train.next_batch(batch_size)
    _,loss=sess.run([train_step, cross_entropy], feed_dict={x: x_batch,y: y_batch})
    #_, loss,acc=sess.run([train_step,cross_entropy,accuracy], feed_dict={x:input_image , y_: input_class})
    
    if i%500==0:    
        Accuracy=sess.run(accuracy,
                           feed_dict={
                        x: data.test.images,
                        y: data.test.labels
                      })
        Accuracy=round(Accuracy*100,2)
        print ("Loss : {} , Accuracy on test set : {} %" .format(loss, Accuracy))
    elif i%100==0:
        print ("Loss : {}" .format(loss))



#testing on the validation dataset
validation_accuracy=round((sess.run(accuracy,
                            feed_dict={
                             x: data.validation.images,
                             y: data.validation.labels
                              }))*100,2)

print ("Accuracy in the validation dataset: {}%".format(validation_accuracy))

#testset predictions
y_test=(sess.run(y_predict,feed_dict={
                             x: data.test.images
                              }))

#Confusion Matrix
true_class=np.argmax(data.test.labels,1)
predicted_class=np.argmax(y_test,1)
cm=confusion_matrix(predicted_class,true_class)
print(cm)





















