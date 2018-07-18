#importing the important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
#libraries used for dealing with test files
import glob
import os 
import random
#for reading images from the text file
from tflearn.data_utils import image_preloader
import math

image_folder = '/catvsdog_classification/train/'
train_data = '/catvsdog_classification/training_data.txt'
test_data = '/catvsdog_classification/test_data.txt'
validation_data = '/catvsdog_classification/validation_data.txt'
train_proportion = 0.7
test_proportion = 0.2
validation_proportion = 0.1

#reading the image directories
imagefile_name = os.listdir(image_folder)
#suffle the data ohterwise the model will be fed with the single class example for the long time and it will not learn properly
random.shuffle(imagefile_name)

#total number of images
total = len(imagefile_name)
#***training data***
fr = open(train_data,'w')
train_files = imagefile_name[0:int(train_proportion*total)]#taking only 70% of the total image as training data
for filename in train_files:
    if filename[0:3] =='cat':
        fr.write(image_folder +'/'+filename+' 0\n')
    elif filename[0:3]=='dog':
        fr.write(image_folder +'/'+filename+' 1\n')
fr.close()
#***test data***
fr = open(test_data,'w')
test_files = imagefile_name[int(math.ceil(train_proportion*total)):int(math.ceil(train_proportion+test_proportion)*total)]
for filename in test_files:
    if filename[0:3] =='cat':
        fr.write(image_folder +'/'+filename+' 0\n')
    elif filename[0:3]=='dog':
        fr.write(image_folder +'/'+filename+' 1\n')
fr.close()

#***validation_data***
fr = open(validation_data,'w')
validation_files = imagefile_name[int(math.ceil((train_proportion+test_proportion)*total)):total]
for filename in validation_files:
    if filename[0:3] =='cat':
        fr.write(image_folder +'/'+filename+' 0\n')
    elif filename[0:3]=='dog':
        fr.write(image_folder +'/'+filename+' 1\n')
fr.close()


# In[11]:


#loading the images
x_train,y_train = image_preloader(train_data,image_shape=(56,56),mode='file',categorical_labels = True,normalize=True)
x_test,y_test = image_preloader(test_data,image_shape=(56,56),mode='file',categorical_labels = True,normalize=True)
x_val,y_val = image_preloader(validation_data,image_shape=(56,56),mode='file',categorical_labels = True,normalize=True)


# In[12]:


print('dataset')
print('number of training data: {} '.format(len(x_train)))
print('number of test data: {} '.format(len(x_test)))
print('number of validation data: {} '.format(len(x_val)))
print('shape of the image {}'.format(x_train[1].shape))
print('sahpe of labels {},number of classes {}'.format(y_train[1].shape,len(y_train[1])))


# In[13]:


#ploting the sample image
plt.imshow(x_train[1])
plt.title('sample image {}'.format(y_train[1]))
plt.axis('off')
plt.show


# In[14]:


#defining the placeholder
x = tf.placeholder(tf.float32,shape=[None,56,56,3],name='input_image')
y = tf.placeholder(tf.float32,shape=[None,2],name='input_class')


# In[15]:


# defining the model
input_layer=x
#first convolutional layer

conv_layer1 = tflearn.layers.conv.conv_2d(input_layer,nb_filter=64,filter_size=5,strides=[1,1,1,1],padding='same',
                                        activation='relu',regularizer='L2',name='conv_layer1')
#applying max pooling output_layer1
output_layer1 = tflearn.layers.conv.max_pool_2d(conv_layer1,2)
128
#secondlutional layer
conv_layer2= tflearn.layers.conv.conv_2d(output_layer1,nb_filter=128,filter_size=5,strides=[1,1,1,1],padding='same',
                                        activation='relu',regularizer='L2',name='conv_layer2')
#applying max pooling output_layer2
output_layer2= tflearn.layers.conv.max_pool_2d(conv_layer2,2)

#third convolutional layer
conv_layer3=tflearn.layers.conv.conv_2d(output_layer2,nb_filter=128,filter_size=5,strides=[1,1,1,1],padding='same',
                                        activation='relu',regularizer='L2',name='conv_layer3')
#applying max pooling output_layer2
output_layer3=tflearn.layers.conv.max_pool_2d(conv_layer3,2)

#fully connected layer1
fc1 = tflearn.layers.core.fully_connected(output_layer3,4096,activation='relu',name = 'FC1_layer')
fc1_dropout_1=tflearn.layers.core.dropout(fc1,0.8)

#fully connected layer2
fc2 = tflearn.layers.core.fully_connected(fc1_dropout_1,4096,activation='relu',name='FC2_layer')
fc2_dropout_2=tflearn.layers.core.dropout(fc2,0.8)

#softmax layer output
y_predict = tflearn.layers.core.fully_connected(fc2_dropout_2,2,activation='softmax',name='y_predict')


# In[16]:


#loss calculation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_predict+np.exp(-10)),reduction_indices=[1]))
#optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#calculate accuracy of our model
correct_prediction = tf.equal(tf.argmax(y_predict,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# In[17]:


#session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
save_path = '/catvsdog_classification/checkpoints'


# In[18]:


#grabing the graph
g = tf.get_default_graph()
# every operation in the graph
[op.name for op in g.get_operations()]


# In[19]:


epoch=1 # run for more iterations according your hardware's power
#change batch size according to your hardware's power. For GPU's use batch size in powers of 2 like 2,4,8,16...
batch_size=20 
no_itr_per_epoch=len(x_train)//batch_size


# In[20]:


no_itr_per_epoch
n_test=len(x_test) #number of test samples
n_val=len(x_val)  #number of validation samples


# In[ ]:


for iteration in range(epoch):
    print("Iteration no: {} ".format(iteration))
    
    previous_batch=0
    # Do our mini batches:
    for i in range(no_itr_per_epoch):
        current_batch=previous_batch+batch_size
        x_input=x_train[previous_batch:current_batch]
        x_images=np.reshape(x_input,[batch_size,56,56,3])
        
        y_input=y_train[previous_batch:current_batch]
        y_label=np.reshape(y_input,[batch_size,2])
        previous_batch=previous_batch+batch_size
        
        _,loss=sess.run([train_step, cross_entropy], feed_dict={x: x_images,y: y_label})
        if i % 100==0 :
            print ("Training loss : {}" .format(loss))
            
   
        
    x_test_images=np.reshape(x_test[0:n_test],[n_test,56,56,3])
    y_test_labels=np.reshape(y_test[0:n_test],[n_test,2])
    Accuracy_test=sess.run(accuracy,
                           feed_dict={
                        x: x_test_images ,
                        y: y_test_labels
                      })
    Accuracy_test=round(Accuracy_test*100,2)
    
    x_val_images=np.reshape(x_val[0:n_val],[n_val,56,56,3])
    y_val_labels=np.reshape(y_val[0:n_val],[n_val,2])
    Accuracy_val=sess.run(accuracy,
                           feed_dict={
                        x: x_val_images ,
                        y: y_val_labels
                      })    
    Accuracy_val=round(Accuracy_val*100,2)
    print("Accuracy ::  Test_set {} % , Validation_set {} % " .format(Accuracy_test,Accuracy_val))


# In[21]:


def process_img(img):
        img=img.resize((56, 56), Image.ANTIALIAS) #resize the image
        img = np.array(img)
        img=img/np.max(img).astype(float) 
        img=np.reshape(img, [1,56,56,3])
        return img


# In[1]:


#test your own images 
test_image=Image.open('alsasian.jpg')
test_image= process_img(test_image)
predicted_array= sess.run(y_predict, feed_dict={x: test_image})
predicted_class= np.argmax(predicted_array)
if predicted_class==0:
    print ("It is a cat")  
else :
    print ("It is a dog  ")


# In[28]:


pwd


# In[24]:


cd C:\catvsdog_classification

