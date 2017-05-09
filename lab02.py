import tflearn
from tflearn.data_utils import image_preloader
import tensorflow as tf
import numpy

#from tflearn.data_augmentation import ImageAugmentation
## Real-time data augmentation
#img_aug = ImageAugmentation()
#img_aug.add_random_flip_leftright()
#img_aug.add_random_rotation(max_angle=25.)
tf.reset_default_graph()
## Data loading and preprocessing
dataset_file = 'MAP.txt' #you dataset file location
X, Y = image_preloader(dataset_file, image_shape=[28, 28],
                       mode='file',grayscale=True ,categorical_labels=True, normalize=True)

##reshape for 4d to match network input
#
X = numpy.asarray(X[:]).reshape([-1, 28, 28, 1])
#for i in range(0,len(X)):
#    numpy.resize(X[i], [28,28,1])

# Building convolutional network
##Data agumentaion
#network = input_data(shape=[None, 28, 28, 1],
#                     data_augmentation=img_aug)

print("hi")
network = tflearn.input_data(shape=[None, 28, 28, 1])
network = tflearn.conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = tflearn.max_pool_2d(network, 2)
network = tflearn.local_response_normalization(network)
network = tflearn.conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = tflearn.max_pool_2d(network, 2)
network = tflearn.local_response_normalization(network)
network = tflearn.fully_connected(network, 128, activation='tanh')
network = tflearn.dropout(network, 0.8)
network = tflearn.fully_connected(network, 256, activation='tanh')
network = tflearn.dropout(network, 0.8)
network = tflearn.fully_connected(network, 3, activation='softmax')
network = tflearn.regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy')

print("bye")
# Training


model = tflearn.DNN(network)
model.fit(X, Y, n_epoch=1,snapshot_step=100, show_metric=True, run_id='convnet_mnist')

"""
#  Testing 
lab = model.predict([X[2]])
print(lab.index(max(lab)))
"""