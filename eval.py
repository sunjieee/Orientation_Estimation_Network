from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
from PIL import Image


DATA_PATH = '/data5/sunjie/output_eval_part9.tfrecords'
BATCH_SIZE = 36
LEARNING_RATE_BASE = 0.001
LEARING_RATE_DECAY = 0.96
MODEL_SAVE_PATH = '/data5/sunjie/model_save_20170507'
MODEL_SAVE_PATH_1 = '/data5/sunjie/model_save_eval_1'
MODEL_NAME = 'model.ckpt'
TRAINING_STEPS = 52

"""
Predefine all necessary layer for the AlexNet
""" 
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
	"""
	Adapted from: https://github.com/ethereon/caffe-tensorflow
	"""
	# Get number of input channels
	input_channels = int(x.get_shape()[-1])
  
	# Create lambda function for the convolution
	convolve = lambda i, k: tf.nn.conv2d(i, k, 
		strides = [1, stride_y, stride_x, 1],
		padding = padding)
  
	with tf.variable_scope(name) as scope:
		# Create tf variables for the weights and biases of the conv layer
		weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
		biases = tf.get_variable('biases', shape = [num_filters])  
    
    
		if groups == 1:
			conv = convolve(x, weights)
      
		# In the cases of multiple groups, split inputs & weights and
		else:
			# Split input and weights and convolve them separately
			input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
			weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
			output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
      
			# Concat the convolved output together again
			conv = tf.concat(axis = 3, values = output_groups)
      
		# Add biases 
		bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
		# Apply relu function
		relu = tf.nn.relu(bias, name = scope.name)
        
		return relu
  
def fc(x, num_in, num_out, name, relu = True):
	with tf.variable_scope(name) as scope:
    
		# Create tf variables for the weights and biases
		weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
		biases = tf.get_variable('biases', [num_out], trainable=True)
    
		# Matrix multiply weights and inputs and add bias
		act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
		if relu == True:
			# Apply ReLu non linearity
			relu = tf.nn.relu(act)      
			return relu
		else:
			return act
    

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
	return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
		strides = [1, stride_y, stride_x, 1],
		padding = padding, name = name)
  
def lrn(x, radius, alpha, beta, name, bias=1.0):
	return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
		beta = beta, bias = bias, name = name)
  
def dropout(x, keep_prob):
	return tf.nn.dropout(x, keep_prob)



def get_input():
	
	#files = tf.train.match_filenames_once(DATA_PATH)

	filename_queue = tf.train.string_input_producer([DATA_PATH], shuffle=False)

	reader = tf.TFRecordReader()

	_,serialized_example = reader.read(filename_queue)

	features = tf.parse_single_example(
	    serialized_example,
	    features={
	        'image':tf.FixedLenFeature([],tf.string),
	        'label':tf.FixedLenFeature([],tf.int64),
	        #'filename':tf.FixedLenFeature([],tf.int64)
	    })

	decoded_image = tf.decode_raw(features['image'], tf.uint8)
	
	reshaped_image = tf.reshape(decoded_image, [227, 227, 3])
	
	retyped_image = tf.image.convert_image_dtype(reshaped_image, tf.float32)

	retyped_image = tf.subtract(retyped_image, 0.5)

  	retyped_image = tf.multiply(retyped_image, 2.0)
	
	theta = tf.cast(features['label'], tf.float32)
	
	label = tf.stack([tf.sin(theta * np.pi / 180.), tf.cos(theta * np.pi / 180.)])
	
	filename = 1#tf.cast(features['filename'], tf.int32)
	
	capacity = 3 * BATCH_SIZE
	
	image_batch, label_batch, out_theta, out_filename = tf.train.batch([retyped_image, label, theta, filename], 
	                                                    batch_size=BATCH_SIZE, 
	                                                    capacity=capacity)

	return image_batch, label_batch, out_theta, out_filename


def inference(images):
	
	# 1st Layer: Conv (w ReLu) -> Pool -> Lrn
	conv1 = conv(images, 11, 11, 96, 4, 4, padding = 'VALID', name = 'conv1')
	pool1 = max_pool(conv1, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
	norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')

	# 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
	conv2 = conv(norm1, 5, 5, 256, 1, 1, groups = 2, name = 'conv2')
	pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
	norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')

	# 3rd Layer: Conv (w ReLu)
	conv3 = conv(norm2, 3, 3, 384, 1, 1, name = 'conv3')

	# 4th Layer: Conv (w ReLu) splitted into two groups
	conv4 = conv(conv3, 3, 3, 384, 1, 1, groups = 2, name = 'conv4')

	# 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
	conv5 = conv(conv4, 3, 3, 256, 1, 1, groups = 2, name = 'conv5')
	pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
	flattened = tf.reshape(pool5, [BATCH_SIZE, -1])
	dim = flattened.get_shape()[1].value
	fc6 = fc(flattened, dim, 4096, name='fc6')
	dropout6 = dropout(fc6, 1.0)

	# 7th Layer: FC (w ReLu) -> Dropout
	fc7 = fc(dropout6, 4096, 4096, name = 'fc7')
	dropout7 = dropout(fc7, 1.0)

	fc8 = fc(dropout7, 4096, 2, relu = False, name='fc8')

	normalized_logits = tf.nn.l2_normalize(fc8, 1, name='l2_normalize')


	return normalized_logits

def get_loss(logits, labels):

	loss = tf.nn.l2_loss(logits - labels)

	return loss


def get_theta(sin_cos, out_theta):
	
	theta = np.arctan2(sin_cos[:, 0] , sin_cos[:, 1]) * 180 / np.pi
	res = 0.0
	for i in range(BATCH_SIZE):
		if (np.abs(theta[i] - out_theta[i]) > 180.0):
			k = 360.0 - np.abs(theta[i] - out_theta[i])
		else:
			k = np.abs(theta[i] - out_theta[i])
		res += k
	print(res)
	return res / BATCH_SIZE





def main():

	with tf.Graph().as_default(), tf.device('/cpu:0'):
		
		image_batch, label_batch, theta, filename = get_input()

		#tf.summary.image('images', image_batch, 128)

		with tf.device('gpu:1'):

			logits = inference(image_batch)
			
			loss = get_loss(logits, label_batch)

		tf.summary.scalar('loss', loss)
			
		summary_op = tf.summary.merge_all()
		
		saver = tf.train.Saver()
		
		init = tf.global_variables_initializer()





		with tf.Session(config=tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False)) as sess:

			sess.run(init)

			ckpt = tf.train.get_checkpoint_state(
						MODEL_SAVE_PATH)

			if ckpt and ckpt.model_checkpoint_path:

				saver.restore(sess, ckpt.model_checkpoint_path)

				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			
			coord = tf.train.Coordinator()
			
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)
			
			summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH_1, sess.graph)
			
			www = 0.0

			for step in range(TRAINING_STEPS):
				
				loss_value, sin_cos, out_theta, out_filename, summary = sess.run([loss, logits, theta, filename, summary_op])

				#print('setp %d: loss = %.4f'  % (step, loss_value))
				#print(out_filename)

				www += get_theta(sin_cos, out_theta)

					
					
				summary_writer.add_summary(summary, step)
			print(www / TRAINING_STEPS)



			coord.request_stop()
			
			coord.join(threads)


if __name__ == '__main__':

	main()
