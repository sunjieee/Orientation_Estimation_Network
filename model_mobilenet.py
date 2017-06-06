from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
from mobilenet_model import *

DATA_PATH = '/data5/sunjie/output_train_part*.tfrecords'
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.0001
LEARING_RATE_DECAY = 0.96
MODEL_SAVE_PATH = '/data5/sunjie/model_save'
MODEL_NAME = 'model.ckpt'
TRAINING_STEPS = 1000000



def get_input():
	
	files = tf.train.match_filenames_once(DATA_PATH)

	filename_queue = tf.train.string_input_producer(files, shuffle=False)

	#reader = tf.TFRecordReader()

	#_,serialized_example = reader.read(filename_queue)
	num_readers = 12
	examples_queue = tf.RandomShuffleQueue(
		capacity=10000 + 3 * BATCH_SIZE,
		min_after_dequeue=10000,
		dtypes=[tf.string])

	enqueue_ops = []
	for _ in range(num_readers):
		reader = tf.TFRecordReader()
		_, value = reader.read(filename_queue)
		enqueue_ops.append(examples_queue.enqueue([value]))

	tf.train.queue_runner.add_queue_runner(
		tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
	serialized_example = examples_queue.dequeue()

	images_and_labels = []

	num_preprocess_threads = 12

	for thread_id in range(num_preprocess_threads):

		features = tf.parse_single_example(
		    serialized_example,
		    features={
		        'image':tf.FixedLenFeature([],tf.string),
		        'label':tf.FixedLenFeature([],tf.int64)
		    })

		decoded_image = tf.decode_raw(features['image'], tf.uint8)
		
		reshaped_image = tf.reshape(decoded_image, [227, 227, 3])
		
		retyped_image = tf.image.convert_image_dtype(reshaped_image, tf.float32)

		retyped_image = tf.subtract(retyped_image, 0.5)

	  	retyped_image = tf.multiply(retyped_image, 2.0)
		
		theta = tf.cast(features['label'], tf.float32)
		
		label = tf.stack([tf.sin(theta * np.pi / 180.), tf.cos(theta * np.pi / 180.)])

		images_and_labels.append([retyped_image, label])
	
	min_after_dequeue = 10000
	
	capacity = min_after_dequeue + 2 * num_preprocess_threads * BATCH_SIZE
	
	image_batch, label_batch = tf.train.shuffle_batch_join(images_and_labels, 
	                                                    batch_size=BATCH_SIZE, 
	                                                    capacity=capacity, 
	                                                    min_after_dequeue=min_after_dequeue)

	return image_batch, label_batch


def inference(images):
	
	models = MobileNets(images, num_classes=2)
	fc8, end_points = models.inference()

	normalized_logits = tf.nn.l2_normalize(fc8, 1, name='l2_normalize')


	return normalized_logits



def get_loss(logits, labels):

	loss = tf.nn.l2_loss(logits - labels)

	return loss


def training(loss):

	global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
			trainable=False)

	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
		global_step,
		2700000 / BATCH_SIZE,
		LEARING_RATE_DECAY,
		staircase=True)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate)

	train_op = optimizer.minimize(loss, global_step=global_step)

	return train_op



def main():

	with tf.Graph().as_default(), tf.device('/cpu:0'):
		
		image_batch, label_batch = get_input()

		#tf.summary.image('images', image_batch, 10)

		with tf.device('gpu:1'):

			logits = inference(image_batch)
			
			loss = get_loss(logits, label_batch)

			train_op = training(loss)

		tf.summary.scalar('loss', loss)
			
		summary_op = tf.summary.merge_all()
		
		saver = tf.train.Saver()
		
		init = tf.global_variables_initializer()





		with tf.Session(config=tf.ConfigProto(
			allow_soft_placement=True,
			log_device_placement=False)) as sess:

			sess.run(init)
			
			coord = tf.train.Coordinator()
			
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)
			
			summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)

			for step in range(TRAINING_STEPS):
				
				_, loss_value = sess.run([train_op, loss])

				if step % 100 == 0:
					
					print('setp %d: loss = %.4f'  % (step, loss_value))
					
					summary = sess.run(summary_op)
					
					summary_writer.add_summary(summary, step)

				if step % 5000 == 0:
					
					checkpoint_path = os.path.join(
						MODEL_SAVE_PATH, MODEL_NAME)
					
					saver.save(sess, checkpoint_path, global_step=step)



			coord.request_stop()
			
			coord.join(threads)


if __name__ == '__main__':

	main()
