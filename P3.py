from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

class Network:
	def __init__(self, input, output):
		self.x = tf.placeholder(tf.float32, [None, input])
		self.W = tf.Variable(tf.zeros([input, output]))
		self.b = tf.Variable(tf.zeros([output]))
		self.y_ = tf.placeholder(tf.float32, [None, output])
		self.y = tf.matmul(self.x, self.W) + self.b
		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
		self.sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
	def train(self, v_train,iterations=1000,lr=0.5):
		train_step = tf.train.GradientDescentOptimizer(lr).minimize(self.cross_entropy)
		for _ in range(iterations):
			batch_xs, batch_ys = v_train.next_batch(100)
			self.sess.run(train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
	def test(self,test_xs,test_ys):
		correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print(self.sess.run(accuracy, feed_dict={self.x: test_xs,self.y_: test_ys}))
		
	
def main(_):
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)
	net= Network(784,10)
	net.train(mnist.train)
	net.test(mnist.test.images, mnist.test.labels)
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)