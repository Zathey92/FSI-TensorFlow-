#!/usr/bin/python
import numpy as np

print("Simulation")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivate(o):
	return o * (1.0 - o)
	
class Network:
	def __init__(self, input, hidden, output):
		self.input = input + 1 # add 1 for bias node
		self.hidden = hidden
		self.output = output
		# set up array of 1s for activations
		self.ai = [1.0] * self.input
		self.ah = [1.0] * self.hidden
		self.ao = [1.0] * self.output
		#Random Weights
		self.wi = np.random.randn(self.input, self.hidden) 
		self.wo = np.random.randn(self.hidden, self.output)
		# create arrays of 0 for changes
		self.ci = np.zeros((self.input, self.hidden))
		self.co = np.zeros((self.hidden, self.output))

	def feedforward(self,v_input):
		if len(v_input) != self.input-1:
			raise ValueError('Wrong number of inputs you silly goose!')
		# input activations
		for i in range(self.input -1): # -1 is to avoid the bias
				self.ai[i] = v_input[i]
		# hidden activations
		for j in range(self.hidden):
			sum = 0.0
			for i in range(self.input):
				sum += self.ai[i] * self.wi[i][j]
				self.ah[j] = sigmoid(sum)
			# output activations
		for k in range(self.output):
			sum = 0.0
			for j in range(self.hidden):
				sum += self.ah[j] * self.wo[j][k]
				self.ao[k] = sigmoid(sum)
		return self.ao[:]
	def backPropagate(self, targets, N):
		"""
		:param targets: y values
		:param N: learning rate
		:return: updated weights and current error
		"""
		if len(targets) != self.output:
			raise ValueError('Wrong number of targets you silly goose!')
		# calculate error terms for output
		# the delta tell you which direction to change the weights
		output_deltas = [0.0] * self.output
		for k in range(self.output):
			error = -(targets[k] - self.ao[k])
			output_deltas[k] = sigmoid_derivate(self.ao[k]) * error
		# calculate error terms for hidden
		# delta tells you which direction to change the weights
		hidden_deltas = [0.0] * self.hidden
		for j in range(self.hidden):
			error = 0.0
			for k in range(self.output):
				error += output_deltas[k] * self.wo[j][k]
			hidden_deltas[j] = sigmoid_derivate(self.ah[j]) * error
		# update the weights connecting hidden to output
		for j in range(self.hidden):
			for k in range(self.output):
				change = output_deltas[k] * self.ah[j]
				self.wo[j][k] -= N * change + self.co[j][k]
				self.co[j][k] = change
		# update the weights connecting input to hidden
		for i in range(self.input):
			for j in range(self.hidden):
				change = hidden_deltas[j] * self.ai[i]
				self.wi[i][j] -= N * change + self.ci[i][j]
				self.ci[i][j] = change
		# calculate error
		error = 0.0
		for k in range(len(targets)):
			error += 0.5 * (targets[k] - self.ao[k]) ** 2
		return error
	def train(self, patterns, iterations = 2500, N = 0.01):
		# N: learning rate
		for i in range(iterations):
			error = 0.0
			for p in patterns:
				inputs = p[0]
				targets = p[1]
				self.feedforward(inputs)
				error = self.backPropagate(targets, N)
			if i % 250 == 0:
				print('error %-.5f' % error)

	def predict(self, X):
		"""
		return list of predictions after training algorithm
		"""
		predictions = []
		for p in X:
			predictions.append(self.feedForward(p))
		return predictions
	def test(self, patterns):
		for p in patterns:
			print('y:', p[1],'x:',p[0],'->', self.feedforward(p[0]))

def load_data():
	print("loading data")
	x_input=[]
	y_input=[]
	out=[]
	combination=[]
	altitudes = list(range(12))
	speeds = list(range(51))
	for s in speeds:
		for a in altitudes:
			"""
			x_input=[s,a]
			pressure =1013.25*(1-6.5*(a/288.15))
			density = pressure/(287.05*(21+273.15))			
			lift=(1/2)*density*s*10*1.60*35.83*10
			if lift>20:
				y_input=[0]
			if lift <5.5 and a>5:
				y_input=[1]
			if lift<8.5 and a<2:
				y_input=[0]
			if lift<9 and a<0.1:
				y_input=[0]
			combination=[x_input,y_input]
			out.append(combination)
			"""
			y_input=[0]
			x_input=[s,a]
			if(s>=10 and a>3):
				y_input=[1]
			if(s<10 and a<1):
				y_input=[1]
			combination=[x_input,y_input]
			out.append(combination)
	return out
def load_random_data():
	print("loading random data")
	x_input=[[0,0],[5,0],[10,0],[8,1],[8,5],[0,5],[10,5],[10,10],[10,15],[3,2],[6,1],[20,5],[20,15],[18,0],[2,1]]
	y_input=[[1],[1],[1],[0],[0],[1],[0],[0],[0],[0],[0],[0],[0],[1],[0]]
	out=[]
	combination=[]
	altitudes = []
	speeds = []
	pair = []
	for i in range(15):
		combination=[x_input[i],y_input[i]]
		out.append(combination)
	return out
def demo():

	X = load_data()

	print(X[0]) # make sure the data looks right

	NN = Network(2, 30, 1)

	NN.train(X)

	NN.test(X)
	
if __name__ == '__main__':
	demo()
