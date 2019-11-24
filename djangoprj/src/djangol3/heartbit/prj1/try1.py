import numpy as np
import pandas as pd


#df=pd.read_csv('heart.csv')

#shuffle_df=df.sample(frac=1)
#outputs=shuffle_df[['target']].values
#inputs=shuffle_df.drop(['target'],axis=1)
#inputs=inputs.values

#scale data

#inputs=inputs/np.amax(inputs,axis=0)
#no_samples=outputs.shape[0]

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
no_samples=3
X = X/np.amax(X, axis=0)
y = y/100
print(no_samples)
class NeuralNetwork:
	def __init__(self, input_nodes,hidden_nodes,output_nodes,no_samples):
		self.input_nodes=input_nodes
		self.hidden_nodes=hidden_nodes
		self.output_nodes=output_nodes
		# init weights
		self.input_hidden_weights=np.random.randn(self.input_nodes,self.hidden_nodes)
		self.hidden_output_weights=np.random.randn(self.hidden_nodes,self.output_nodes)
		
		#init bias
		self.input_hidden_bias=np.random.randn(no_samples,1)
		self.hidden_output_bias=np.random.randn(no_samples,1)
		#print(self.input_hidden_bias.shape)
	def Sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def SigmoidPrime(self,s):
		return s * (1 - s)


	def Forward(self,x): # FeedForward algo
		self.z=np.dot(x,self.input_hidden_weights) +self.input_hidden_bias
		self.a=self.Sigmoid(self.z)
		self.z2=np.dot(self.a,self.hidden_output_weights) +self.hidden_output_bias
		self.a2=self.Sigmoid(self.z2)
		#print(self.a2)
		return self.a2 # target 
	

	def BackProp(self,x,y,target):
		
		self.error=target-y
		self.output_delta=self.error*self.SigmoidPrime(target)
		self.z=self.output_delta.dot(self.hidden_output_weights.T)
		self.z_delta=self.z*self.SigmoidPrime(self.a)
		self.input_hidden_weights-=x.T.dot(self.z_delta)
		self.hidden_output_weights-=self.a.T.dot(self.output_delta)
		#print(self.input_hidden_weights)


	def Loss(self,y,target,no_samples):
		return 1/no_samples *np.sum((target-y)**2)

	def Train(self,x,y):
		t=self.Forward(x)
		self.BackProp(x,y,t)
		return t

nn=NeuralNetwork(2,50,1,no_samples)
print('Training ............')
for i in range(1,500):
	target=nn.Train(X,y)
	loss=nn.Loss(y,target,no_samples)
	print('Iteration'+str(i)+'\t Loss :'+str(loss))
	

l=nn.Loss(y,target,no_samples)	
print('Iteration'+str(i)+'\t Final Loss :'+str(l))
