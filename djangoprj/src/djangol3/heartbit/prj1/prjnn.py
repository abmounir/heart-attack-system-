import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('heart.csv')

shuffle_df=df.sample(frac=1)
outputs=shuffle_df[['target']]

inputs=shuffle_df.drop(['target','thalach','exang','oldpeak','slope','ca','thal','age','sex','cp'],axis=1)
va=outputs.tail(1).values
#scale data
inputs=inputs/np.amax(inputs,axis=0)


pred=inputs.tail(1).values
inputs=inputs.head(30).values
outputs=outputs.head(30).values
no_samples=outputs.shape[0]
print(no_samples)
class NeuralNetwork:
	def __init__(self, input_nodes,hidden1_nodes,hidden2_nodes,output_nodes,no_samples):
		self.input_nodes=input_nodes
		self.hidden1_nodes=hidden1_nodes
		self.hidden2_nodes=hidden2_nodes
		self.output_nodes=output_nodes

		# init weights
		self.input_hidden_weights=np.random.randn(self.input_nodes,self.hidden1_nodes)
		self.hidden_hidden_weights=np.random.randn(self.hidden1_nodes,self.hidden2_nodes)
		self.hidden_output_weights=np.random.randn(self.hidden2_nodes,self.output_nodes)
		
		#init bias
		self.input_hidden_bias=np.ones((no_samples,1),dtype=float)
		self.hidden_hidden_bias=np.ones((no_samples,1),dtype=float)
		self.hidden_output_bias=np.ones((no_samples,1),dtype=float)
		#print(self.input_hidden_bias.shape)
		
	def Sigmoid(self,x):
		return 1/(1+np.exp(-x))

	def SigmoidPrime(self,s):
		return s * (1 - s)


	def Forward(self,x): # FeedForward algo
		self.z=np.dot(x,self.input_hidden_weights) +self.input_hidden_bias
		self.a=self.Sigmoid(self.z)
		self.z1=np.dot(self.a,self.hidden_hidden_weights)+self.hidden_hidden_bias
		self.a1=self.Sigmoid(self.z1)
		self.z2=np.dot(self.a1,self.hidden_output_weights) +self.hidden_output_bias
		self.a2=self.Sigmoid(self.z2)
		#print(self.a2)
		return self.a2 # target 
	

	def BackProp(self,x,y,target):
		
		self.error=target-y
		self.output_delta=self.error*self.SigmoidPrime(target)
		
		self.z2_error=self.output_delta.dot(self.hidden_output_weights.T)
		self.z2_delta=self.z2_error*self.SigmoidPrime(self.a1)
		
		self.z1_error=self.z2_delta.dot(self.hidden_hidden_weights.T)
		self.z1_delta=self.z1_error*self.SigmoidPrime(self.a)

		self.input_hidden_weights-=x.T.dot(self.z1_delta)
		self.hidden_hidden_weights-=self.a.T.dot(self.z2_delta)
		self.hidden_output_weights-=self.a1.T.dot(self.output_delta)
		#print(self.input_hidden_weights)


	def Loss(self,y,target,no_samples):
		return 1/no_samples *np.sum((target-y)**2)

	def Train(self,x,y):
		t=self.Forward(x)
		self.BackProp(x,y,t)
		return t
	def predict(self,x):
		return self.Forward(x)

nn=NeuralNetwork(4,50,30,1,no_samples)
print('Training ............')
ls=[]
it=[]
for i in range(1,3000):
	target=nn.Train(inputs,outputs)
	loss=nn.Loss(outputs,target,no_samples)
	ls.append(loss)
	it.append(i)
	print('Iteration'+str(i)+'\t Loss :'+str(loss))
	

l=nn.Loss(outputs,target,no_samples)	
print('Iteration'+str(i)+'\t Final Loss :'+str(l))
print(pred)
p=nn.predict(pred)
print(p)
pr=(np.sum(p))/no_samples

print('Valeur actuelle : ')
print(va)

pr=round(pr)

print('#######----------------------#################')
print('Valeur predite :')
print(pr)

plt.plot(it,ls)
plt.show()