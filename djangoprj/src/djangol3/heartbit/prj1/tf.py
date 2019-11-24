import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.reset_default_graph()
num_sapmles=1700
x=tf.placeholder(tf.float32,[num_sapmles,None])
y=tf.placeholder(tf.float32,[num_sapmles,1])
data=np.linspace(10,100,num_sapmles)

#y=4x**3 + 3x**2 +1
lbs=4*np.power(data,3) + 3*np.power(data,2) +1
#lbs=lbs.reshape(100,1)
#print(lbs.shape)
#plt.scatter(data,lbs)
#plt.show()
data=data.reshape(num_sapmles,1)
data=np.float32(data)
lbs=lbs.reshape(num_sapmles,1)

def Model(data,hidden_nodes1,hidden_nodes2):
	hl1={'weight':tf.Variable(tf.random_normal([1,hidden_nodes1])),'bias':tf.Variable(tf.random_normal([hidden_nodes1]))}
	hl2={'weight':tf.Variable(tf.random_normal([hidden_nodes1,hidden_nodes2])),'bias':tf.Variable(tf.random_normal([hidden_nodes2]))}
	output_layer={'weight':tf.Variable(tf.random_normal([hidden_nodes2,1])),'bias':tf.Variable(tf.random_normal([1]))}
	init=tf.global_variables_initializer()

	#sess=tf.Session()
	#sess.run(init)

	l1=tf.add(tf.matmul(data,hl1['weight']),hl1['bias'])
	l1=tf.nn.sigmoid(l1)

	l2=tf.add(tf.matmul(l1,hl2['weight']),hl2['bias'])
	l2=tf.nn.sigmoid(l2)

	out=tf.add(tf.matmul(l2,output_layer['weight']),output_layer['bias'])
	
	return out

def Train(data,hn1,hn2):
	pred=Model(data,hn1,hn2)
	init=tf.global_variables_initializer()
	cost = tf.losses.mean_squared_error(y,pred)
	optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(cost)
	with tf.Session() as sess:
		sess.run(init)
		predlist=[]
		for epoch in range(11000):
			c=sess.run(optimizer,feed_dict={x:data,y:lbs})
			cc=sess.run(cost,feed_dict={x:data,y:lbs})
			if epoch % 1000 == 0 :
				print('epoch : '+str(epoch)+'  Loss :',(cc))
		predlist.append(sess.run(pred))
		#writer = tf.summary.FileWriter('./gr', sess.graph)
		#writer.close()
	print(len(predlist))	
	plt.scatter(data,predlist)
	plt.scatter(data,lbs)
	plt.show()
data=data/np.amax(data,axis=0)
lbs=lbs/np.amax(lbs,axis=0)
Train(data,30,30)		


