import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.reset_default_graph()

#Preprocessing 
df=pd.read_csv('heart.csv')
num_sapmles=270
shuffle_df=df.sample(frac=1)
outputs=shuffle_df[['target']]
no_samples=outputs.shape[0]
print(no_samples)
inputs=shuffle_df.drop(['target','thalach','exang','oldpeak','slope','ca','thal','age','sex','cp'],axis=1)
va=outputs.tail(1).values

#scale data

inputs=inputs/np.amax(inputs,axis=0)
test_x=inputs.tail(33).values
test_y=outputs.tail(33).values
test_x=np.float32(test_x)
test_y=np.float32(test_y)
inputs=inputs.head(num_sapmles).values
outputs=outputs.head(num_sapmles).values

inputs=np.float32(inputs)
outputs=np.float32(outputs)
print(test_x.shape)

x=tf.placeholder(tf.float32,[None,4])
y=tf.placeholder(tf.float32,[None,1])

hidden_nodes1=200
hidden_nodes2=100
hl1={'weight':tf.Variable(tf.random_normal([4,hidden_nodes1]),name='w1'),'bias':tf.Variable(tf.random_normal([hidden_nodes1]),name='b1')}
hl2={'weight':tf.Variable(tf.random_normal([hidden_nodes1,hidden_nodes2]),name='w2'),'bias':tf.Variable(tf.random_normal([hidden_nodes2]),name='b2')}
output_layer={'weight':tf.Variable(tf.random_normal([hidden_nodes2,1]),name='w3'),'bias':tf.Variable(tf.random_normal([1]),name='b3')}
    

#sess=tf.Session()
#sess.run(init)

l1=tf.add(tf.matmul(inputs,hl1['weight']),hl1['bias'])
l1=tf.nn.sigmoid(l1)

l2=tf.add(tf.matmul(l1,hl2['weight']),hl2['bias'])
l2=tf.nn.sigmoid(l2)

out=tf.add(tf.matmul(l2,output_layer['weight']),output_layer['bias'])
saver = tf.train.Saver()
    

#Model(inputs,4,5)

def Train(data):
    pred=tf.nn.sigmoid(out)
    print(pred)
    #### init 
    init=tf.global_variables_initializer()
    cross = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred)
    cost = tf.reduce_mean(cross)
    optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    correct_prediction = tf.equal(tf.round(pred),y, name="correct_prediction")
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
    test_feed = {x:test_x}
    print(test_x.shape)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(20000) :
            c=sess.run(optimizer,feed_dict={x:data,y:outputs}) 
            cc=sess.run(cost,feed_dict={x:data,y:outputs})
            #validation_accuracy = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
            if epoch % 1000 == 0 :
                print('epoch : '+str(epoch)+'\t Loss :',(cc))
        saver.save(sess, "./hrt.ckpt")
        w1=sess.run(hl1['weight'])
        b1=sess.run(hl1['bias'])
        
        w2=sess.run(hl2['weight'])
        b2=sess.run(hl1['bias'])
        
        w3=sess.run(output_layer['weight'])
        b3=sess.run(output_layer['bias'])
       
    return w1,b1,w2,b2,w3,b3
    ##########################
        
q,w,e,r,t,y=Train(inputs) 

###########
tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("./hrt.ckpt.meta") 
with tf.Session() as sess:
    imported_meta.restore(sess, tf.train.latest_checkpoint('./'))
    w11=sess.run('w1:0')
    b11=sess.run('b1:0')
    w22=sess.run('w2:0')
    b22=sess.run('b2:0')
    w33=sess.run('w3:0')
    b33=sess.run('b3:0')
    l11=tf.add(tf.matmul(test_x,w11),b11)
    l11=tf.nn.sigmoid(l11)

    l22=tf.add(tf.matmul(l11,w22),b22)
    l22=tf.nn.sigmoid(l22)

    outt=tf.add(tf.matmul(l22,w33),b33)
    outt=tf.nn.sigmoid(outt)
    outt=sess.run(outt)
    
clf = np.where(outt<0.5,0,1)
accrc = np.sum(clf==test_y) / len(test_y)
print('Accuracy : '+str(accrc*100)+' %')
