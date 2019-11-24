import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.reset_default_graph()
def data_prepro():
	#Preprocessing 
	df=pd.read_csv("C:/Users/DELL/Desktop/djangoprj/src/djangol3/heartbit/prj1/HeartDisease.csv")
	num_sapmles=365
	shuffle_df=df.sample(frac=1)
	outputs=shuffle_df[['num']]
	no_samples=outputs.shape[0]
	print(no_samples)
	inputs=shuffle_df.drop(['num','thalach','exang','oldpeak','Place','thalach','Age','Sex','ID'],axis=1)
	va=outputs.tail(1).values
	inputs['trestbps']=inputs['trestbps'].fillna(np.mean(inputs['trestbps']))
	inputs['fbs']=inputs['fbs'].fillna(np.mean(inputs['fbs']))
	inputs['cp']=inputs['cp'].fillna(np.mean(inputs['cp']))
	inputs['chol']=inputs['chol'].fillna(np.mean(inputs['chol']))
	inputs['restecg']=inputs['restecg'].fillna(np.mean(inputs['restecg']))

	#scale data

	inputs=inputs/np.amax(inputs,axis=0)
	test_x=inputs.tail(92).values
	test_y=outputs.tail(92).values
	test_x=np.float32(test_x)
	test_y=np.float32(test_y)
	inputs=inputs.head(num_sapmles).values
	outputs=outputs.head(num_sapmles).values

	inputs=np.float32(inputs)
	outputs=np.float32(outputs)
	print(test_x.shape)

	return test_x,test_y