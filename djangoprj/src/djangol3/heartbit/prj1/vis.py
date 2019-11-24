import numpy as np
import matplotlib.pyplot as plt

names=['Momentum','SGD','NAG','AdaDelta']
dict_loss={}
for name in names :
	loss_file=open(name+'_loss.txt','r')
	loss_list=[]
	for i in loss_file :
		loss_list.append(float(i.replace('\n','')))
	dict_loss[name]=loss_list
	

#print(dict_loss)		

epl=[i for i in range(30000)]

colors=['b','y','g','r']
c=0
for i in dict_loss:
	plt.plot(epl,dict_loss[i],colors[c],label=i)
	c+=1

plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Erreur')
plt.show()
