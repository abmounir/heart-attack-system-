from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from time import gmtime, strftime
import mysql.connector
import MySQLdb
from django.contrib import auth
from mysql.connector import Error
from .models import Patient,Analyses,Medecin,RendezVous,Diagnostic
from django.contrib import messages
from .forms import UserRegisterForm,AnalyseForm
#from .prj1.datta import data_prepro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.reset_default_graph()
numL=10
def Forward(dictLayer,inputs):
	lst=[]
	list_all_outs=[]
	o1=tf.nn.sigmoid(tf.add(tf.matmul(inputs,dictLayer['layer0'][0]),dictLayer['layer0'][1]))
	lst.append(o1)
	list_all_outs.append(o1)
	del(dictLayer['layer0'])
	print(list_all_outs)
	for k in dictLayer:
		o=tf.nn.sigmoid(tf.add(tf.matmul(lst[0],dictLayer[k][0]),dictLayer[k][1]))
		print(o)
		list_all_outs.append(o)
		del(lst[0])
		lst.append(o)  
    
	return list_all_outs[-1]

def home(request):
	if request.method == 'POST':
		p=Patient.objects.create(username=request.POST.get('username'),nom_patient=request.POST.get('nom'),prenom_patient=request.POST.get('prenom'),email=request.POST.get('email'),adresse=request.POST.get('adresse'),sexe=request.POST.get('sexe'),age=request.POST.get('age'),telephone=request.POST.get('telephone'),password=request.POST.get('password'))
		if p :
			messages.success(request,f'Account created successfully.')
		else :
			messages.warning(request,f'Wrong info. Try again!')	
	return render(request,'heartbit/home.html')

def login(request):
	if request.method=='POST':
		p=Patient.objects.get(username=request.POST.get('username'))
		if p :
			if p.password == request.POST.get('password') :
				request.session['id']=p.id
				request.session['username']=p.username
				request.session['password']=p.password
				return redirect('heartbit-home')
			else:
				return HttpResponse('wrong info ')

	return render(request,'heartbit/login.html')


def CheckRdv(request,idm):
	confirm=''
	d=''
	rdv_list=[]
	allrdv=RendezVous.objects.filter(patient_id_id=request.session['id']).order_by('id').last()	
	#r=RendezVous.objects.get(patient_id_id=request.session['id'],medecin_id_id=idm)
	if allrdv :
		confirm=allrdv.confirmer_Rendez_Vous
		d=allrdv.date
		print(confirm)
	return confirm,d

def logout(request):
	auth.logout(request)
	return render(request,'heartbit/logout.html')	

def medecin(request,medecin_id):
	confirm=''
	medecins=Medecin.objects.get(id=medecin_id)
	if request.method == 'POST':
		idm=request.POST.get('rdv')
		date=request.POST.get('date')
		if idm and date :
			print('idm : '+(idm))
			RendezVous.objects.create(medecin_id_id=idm,patient_id_id=request.session['id'],date=date)

		else:
			print('no value')
		
	confirm,d=CheckRdv(request,medecin_id)

	context={'medecin':medecins,'confirm':confirm,'date':d}

	return render(request,'heartbit/medecin.html',context)

def showMedecin(request):
	med=Medecin.objects.all()
	context={'medecins':med}
	
	return render(request,'heartbit/meddispo.html',context)


#prediction
def predict(request):
	out=np.array([])
	y_f=''
	if request.method=='POST':
		cp=request.POST.get('cp')
		trestbps=request.POST.get('trestbps')
		fbs=request.POST.get('fbs')
		restecg=request.POST.get('restecg')
		chol=request.POST.get('chol')
		thalach=request.POST.get('thalach')
		exang=request.POST.get('exang')
		oldpeak=request.POST.get('oldpeak')
		p=Patient.objects.get(id=request.session['id'])
		if p.sexe == 'm':
			sexe=1
		elif p.sexe=='f':
			sexe=0

		age=p.age
		if int(age) < 75 :
			age=age/75
		else :
			age=1			
		anal=Analyses.objects.create(cp=request.POST.get('cp'),trestbps=request.POST.get('trestbps'),fbs=request.POST.get('fbs'),restecg=request.POST.get('restecg'),chol=request.POST.get('chol'),thalach=thalach,exang=exang,oldpeak=oldpeak)

		
		if anal :
			anal_id=anal.id
	
			
		if float(fbs) > 120.0 :
			(fbs)=1.0
		else:
			(fbs)=0.0
			
			#out=Forward(pr_dict,arr)
			#out=sess.run(out)
			#out=out.reshape(-1)
			#out=out[0]
			
			#print(out)
			#print('cp :'+str(cp))
			#print('ecg :'+str(restecg))
		arr= np.array([[float(age),float(sexe),float(cp)/4.0,float(trestbps)/200.0,float(chol)/603.0,float(fbs),float(restecg)/2.0,float(thalach)/190.0,float(exang),float(oldpeak)/5.0]],dtype=np.float32)

		#ensl = pickle.load(open('C:/Users/DELL/Desktop/djangoprj/src/djangol3/heartbit/prj1/ensemble4.sav', 'rb'))
		#res1 = ensl.predict(arr)
		#rf2=pickle.load(open('C:/Users/DELL/Desktop/djangoprj/src/djangol3/heartbit/prj1/rf.sav', 'rb'))
		#res2=rf2.predict(arr)

		#ar=np.array([])
		#ar=(res1+res2)/2
		#y_f=np.array([])
		#ar=ar.round()

		#ar=ar.reshape(1,1)
		#y_f=(ar+out)/2
		#y_f=y_f.reshape(-1)
		#y_f=y_f[0]
		

		
		#test=np.array([[float(26)/75,float(0),float(3)/4.0,float(120)/200.0,float(250)/603.0,float(0),float(1)/2.0,float(111)/190.0,float(1),float(2)/5.0]],dtype=np.float32)
		nv=pickle.load(open('C:/Users/DELL/Desktop/djangoprj/src/djangol3/heartbit/prj1/nv5.sav','rb'))
		ensemble=pickle.load(open('C:/Users/DELL/Desktop/djangoprj/src/djangol3/heartbit/prj1/ensmble.sav','rb'))
		y_el=ensemble.predict_proba(arr)
		if y_el[0][1]>=0.50 :
			y_f=nv.predict_proba(arr)
		else :
			y_f=ensemble.predict_proba(arr)	

		y_f=y_f[0][1]*100
		diag=Diagnostic.objects.create(analyse_id_id=anal_id,patient_id_id=request.session['id'],diag_result=y_f)
		
		print('final output :'+str(y_f)+'%')

	return render(request,'heartbit/dashboard.html',{'out':y_f})