from django.db import models
from django import forms

class Medecin(models.Model):
	nom_medecin=models.CharField(max_length=30)
	prenom_medecin=models.CharField(max_length=30)
	email=models.EmailField(max_length=30)
	adresse_cabinet=models.CharField(max_length=100)
	telephone=models.CharField(max_length=100,default='')
	password = models.CharField(max_length=100,default='')

class Patient(models.Model):
	username=models.CharField(max_length=30,default='')
	nom_patient=models.CharField(max_length=30)
	prenom_patient=models.CharField(max_length=30)
	age=models.IntegerField(default=0)
	email=models.EmailField(max_length=30)
	adresse=models.CharField(max_length=100)
	telephone=models.CharField(max_length=100,default='')
	SEX_CHOICES = (
        ('F', 'Female',),
        ('M', 'Male',)
    )

	sexe = models.CharField(
        max_length=1,
        choices=SEX_CHOICES,
    )
	password = models.CharField(max_length=100,default='')
	
	#medecin = models.ForeignKey(Medecin,on_delete=models.CASCADE,default='1')

class Analyses(models.Model):
	
	cp=models.IntegerField(default=0)
	trestbps=models.IntegerField(default=0)
	fbs=models.IntegerField(default=0)
	restecg=models.IntegerField(default=0)
	chol=models.IntegerField(default=0)
	thalach=models.IntegerField(default=0)
	exang=models.IntegerField(default=0)
	oldpeak=models.IntegerField(default=0)

class Diagnostic(models.Model):
	analyse_id=  models.ForeignKey(Analyses,on_delete=models.CASCADE)
	patient_id = models.ForeignKey(Patient,on_delete=models.CASCADE)
	diag_result= models.FloatField(default=0.0)

	
class RendezVous(models.Model):
	patient_id = models.ForeignKey(Patient,on_delete=models.CASCADE)
	medecin_id = models.ForeignKey(Medecin,on_delete=models.CASCADE)
	date = models.DateField(max_length=8)
	conf=(('c','Ok'),('n','No'))
	confirmer_Rendez_Vous=models.CharField(max_length=1,choices=conf,default='')

