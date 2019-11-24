from django import forms
from django.contrib.auth.forms import UserCreationForm
from heartbit.models import Patient,Medecin,Analyses
from django.contrib.auth.models import User
from django.db import models

SEX_CHOICES = (
        ('F', 'Female',),
        ('M', 'Male',)
    )

class UserRegisterForm(UserCreationForm):
	nom_patient=forms.CharField(max_length=30)
	prenom_patient=forms.CharField(max_length=30)
	age=forms.IntegerField()
	email=forms.EmailField(max_length=30)
	adresse=forms.CharField(max_length=100)
	telephone=forms.CharField(max_length=100)
	sexe = forms.ChoiceField(choices = SEX_CHOICES, label="", initial='', widget=forms.Select(), required=True)
	class Meta:

		model=User
		fields=['username','nom_patient','prenom_patient','age','sexe','email','adresse','telephone']



class AnalyseForm(forms.Form):
	cp=forms.IntegerField()
	trestbps=forms.IntegerField()
	fbs=forms.IntegerField()
	restecg=forms.IntegerField()
	chol=forms.IntegerField()
	patient=forms.IntegerField()


