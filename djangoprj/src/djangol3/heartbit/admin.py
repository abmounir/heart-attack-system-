from django.contrib import admin
from .models import Patient,Medecin,Analyses,Diagnostic,RendezVous

admin.site.register(Patient)
admin.site.register(Medecin)
admin.site.register(Analyses)
admin.site.register(RendezVous)
admin.site.register(Diagnostic)
