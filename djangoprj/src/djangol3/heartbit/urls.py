from django.urls import path
from . import views


urlpatterns = [
    path('', views.home,name='heartbit-home'),
    path('login/', views.login,name='heartbit-login'),
    path('logout-545145/', views.logout,name='heartbit-logout'),
     path('medecin/', views.showMedecin,name='heartbit-meddispo'),
    path('medecin/<medecin_id>', views.medecin,name='heartbit-medecin'),
	path('Dashboard/',views.predict,name='heartbit-predict')

]
