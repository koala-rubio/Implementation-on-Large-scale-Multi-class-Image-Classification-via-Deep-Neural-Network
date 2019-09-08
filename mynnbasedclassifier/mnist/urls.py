from django.urls import include, path

from . import views

urlpatterns = [
    path('index/', views.index),
    path('typeChoose/', views.typeChoose, name = 'name_typeChoose'),
    path('hitRate/', views.hitRate, name = 'name_hitRate'),
]