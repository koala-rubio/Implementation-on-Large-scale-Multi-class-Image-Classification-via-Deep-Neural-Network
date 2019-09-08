from django.urls import include, path

from . import views

urlpatterns = [
    path('index/', views.index),
    path('classifyResult/', views.classifyResult, name = 'name_classifyResult'),
]