from django.urls import include, path

from . import views

urlpatterns = [
    path('index/', views.index),
    path('hitRate/', views.hitRate, name = 'name_hitRate'),
]