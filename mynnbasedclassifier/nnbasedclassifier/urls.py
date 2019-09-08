from django.urls import include, path

from . import views

urlpatterns = [
    path('index/', views.index, name = 'name_index'),
    path('typeChoose/', views.typeChoose, name = 'name_datasetChoose'),
    path('mnist/', include(('mnist.urls', 'a'), namespace = 'namespace_mnist')),
    path('cifar10/', include(('cifar10.urls', 'a'), namespace = 'namespace_cifar10')),
    path('diy/', include(('diy.urls', 'a'), namespace = 'namespace_diy')),
]