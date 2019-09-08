from django.shortcuts import render

from . import models

from django.http import HttpResponse

# Create your views here.

def index(request):
    return render(request, 'entrance/index.html')

def typeChoose(request):
    da_type = request.POST.get('da_type', 'MNIST')
    if da_type == 'MNIST':
        return render(request, 'mnist/index.html')
    elif da_type == 'CIFAR10':
        return render(request, 'cifar10/index.html')
    else:
        return render(request, 'diy/index.html')
        #return HttpResponse('Welcome, this method is '+da_type+'')