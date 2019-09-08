from django.db import models

# Create your models here.

class training_record_cifar(models.Model):
    record_cifar = models.CharField(max_length = 2048, default = "No record")