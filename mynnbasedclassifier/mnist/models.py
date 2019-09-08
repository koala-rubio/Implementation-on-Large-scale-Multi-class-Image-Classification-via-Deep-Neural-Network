from django.db import models

# Create your models here.

class training_record_mnist(models.Model):
    record_mnist = models.CharField(max_length = 2048, default = "No record")