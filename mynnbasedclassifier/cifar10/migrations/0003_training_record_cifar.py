# Generated by Django 2.2 on 2019-05-13 09:08

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('cifar10', '0002_delete_training_record_cifar'),
    ]

    operations = [
        migrations.CreateModel(
            name='training_record_cifar',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('record_cifar', models.CharField(default='No record', max_length=2048)),
            ],
        ),
    ]
