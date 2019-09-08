# Generated by Django 2.2 on 2019-05-13 09:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mnist', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='training_record_mnist',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('record_mnist', models.CharField(default='No record', max_length=2048)),
            ],
        ),
        migrations.DeleteModel(
            name='training_record',
        ),
    ]
