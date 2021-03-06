# Generated by Django 2.2.1 on 2019-07-12 15:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('heartbit', '0003_auto_20190604_0944'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='patient',
            name='medecin',
        ),
        migrations.AddField(
            model_name='analyses',
            name='exang',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='analyses',
            name='oldpeak',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='analyses',
            name='thalach',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='rendezvous',
            name='confirmer_Rendez_Vous',
            field=models.CharField(choices=[('c', 'Ok'), ('n', 'No')], default='', max_length=1),
        ),
    ]
