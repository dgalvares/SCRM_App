# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-06-16 03:51
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scrm', '0002_extracao_concluida'),
    ]

    operations = [
        migrations.RenameField(
            model_name='extracao',
            old_name='concluida',
            new_name='concluido',
        ),
        migrations.AddField(
            model_name='processamento',
            name='concluido',
            field=models.BooleanField(default=False),
        ),
    ]
