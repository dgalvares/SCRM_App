from django.contrib import admin
from . import models

# Register your models here.

admin.site.register(models.Extracao)
admin.site.register(models.Processamento)
admin.site.register(models.DatasetTreito)
admin.site.register(models.AvaliacaoPolaridade)