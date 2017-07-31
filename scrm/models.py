from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class DatasetTreito(models.Model):
    nome = models.CharField(max_length=30)
    fonte = models.CharField(max_length=30)
    data = models.DateField(null=False)
    arquivo = models.FileField(upload_to='media/dataset')
    valido = models.BooleanField(default=False)

    def __str__(self):
        return self.nome


class Extracao(models.Model):
    usuario = models.ForeignKey(User)
    instante = models.DateTimeField()
    fonte = models.CharField(max_length=30)
    termo = models.CharField(max_length=50)
    concluido = models.BooleanField(default=False)
    quantidade = models.IntegerField(default=0)

    def __str__(self):
        return self.fonte + " : "+self.termo


class Processamento(models.Model):
    usuario = models.ForeignKey(User)
    extracao = models.ForeignKey(Extracao)
    data = models.DateTimeField()

    concluido = models.BooleanField(default=False)


class AvaliacaoPolaridade(models.Model):
    processamento = models.ForeignKey(Processamento)
    frase = models.TextField()
    polaridade = models.IntegerField()