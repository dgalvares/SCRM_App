import datetime
import pprint

import yweather
from urllib import request
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm, AdminPasswordChangeForm
from django.http import HttpResponse
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

from social_django.models import UserSocialAuth

from utils import twitter_api_auth
from utils import svm
from utils import twitter_stream_extraction
from utils.db_connection import MongodbConnector

from scrm.models import Extracao, Processamento, AvaliacaoPolaridade

import threading
import time


# Create your views here.
def index(request):
    return redirect(home)


@login_required
def home(request):
    return render(request, 'scrm/home.html')


@login_required
def settings(request):
    user = request.user

    try:
        twitter_login = user.social_auth.get(provider='twitter')
    except UserSocialAuth.DoesNotExist:
        twitter_login = None

    try:
        facebook_login = user.social_auth.get(provider='facebook')
    except UserSocialAuth.DoesNotExist:
        facebook_login = None

    can_disconnect = (user.social_auth.count() > 1 or user.has_usable_password())

    return render(request, 'scrm/settings.html', {
        'twitter_login': twitter_login,
        'facebook_login': facebook_login,
        'can_disconnect': can_disconnect
    })


@login_required
def password(request):
    if request.user.has_usable_password():
        PasswordForm = PasswordChangeForm
    else:
        PasswordForm = AdminPasswordChangeForm

    if request.method == 'POST':
        form = PasswordForm(request.user, request.POST)
        if form.is_valid():
            form.save()
            update_session_auth_hash(request, form.user)
            messages.success(request, 'Your password was successfully updated!')
            return redirect('password')
        else:
            messages.error(request, 'Please correct the error below.')
    else:
        form = PasswordForm(request.user)
    return render(request, 'scrm/password.html', {'form': form})


@login_required
def novaColeta(request):
    return render(request, 'scrm/coleta/nova_coleta.html')


@login_required
def coletasEfetuadas(request):
    user_id = request.user.id
    user = User.objects.get(pk=user_id)
    retorno = {}
    extracao_list = Extracao.objects.all().filter(usuario=user)
    retorno['extracao_list'] = extracao_list
    return render(request, 'scrm/coleta/coletas.html', {"retorno": retorno})


@login_required
def deletarColeta(request):
    extracao_id = request.POST.get('extracao_id')
    extracao = Extracao.objects.all().filter(id=extracao_id)
    db = MongodbConnector.connect_db('socialNet')
    collection = MongodbConnector.connect_collection(db, 'twitter')
    collection.delete_many({"extracao":str(extracao_id)})
    extracao.delete()
    return redirect(coletasEfetuadas)


@login_required
def novaAnalise(request):
    user_id = request.user.id
    user = User.objects.get(pk=user_id)
    retorno = {}
    extracao_list = Extracao.objects.all().filter(usuario=user, concluido=True)
    retorno['extracao_list'] = extracao_list
    if request.POST.get('extracao_id'):
        retorno['extracao_id'] = request.POST.get('extracao_id')
    return render(request, 'scrm/analise/nova_analise.html', {"retorno":retorno})


@login_required
def analisesEfetuadas(request):
    user_id = request.user.id
    user = User.objects.get(pk=user_id)
    retorno = {}
    processamento_list = Processamento.objects.all().filter(usuario=user)

    retorno['process_list'] = processamento_list
    return render(request, 'scrm/analise/analises.html',{"retorno":retorno})


@login_required
def coletarTwitter(request):
    filtro = request.POST.get('filtro')
    max_twitts = request.POST.get('max_posts')
    user_id = request.user.id
    user = User.objects.get(pk=user_id)
    extracao = Extracao()
    extracao.usuario = user
    extracao.termo = filtro
    extracao.instante = datetime.datetime.now()
    extracao.concluido = False

    extracao.fonte = 'twitter'
    extracao.save()
    print("run twitter extraction...")
    tse = twitter_stream_extraction
    t = threading.Thread(target=tse.extrair, args=(filtro, max_twitts, str(extracao.id), extracao))
    t.start()
    return redirect(coletasEfetuadas)


@login_required
def coletarFacebook(request):
    filtro = request.POST.get('filtro')
    user_id = request.user.id
    user = User.objects.get(pk=user_id)
    extracao = Extracao()
    extracao.usuario = user
    extracao.termo = filtro
    extracao.instante = time.time()
    extracao.concluido = False

    extracao.fonte = 'facebook'
    extracao.save()
    print("run facebook extraction...")


@login_required
def pol_table(request):
    process_id = request.POST.get('process_id')
    process = Processamento.objects.get(pk=process_id)
    pol_list = AvaliacaoPolaridade.objects.all().filter(processamento=process)
    html = render_to_string('scrm/analise/pol_table.html', {"pol_list":pol_list})
    return HttpResponse(html)


@login_required
def twitter_table(request):
    extracao_id = request.POST.get('extracao_id')
    db = MongodbConnector.connect_db('socialNet')
    collection = MongodbConnector.connect_collection(db, 'twitter')
    twitter_list = collection.find({"extracao": str(extracao_id)})
    html = render_to_string('scrm/coleta/twitter/table_resultados_twitter.html', {"twitts": twitter_list})
    return HttpResponse(html)


@login_required
def twitter_trends(request):
    client = yweather.Client()
    country_id = client.fetch_woeid('Brazil')
    api = twitter_api_auth.twitter_auth()
    response = api.trends_place(country_id)
    data = response[0]
    trends = data['trends']
    html = render_to_string('scrm/table_twitter_trends.html', {"trends": trends[:4]})
    return HttpResponse(html)


@login_required
def analisar(request):
    user_id = request.user.id
    user = User.objects.get(pk=user_id)
    processamento = Processamento()
    processamento.usuario = user
    processamento.data = datetime.datetime.now()
    extracao_id = request.POST.get('extracao_opt')
    extracao = Extracao.objects.get(pk=extracao_id)
    processamento.extracao = extracao
    print(extracao.id)
    processamento.save()
    db = MongodbConnector.connect_db('socialNet')
    collection = MongodbConnector.connect_collection(db, 'twitter')
    twitters = collection.find({"extracao": str(extracao.id)})
    colecao = []
    polaridades = []
    for twitt in twitters:
        colecao.append(twitt['text'])
        av_polaridade = AvaliacaoPolaridade()
        av_polaridade.processamento = processamento
        av_polaridade.frase = (twitt['text'])
        polaridades.append(av_polaridade)
    # predicted = svm.multiclass('teste', 'SVM', colecao)
    t = threading.Thread(target=svm.multiclass,args=('base teste','SVM',colecao,polaridades,processamento))
    print("start analise")
    t.start()
    # for item in predicted:
    #     avaliacao = AvaliacaoPolaridade()
    #     avaliacao.processamento = processamento
    #     avaliacao.polaridade = item

    return redirect(analisesEfetuadas)


@login_required
def list_extraction_opts(request):
    user_id = request.user.id
    user = User.objects.get(pk=user_id)
    extracao_list = Extracao.objects.all().filter(usuario=user, concluido=True)
    list = []
    for item in extracao_list:
        list.append(str(item.fonte)+": "+str(item.termo+"  "+str(item.quantidade)+" resultados"))
    html = render_to_string('scrm/select_options_list.html', {"list": list})
    return HttpResponse(html)


@login_required
def update_coleta_status(request):
    user_id = request.user.id
    extracao_id = request.POST.get('extracao_id')
    extracao = Extracao.objects.get(pk=extracao_id)
    db = MongodbConnector.connect_db('socialNet')
    collection = MongodbConnector.connect_collection(db, 'twitter')
    done = extracao.concluido
    fonte = extracao.fonte
    quantidade = collection.find({"extracao": extracao_id}).count()
    return JsonResponse({"done":done,"fonte":fonte,"quantidade":quantidade})


@login_required
def update_process_status(request):
    process_id = request.POST.get('process_id')
    process = Processamento.objects.get(pk=process_id)
    done = process.concluido
    return JsonResponse({"done":done})