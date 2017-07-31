"""SCRM_App URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf import settings
from django.conf.urls import url, include
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth import views as auth_views
from scrm import views as app_views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^$',app_views.index,name='index'),
    url(r'^home/$',app_views.home,name='home'),
    url(r'^novaColeta/$',app_views.novaColeta,name='novaColeta'),
    url(r'^coletasEfetuadas/$',app_views.coletasEfetuadas,name='coletasEfetuadas'),
    url(r'^coletarTwitter/$',app_views.coletarTwitter,name='coletarTwitter'),
    url(r'^coletarFacebook/$',app_views.coletarFacebook,name='coletarFacebook'),
    url(r'^deletarColeta/$',app_views.deletarColeta,name='deletarColeta'),
    url(r'^novaAnalise/$',app_views.novaAnalise, name='novaAnalise'),
    url(r'^analisesEfetuadas/$',app_views.analisesEfetuadas, name='analisesEfetuadas'),
    url(r'^analisar/$',app_views.analisar, name='analisar'),
    url(r'^pol_table/$',app_views.pol_table, name='pol_table'),
    url(r'^twitter_table/$',app_views.twitter_table,name='twitter_table'),
    url(r'^twitter_trends/$',app_views.twitter_trends,name='twitter_trends'),
    url(r'^coletasOpts/$',app_views.list_extraction_opts,name='coletasOpts'),
    url(r'^updateColetaStatus/$',app_views.update_coleta_status,name='update_coleta_status'),
    url(r'^updateProcessStatus/$',app_views.update_process_status, name='update_process_status'),

    # url('', include('social_django.urls', namespace='social')),

    url(r'^login/$', auth_views.login, name='login'),
    url(r'^logout/$', auth_views.logout_then_login, name='logout'),
    url(r'^oauth/', include('social_django.urls', namespace='social')),
    url(r'^settings/$',app_views.settings, name='settings'),
    url(r'^settings/password/$', app_views.password, name='password'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
