"""
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]"""
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from. import views

urlpatterns = [
   
    path('', views.home, name = 'home'),
    path('Import', views.Import, name = 'Import'),
        
]+ static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
