from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    path('index.html', views.index),
    path('detect/', views.detect, name='detect'),

    path('about/', views.about_view, name='about'),
    path('about.html', views.about_view),
    # path('detect/', DetectView.as_view(), name='detect'),

    path('anup/', views.anup_view, name="anup"),
    path('anup.html', views.anup_view),

    path('saurav/', views.saurav_view, name="saurav"),
    path('saurav.html', views.saurav_view),
    
    path('admin-panel/', views.admin_panel_view, name='admin_panel'),
    path('set_model/', views.set_model, name='set_model'),
    

] + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])