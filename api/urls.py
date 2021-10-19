from django.urls import path
from django.views.generic import TemplateView
from . import views

app_name = 'api'
urlpatterns = [
    path('request', TemplateView.as_view(template_name='api/request.html'), name='request'),
    path('predict', views.predict, name='predict'),
    
    path('request2', TemplateView.as_view(template_name='api/request2.html'), name='request2'),
    path('predict2', views.predict2, name='predict2'),
    
]