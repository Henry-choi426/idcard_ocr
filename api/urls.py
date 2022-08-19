from django.urls import path
from django.views.generic import TemplateView
from . import views
from django.conf.urls.static import static
from django.conf import settings

app_name = 'api'
urlpatterns = [
    path('request', TemplateView.as_view(template_name='api/request.html'), name='request'),
    path('predict', views.predict, name='predict'),
    path('predict2', views.predict2, name='predict2'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
