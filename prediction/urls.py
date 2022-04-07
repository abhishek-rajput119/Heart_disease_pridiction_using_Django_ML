from django.urls import path
from . import views
app_name='predict'
urlpatterns=[
path(r'^(?P<pk>\d+)$',views.PredictRisk,name='predict')
]
