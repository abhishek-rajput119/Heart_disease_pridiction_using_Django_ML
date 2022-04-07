from django.urls import path
from . import views
app_name = 'accounts'
urlpatterns=[
    path(r'^register/$', views.register, name='register'),
    path(r'^logout/$',views.user_logout,name='logout'),
    path(r'^profile/(?P<pk>\d+)/$', views.ProfileDetailView.as_view(), name='profile'),

]
