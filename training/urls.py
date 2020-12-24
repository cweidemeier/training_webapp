from django.contrib import admin
from django.urls import  path
from . import views

from training.dash_apps.finished_apps import simpleexample


urlpatterns = [
    path('', views.home, name = 'home'),
    path('add_training', views.add_training, name = 'add_training'),
    path('add_exercise', views.add_exercise, name = 'add_exercise'),
    path('training_list', views.training_list, name = 'list'),
    path('todo', views.todo, name = 'todo'),
    path('training_list/<int:id>', views.training_list_redirect, name = 'ex'),
    path('dashboard', views.dashboard, name = 'dash'),
    

]