from django.contrib import admin
from django.urls import  path
from . import views


urlpatterns = [
    path('', views.home, name = 'home'),
    path('add_training', views.add_training, name = 'add_training'),
    path('add_exercise', views.add_exercise, name = 'add_exercise'),
    path('training_list', views.training_list, name = 'list'),
    path('todo', views.todo, name = 'todo'),
    path('training_list/<str:username>/<int:id>', views.training_list_redirect, name = 'ex'),
    path('dashboard', views.dashboard, name = 'dash'),

    path('training_list/<int:id>_delete', views.training_delete, name = 'delete_training'),
    path('training_list/<str:username>/<int:id>_delete', views.exercise_delete, name = 'delete_exercise'),
    path('training_list/<str:username>/<int:id>_edit', views.training_edit, name = 'training_edit'),
] 