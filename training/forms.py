from django import forms 
from .models import  Exercise,  Training
from django.forms import ModelForm


class TrainingForm(ModelForm):
    class Meta:
        model = Training
        fields = [ 'training_type', 'location', 'training_date', 'training_time']
        labels = {'training_type': 'What kind of workout?',
                  'training_date': 'Date', 
                  'training_time': 'Time'}

class ExerciseForm(ModelForm):
    class Meta:
        model = Exercise
        fields = [ 'exercise', 'reps']


class Training_list_searchForm(ModelForm):
    class Meta: 
        model = Training
        fields = ['training_type']  


