from django import forms 
from .models import  Exercise,  Training
from django.forms import ModelForm


class TrainingForm(ModelForm):
    class Meta:
        model = Training
        fields = [ 'training_type', 'location', 'training_date']
    

class ExerciseForm(ModelForm):
    class Meta:
        model = Exercise
        fields = [ 'exercise', 'reps']


class Training_list_searchForm(ModelForm):
    class Meta: 
        model = Training
        fields = ['training_type']   #, 'training_day']  #vllt ist date interessant für später


