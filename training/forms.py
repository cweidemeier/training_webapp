from django import forms 
from .models import  Exercise,  Training
from django.forms import ModelForm
from time import strftime

class TrainingForm(ModelForm):
    class Meta:
        model = Training
        fields = ['training_type', 'location', 'training_date', 'training_time']
        labels = {'training_type': 'What kind of workout?',
                  'training_date': 'Date', 
                  'training_time': 'Time'}
    # to update time in form, when page is refreshed. 
    def __init__(self, *args, **kwargs):
        kwargs.update(initial={
            'training_time': strftime('%H:%M')
        })
        super(TrainingForm, self).__init__(*args, **kwargs)


class ExerciseForm(ModelForm):
    class Meta:
        model = Exercise
        fields = [ 'exercise', 'reps']


class Training_list_searchForm(ModelForm):
    class Meta: 
        model = Training
        fields = ['training_type']  


