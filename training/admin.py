from django.contrib import admin
from .models import Training_type,Training_location,Exercise_name,Training,Exercise
from .forms import TrainingForm, ExerciseForm


# Register your models here.

class TrainingAdmin(admin.ModelAdmin):
    list_display = ['training_type', 'location', 'training_date']
    form = TrainingForm
    list_filter = ['training_type', 'location']


class ExerciseAdmin(admin.ModelAdmin):
    list_display = ['training_ID', 'exercise', 'reps']
    form = ExerciseForm
    list_filter = ['training_ID', 'exercise']



admin.site.register(Training, TrainingAdmin)
admin.site.register(Training_type)
admin.site.register(Training_location)
admin.site.register(Exercise_name)
admin.site.register(Exercise, ExerciseAdmin)


