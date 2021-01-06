from django.contrib import admin
from django.db import models
from datetime import date
from datetime import datetime
from django.db.models.deletion import CASCADE 






class Training_type(models.Model):
    training_type = models.CharField(max_length=30, blank=True)
    user_name = models.CharField(max_length=30) 

    def __str__(self):
        return self.training_type


class Training_location(models.Model):
    location = models.CharField(max_length=30, blank = True)
    user_name = models.CharField(max_length=30)
    def __str__(self):
        return self.location


class Exercise_name(models.Model):
    exercise_name = models.CharField(max_length=50)
    user_name = models.CharField(max_length=30)
    
    def __str__(self):
        return str(self.exercise_name)



class Training(models.Model):
    training_ID = models.AutoField(primary_key=True)
    location = models.ForeignKey(Training_location, on_delete=CASCADE, default = 2)
    training_date = models.DateField(default=date.today, auto_now=False)
    training_time = models.TimeField(default = datetime.now().time(), auto_now = False)
    training_type = models.ForeignKey(Training_type, blank=False, null=True, on_delete=CASCADE)
    user_name = models.CharField(max_length=30)

    class Meta:
        ordering = ('-training_ID',)

    def __str__(self):
        return str(self.training_ID)



class Exercise(models.Model): 
    training_ID = models.ForeignKey(Training, on_delete=CASCADE, null=True) 
    exercise = models.ForeignKey(Exercise_name, on_delete=CASCADE, null=True)
    reps = models.IntegerField(choices=list(zip(range(1, 21), range(1, 21))), unique=False)
    user_name = models.CharField(max_length=30)

    def __str__(self):
        return str(self.exercise)





