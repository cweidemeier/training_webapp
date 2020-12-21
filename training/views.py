from django.shortcuts import render, redirect
from .forms import TrainingForm, ExerciseForm, Training_list_searchForm
from .models import Training, Exercise
# Create your views here.


# graph test 
from io import StringIO
import pandas as pd
import calmap



#####################################
## avticity plot 
#####################################


def plot_acticityplot():
    # create pandas series with all training dates from database / query 
    dates = []
    for i in range(len(Training.objects.values('training_date'))):
        dates.append(Training.objects.values('training_date')[i]['training_date'])
    events = pd.Series(dates)

    # reindex series, such that the index is a DatetimeIndex 
    temp_series = pd.DatetimeIndex(events.values)
    events = events.reindex(temp_series)

    # reference the type of training for each date
    for i in range(len(dates)): 
         events.loc[dates[i]] = Training.objects.values('training_type')[i]['training_type']
    
    # plot and safe image
    imgdata = StringIO()
    fig = calmap.yearplot(events, dayticks=[0, 2, 4, 6])
    fig.figure.savefig('./training/static/img/test.svg', transparent=True, bbox_inches='tight')

    # imgdata.seek(0)
    # data = imgdata.getvalue()
    # return data 



#####################################


def home(request):
    subtitle = 'Activity plot'
    title = 'Welcome to TRAIN WITH CAN'
     
    context = { 'title': title,  'subtitle':subtitle }
    return render(request, 'home.html', context)


def add_training(request):
    title = 'Add Training'
    form = TrainingForm(request.POST or None)
    if form.is_valid():
        form.save()
        # render activity_plot and save 
        plot_acticityplot()
        return redirect('/add_exercise')

    context = { 'title': title , 'form': form}
    return render(request, 'training_entry.html', context) 



def add_exercise(request):
    title = 'Add Exercise'
    form = ExerciseForm(request.POST or None, initial=Training.objects.values().first())
    if form.is_valid():
        form.save()
        return redirect('/add_exercise')
        # add button: redirect to training_list. muss man in html machen. 

    context = { 'title': title , 'form': form}
    return render(request, 'exercise_entry.html', context) 



def training_list(request):
    title = 'List of all trainings'
    queryset = Training.objects.all()
    form = Training_list_searchForm(request.POST or None)
    context = { 'title': title, 'form':form,  'queryset': queryset}
    
    if request.method == 'POST':
        queryset = Training.objects.all().order_by('-training_day').filter(training_type__icontains=form['training_type'].value())
        context = {'title': title, 'queryset':queryset, 'form':form}
    return render(request, 'training_list.html', context)



def training_list_redirect(request, id=None):
    title = f'Training: {id}'
    queryset = Exercise.objects.filter(training_ID= id)

    context = {'title': title, 'queryset':queryset}
    return render(request, 'exercise_list.html', context)
    


def dashboard(request):
    
    title = 'There will be a dashboard soon'
    context = { 'title': title }
    return render(request, 'dashboard.html', context)
