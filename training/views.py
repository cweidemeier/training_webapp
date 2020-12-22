from django.shortcuts import render, redirect
from .forms import TrainingForm, ExerciseForm, Training_list_searchForm
from .models import Exercise_name, Training, Exercise
# Create your views here.


# graph test 
from io import StringIO
import pandas as pd
import calmap
from django.db.models import Sum
import pandas as pd

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
    fig.figure.savefig('./training/static/img/activity_plot.svg', transparent=True, bbox_inches='tight')



def plot_histograms():
    exercise = []
    reps_sum = []

    # # histogram for total of reps per exercise 
    # for e in range(len(Exercise_name.objects.all())):
    #     exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
    #     rep = Exercise.objects.filter(exercise = e+1).aggregate(Sum('reps'))['reps__sum']
    #     reps_sum.append(0 if rep is None else rep)

    # histogram for total of exercises done - vllt. noch ein fehler ? 
    for e in range(len(Exercise_name.objects.all())):
        exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
        rep = Exercise.objects.filter(exercise_id = e+1).count()
        reps_sum.append(0 if rep is None else rep)

    # convert to df 
    df = pd.DataFrame({'exercise name': exercise,'reps': reps_sum})
    df.set_index('exercise name', inplace=True)
    df.sort_values('reps', inplace=True, ascending = False)
    
    #plot and safe 
    imgdata = StringIO()
    fig = df.plot(y='reps', kind='bar', legend=False)
    fig.figure.savefig('./training/static/img/exercise_histo.svg', transparent=True, bbox_inches='tight')




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
       

    context = { 'title': title , 'form': form}
    return render(request, 'exercise_entry.html', context) 



def training_list(request):
    title = 'List of all trainings'
    queryset = Training.objects.all()
    form = Training_list_searchForm(request.POST or None)

    if request.method == 'POST':
        queryset =  Training.objects.all().filter(training_type = form['training_type'].value())

    context = { 'title': title, 'form':form,  'queryset': queryset}
    return render(request, 'training_list.html', context)



def training_list_redirect(request, id=None):
    title = f'Training: {id}'
    queryset = Exercise.objects.filter(training_ID= id)

    context = {'title': title, 'queryset':queryset}
    return render(request, 'exercise_list.html', context)
    


def dashboard(request):
    title = 'Interactive Dashboard - soon'
    plot_histograms()
    context = { 'title': title }
    return render(request, 'dashboard.html', context)
