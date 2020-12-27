from dash.dependencies import Output
from django.shortcuts import render, redirect, get_object_or_404
from .forms import TrainingForm, ExerciseForm, Training_list_searchForm
from .models import Exercise_name, Training, Exercise

# import plot functions from plots.py 
from .plots import * 
from datetime import date



from django.contrib.auth.decorators import permission_required



def home(request):
    subtitle = 'Activity plot'
    title = 'Welcome to TRAIN WITH CanDyman'
    today = date.today()
    last_training = Training.objects.values().order_by('-training_date').first()['training_date']
    delta = today - last_training
    subsubtitle = f"It's been {delta.days} days since your last training!"
    context = { 'title': title,  'subtitle':subtitle, 'plot_act': display_years(get_training_days(), [datetime.datetime.now().year]), 'subsubtitle': subsubtitle }
    return render(request, 'home.html', context)


def add_training(request):
    title = 'Add Training'
    if request.user.is_authenticated:
        form = TrainingForm(request.POST or None)
        if form.is_valid():
            form.save()
            # render activity_plot and save 
            plot_acticityplot()
            return redirect('/add_exercise')
    else: 
        form = TrainingForm(request.POST or None)
        if form.is_valid():
            return redirect('/add_exercise')
    context = { 'title': title , 'form': form}
    return render(request, 'training_entry.html', context) 



def add_exercise(request):
    title = 'Add Exercise'
    if request.user.is_authenticated:
        form = ExerciseForm(request.POST or None, initial=Training.objects.values().first())
        if  request.method == 'POST' and 'save' in request.POST:
            if form.is_valid():
                form.save()
                return redirect('/add_exercise')
        if  request.method == 'POST' and 'finish' in request.POST:
            if form.is_valid():
                form.save()
                return redirect('/')
    else:
        form = ExerciseForm(request.POST or None, initial=Training.objects.values().first())
        if  request.method == 'POST' and 'save' in request.POST:
            if form.is_valid():
                return redirect('/add_exercise')
        if  request.method == 'POST' and 'finish' in request.POST:
            if form.is_valid():
                return redirect('/')

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
    queryset = Exercise.objects.filter(training_ID = id)

    context = {'title': title, 'queryset':queryset}
    return render(request, 'exercise_list.html', context)
    

@permission_required('training.add_choice', login_url='/')
def todo(request):
    return render(request, 'Todo.html')



def dashboard(request):
    context = {'plot1': plot_histograms_exercise(), 
               'plot2': plot_histograms_reps(), 
               'plot3': plot_histograms_reppset(), 
               'plot4': plot_histograms_days(),
               'plot5': plot_histograms_types()}
    return render(request, 'dashboard.html', context)
