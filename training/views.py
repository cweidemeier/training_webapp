
from django.shortcuts import render, redirect, get_object_or_404
from .forms import TrainingForm, ExerciseForm, Training_list_searchForm
from .models import Exercise_name, Training, Exercise

# import plot functions from plots.py 
from .plots import * 
from datetime import date
from datetime import datetime 

from django.contrib.auth.decorators import login_required
from django.contrib.auth.decorators import permission_required


def home(request):
    if request.user.is_authenticated:
        title = f'Welcome {request.user}'
        subsubsubtitle = ''
        login = ''
        query = Training.objects.all().filter(user_name = request.user)
        if len(query) != 0:
            last_training = query.values().order_by('-training_date').first()['training_date']
        else: 
            last_training = date.today()
    else: 
        title = 'Welcome anonymous user'
        subsubsubtitle = 'You will see data of a dummy user'
        login = '<a class="nav-link" style="color: black" href="/accounts/login">Login to see your workout progress</a>'
        query = Training.objects.all().filter(user_name = 'test_user')
        last_training = query.values().order_by('-training_date').first()['training_date']
    

    today = date.today()
    delta = today - last_training
    if delta.days == 1: 
        subtitle = f"It's been {delta.days} day since your last training!"
    else: 
        subtitle = f"It's been {delta.days} days since your last training!"

    context = { 'title': title,  
    'subtitle':subtitle, 
    'subsubsubtitle': subsubsubtitle,
    'login': login,
    'plot_act': display_years(get_training_days(request), [datetime.now().year], request),
    'username': request.user }
    return render(request, 'base.html', context)




def add_training(request):
    title = 'Add Workout'
    if request.user.is_authenticated:
        form = TrainingForm(request.POST or None)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.user_name = request.user
            obj.save()
            return redirect('/add_exercise')
    else: 
        form = TrainingForm(request.POST or None)
        if form.is_valid():
            return redirect('/add_exercise')

    context = { 'title': title , 'form': form, 'username': request.user}
    return render(request, 'training_entry.html', context) 


def add_exercise(request):
    title = 'Add Exercise to last Workout'
    if request.user.is_authenticated:
        query = Training.objects.all().filter(user_name = request.user)
        form = ExerciseForm(request.POST or None, initial=query.values().first())
        if  request.method == 'POST' and 'save' in request.POST:
            if form.is_valid():
                obj = form.save(commit=False)
                obj.training_ID = Training.objects.get(training_ID = query.values().first()['training_ID'])
                obj.user_name = request.user
                obj.save()
                return redirect('/add_exercise')
        if  request.method == 'POST' and 'finish' in request.POST:
            if form.is_valid():
                obj = form.save(commit=False)
                obj.training_ID = Training.objects.get(training_ID = query.values().first()['training_ID'])
                obj.user_name = request.user
                obj.save()
                return redirect('/')

    else:
        form = ExerciseForm(request.POST or None, initial=Training.objects.values().first())
        if  request.method == 'POST' and 'save' in request.POST:
            if form.is_valid():
                return redirect('/add_exercise')
        if  request.method == 'POST' and 'finish' in request.POST:
            if form.is_valid():
                return redirect('/')

    context = { 'title': title , 'form': form, 'username': request.user}
    return render(request, 'exercise_entry.html', context) 


def training_list(request):
    title = 'Your Workouts'
    if request.user.is_authenticated:
        queryset = Training.objects.all().order_by('-training_date').filter(user_name=request.user)
        form = Training_list_searchForm(request.POST or None)
        if request.method == 'POST':
            queryset = queryset.filter(training_type = form['training_type'].value())
    else: 
        queryset = Training.objects.all().order_by('-training_date').filter(user_name='test_user')
        form = Training_list_searchForm(request.POST or None)
        if request.method == 'POST':
            queryset = queryset.filter(training_type = form['training_type'].value())

    context =  {'title': title, 
                'form':form,  
                'queryset': queryset, 
                'username': request.user}
    return render(request, 'training_list.html', context)


def training_edit(request, id=None, username= None):  
    form = ExerciseForm(request.POST or None)
    a = Training.objects.get(training_ID = id)
    if request.user.is_authenticated:
        if  request.method == 'POST':
            if form.is_valid():
                instance = form.save(commit=False)
                instance.training_ID = a
                instance.user_name = username
                instance.save()
                return redirect(f'/training_list/{username}/{id}')
                
    context = {'title': 'Add Exercise',
               'form': form,
               'username': request.user}
    return render(request, "exercise_entry_2.html", context)


@login_required(login_url='/training_list')
def training_delete(request, id=None):
   instance = get_object_or_404(Training, training_ID=id)
   instance.delete()
   return redirect("/training_list")


def training_list_redirect(request, id=None, username = None):
    username = request.user
    queryset = Exercise.objects.filter(training_ID = id)
    x = Training_type.objects.values().filter(id = Training.objects.values().filter(training_ID = id)[0]['training_type_id'])[0]['training_type']
    y = Training.objects.values().filter(training_ID = Training.objects.values().filter(training_ID = id)[0]['training_ID'])[0]['training_date'].strftime(' %d. %b %Y')
    title = f'{x}, {y}'

    link = f'/training_list/{username}/{id}_edit'
    context = {'title': title, 'queryset':queryset, "link": link, 'username': request.user}
    return render(request, 'exercise_list.html', context)
    

@login_required(login_url='/training_list')
def exercise_delete(request, id, username):
    training_id = Exercise.objects.values().filter(id = id)[0]['training_ID_id']
    instance = get_object_or_404(Exercise, id = id)
    instance.delete()
    return redirect(f"/training_list/{username}/{training_id}")


@permission_required('training.add_choice', login_url='/')
def todo(request):
    return render(request, 'Todo.html',{'username': request.user} )


def dashboard(request): 
    title = 'Workout Statistics'
    context = {'plot3': plot_histograms_exercise(request), 
               'plot5': plot_histograms_reps(request), 
               #'plot4': plot_histograms_reppset(request), 
               'plot1': plot_pie_types(request),
               'plot2': plot_heatmap_week(request),
               'plot6': reps_sets(request),
               'title':title,
               'username': request.user}
    return render(request, 'dashboard.html', context)


def dashboard2(request): 
    title = 'Workout statistics by exercise'
    context = {'plot1': reps_sets(request),
               'title':title,
               'username': request.user}
    return render(request, 'dashboard2.html', context)

