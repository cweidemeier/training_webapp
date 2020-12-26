from .models import Exercise_name, Training, Exercise, Training_type


# graph test 
import pandas as pd
import calmap
from django.db.models import Sum


# plotly 
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go





def plot_acticityplot():
    # create pandas series with all training dates from database / query 
    dates = []
    for i in range(len(Training.objects.values('training_date'))):
        dates.append(Training.objects.values('training_date')[i]['training_date'])
    events = pd.Series(dates)

    # reindex series, such that the index is a DatetimeIndex - necessary for calmap
    temp_series = pd.DatetimeIndex(events.values)
    events = events.reindex(temp_series)

    # reference the type of training for each date
    for i in range(len(dates)): 
         events.loc[dates[i]] = Training.objects.values('training_type')[i]['training_type']
    
    # plot and safe image
    fig = calmap.yearplot(events, dayticks=[0, 2, 4, 6])
    fig.figure.savefig('./training/static/img/activity_plot.svg', transparent=True, bbox_inches='tight')




def plot_histograms_exercise():
    exercise = []
    reps_sum = []

    # histogram for total of exercises done - vllt. noch ein fehler ? 
    for e in range(len(Exercise_name.objects.all())):
        exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
        rep = Exercise.objects.filter(exercise_id = e+1).count()
        reps_sum.append(0 if rep is None else rep)

    # convert to df 
    df = pd.DataFrame({'exercise name': exercise,'reps': reps_sum})
    df.sort_values('reps', inplace=True, ascending = False)
    
    #plot and save 
    fig = px.bar(df, x='exercise name', y='reps', opacity = 0.7, title= 'Total number of sets per exercise:', hover_name = 'exercise name', hover_data = {'reps': False, 'exercise name': False}, text = 'reps')
    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       
       })
    # second attempt with plotly go. 
    # my_data = [go.Bar( x = df['exercise name'], y = df.reps)]
    # my_layout = go.Layout({"title": "Views by publisher",
    #                        "yaxis": {"title":"Views"},
    #                        "xaxis": {"title":"Publisher"},
    #                        "showlegend": False})

    #fig = go.Figure(data = my_data, layout = my_layout)

    #py.iplot(fig)
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div


def plot_histograms_reps():
    exercise = []
    reps_sum = []

    # # histogram for total of reps per exercise 
    for e in range(len(Exercise_name.objects.all())):
        exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
        rep = Exercise.objects.filter(exercise = e+1).aggregate(Sum('reps'))['reps__sum']
        reps_sum.append(0 if rep is None else rep)

    # convert to df 
    df = pd.DataFrame({'exercise name': exercise,'reps': reps_sum})
    df.sort_values('reps', inplace=True, ascending = False)

    #plot and save 
    fig = px.bar(df, x='exercise name', y='reps', opacity = 0.7, title= 'Total number of reps per exercise:', hover_name = 'exercise name', hover_data = {'reps': False, 'exercise name': False}, text = 'reps')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        
        })
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div


def plot_histograms_reppset():
    exercise = []
    reps_sum = []

    # # histogram for total of reps per exercise 
    for e in range(len(Exercise_name.objects.all())):
        exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
        total_rep = Exercise.objects.filter(exercise = e+1).aggregate(Sum('reps'))['reps__sum']
        sets = Exercise.objects.filter(exercise_id = e+1).count()
        if sets == None or total_rep == None:
            reppset = 0
        else: 
            reppset = round(total_rep/sets, 2)
        reps_sum.append(0 if reppset is None else reppset)

    # convert to df 
    df = pd.DataFrame({'exercise name': exercise,'reps': reps_sum})
    df.sort_values('reps', inplace=True, ascending = False)

    #plot and save 
    fig = px.bar(df, x='exercise name', y='reps', opacity = 0.7, title= 'Average reps per set:', hover_name = 'exercise name', hover_data = {'reps': False, 'exercise name': False}, text = 'reps')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        
        })
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div



def plot_histograms_days():
    # # histogram for total of reps per exercise 
    weekDays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dic = {weekDays[i]:0 for i in range(7)}
    for i in range(len(Training.objects.all())):
        dic[f'{weekDays[Training.objects.values_list("training_date")[i][0].weekday()]}'] += 1

    # convert to df 
    df = pd.DataFrame()
    df['weekdays']  = dic.keys()
    df['frequency'] = dic.values()
    #print(df)
    # df.sort_values( inplace=True, ascending = False)

    #plot and save 
    fig = px.bar(df, x='weekdays', y='frequency', opacity = 0.7, title= 'Trainings per weekday:', hover_name = 'weekdays', hover_data = {'weekdays': False, 'frequency': False}, text = 'frequency')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        
        })
    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div



def plot_histograms_types():
    training = []
    sum_ = []

    # histogram - number training types 
    for e in range(len(Training_type.objects.all())):
        training.append(Training_type.objects.values('training_type')[e]['training_type'])
        temp = Training.objects.filter(training_type = e+1).count()
        sum_.append(0 if temp is None else temp)

    # convert to df 
    df = pd.DataFrame({'training_type': training,'frequency': sum_})
    df.sort_values('frequency', inplace=True, ascending = False)
    
    #plot and save 
    fig = px.bar(df, x='training_type', y='frequency', opacity = 0.7, title= 'Frequency per training type:', hover_name = 'training_type', hover_data = {'frequency': False, 'training_type': False}, text = 'frequency')
    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       
       })

    plot_div = plot(fig, output_type='div', include_plotlyjs=False)
    return plot_div