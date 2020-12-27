from .models import Exercise_name, Training, Exercise, Training_type


# graph test 
import pandas as pd
import calmap
from django.db.models import Sum


# plotly 
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go




# plot activity plot 
import datetime
from datetime import timedelta, date
import numpy as np
from plotly.subplots import make_subplots




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
    config = {'displayModeBar': False}
    plot_div = plot(fig, output_type='div', config = config )
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
    config = {'displayModeBar': False}
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config)
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
    config = {'displayModeBar': False}
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config)
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
    config = {'displayModeBar': False}
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config )
    return plot_div



def plot_histograms_types():
    training = []
    sum_ = []

    # histogram - number training types 
    for e in range(len(Training_type.objects.all())):
        training.append(Training_type.objects.values('training_type')[e]['training_type'])
        temp = Training.objects.filter(training_type = e+1).count()
        sum_.append(0 if temp is None else temp)
    print(training)
    # convert to df 
    df = pd.DataFrame({'training_type': training,'frequency': sum_})
    df.sort_values('frequency', inplace=True, ascending = False)
    
    #plot and save 
    fig = px.bar(df, x='training_type', y='frequency', opacity = 0.7, title= 'Frequency per training type:', hover_name = 'training_type', hover_data = {'frequency': False, 'training_type': False}, text = 'frequency')
    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       
       })
    config = {'displayModeBar': False}
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config)
    return plot_div





# MIT LICENSE
def display_year(z,
                 year: int = None,
                 month_lines: bool = True,
                 fig=None,
                 row: int = None):
    
    if year is None:
        year = datetime.datetime.now().year
        
    if year == 2020 or year == 2024:
        data = np.zeros(366)
        data[:len(z)] = z
    else:  
        data = np.zeros(365)
        data[:len(z)] = z



    d1 = datetime.date(year, 1, 1)
    d2 = datetime.date(year, 12, 31)

    delta = d2 - d1

    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_days =   [31,    28,    31,     30,    31,     30,    31,    31,    30,    31,    30,    31]
    if year == 2020 or year == 2024: 
        month_days =   [31,    29,    31,     30,    31,     30,    31,    31,    30,    31,    30,    31]
    month_positions = (np.cumsum(month_days) - 15)/7

    dates_in_year = [d1 + datetime.timedelta(i) for i in range(delta.days+1)] #gives me a list with datetimes for each day a year
    weekdays_in_year = [i.weekday() for i in dates_in_year] #gives [0,1,2,3,4,5,6,0,1,2,3,4,5,6,…] (ticktext in xaxis dict translates this to weekdays
    
    weeknumber_of_dates = [int(i.strftime("%V")) if not (int(i.strftime("%V")) == 1 and i.month == 12) else 53
                           for i in dates_in_year] #gives [1,1,1,1,1,1,1,2,2,2,2,2,2,2,…] name is self-explanatory

    types = []
    dates = []
    for i in range(len(Training.objects.values('training_date', 'training_type'))):
        types.append(Training.objects.values('training_type')[i]['training_type'])
        dates.append(Training.objects.values('training_date')[i]['training_date'])
    text = ['' for i in dates_in_year] #gives something like list of strings like ‘2018-01-25’ for each date..
 
    for i in range(len(dates)): 
        if dates[i] in dates_in_year: 
            x = types[i] - 1
            # print(Training_type.objects.values('training_type')[x]['training_type'])
            
            text[dates_in_year.index(dates[i])] +=  Training_type.objects.values('training_type')[x]['training_type'] + ', ' + str(dates_in_year[dates_in_year.index(dates[i])])




    #4cc417 green #347c17 dark green
    colorscale=[[False, 'lightgrey'], [True, '#347c17']]
    
    # handle end of year
    

    data = [
        go.Heatmap(
            x=weeknumber_of_dates,
            y=weekdays_in_year,
            z=data,
            text=text,
            hoverinfo='text',
            xgap=3, # this
            ygap=3, # and this is used to make the grid-like apperance
            showscale=False,
            colorscale=colorscale
        )
    ]
    
        
    if month_lines:
        kwargs = dict(
            mode='lines',
            line=dict(
                color='#9e9e9e',
                width=1
            ),
            hoverinfo='skip'
            
        )
        for date, dow, wkn in zip(dates_in_year,
                                  weekdays_in_year,
                                  weeknumber_of_dates):
            if date.day == 1:
                data += [
                    go.Scatter(
                        x=[wkn-.5, wkn-.5],
                        y=[dow-.5, 6.5],
                        **kwargs
                    )
                ]
                if dow:
                    data += [
                    go.Scatter(
                        x=[wkn-.5, wkn+.5],
                        y=[dow-.5, dow - .5],
                        **kwargs
                    ),
                    go.Scatter(
                        x=[wkn+.5, wkn+.5],
                        y=[dow-.5, -.5],
                        **kwargs
                    )
                ]
                    
                    
    layout = go.Layout(
        height=250,
        yaxis=dict(
            showline=False, showgrid=False, zeroline=False,
            tickmode='array',
            ticktext=['Mon ',  'Wed ',  'Fri ',  'Sun '],
            tickvals=[0,  2,  4,  6],
            autorange="reversed"
        ),
        xaxis=dict(
            showline=False, showgrid=False, zeroline=False,
            tickmode='array',
            ticktext=month_names,
            tickvals=month_positions
        ),
        font={'size':10, 'color':'black'},
        plot_bgcolor=('#fff'),
        margin = dict(t=40),
        showlegend=False
    )
    
    if fig is None:
        fig = go.Figure(data=data, layout=layout)
    else:
        fig.add_traces(data, rows=[(row+1)]*len(data), cols=[1]*len(data))
        fig.update_layout(layout)
        fig.update_xaxes(layout['xaxis'])
        fig.update_yaxes(layout['yaxis'])

    fig['layout']['yaxis']['scaleanchor']='x'

    return fig


def display_years(z, years):
    fig = make_subplots(rows=len(years), cols=1, subplot_titles=years)
    for i, year in enumerate(years):
        data = z[i*365 : (i+1)*365]
        display_year(data, year=year, fig=fig, row=i)
    
    
    
    config = {'displayModeBar': False}



    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       })
    
    plot_div = plot(fig, output_type='div', config=config)
    return plot_div




# Function that returns a 0-list with ones, when the day was a training day. 
def get_training_days(): 
    dates = []
    for i in range(len(Training.objects.values('training_date'))):
        dates.append(Training.objects.values('training_date')[i]['training_date'])

    sdate = date(datetime.datetime.now().year, 1, 1)   # start date
    edate = date(datetime.datetime.now().year, 12, 31)   # end date

    delta = edate - sdate

    this_year = []
    for i in range(delta.days + 1):
        this_year.append(sdate + timedelta(days=i))


    for i in range(len(this_year)): 
        if this_year[i] in dates: 
            this_year[i] = 1 
        else: 
            this_year[i] = 0 

    return this_year