from .models import Exercise_name, Training, Exercise, Training_type

# graph test 
import pandas as pd
from django.db.models import Sum


# plotly 
from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go
from plotly.colors import n_colors




# plot activity plot 
import datetime
from datetime import timedelta, date
import numpy as np
from plotly.subplots import make_subplots








def plot_histograms_exercise(request):
    exercise = []
    reps_sum = []
    request = request
    if request.user.is_authenticated:
        query = Exercise.objects.all().filter(user_name = request.user)
        # histogram total sets of individual exercises
        for e in range(len(Exercise_name.objects.all())):
            exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
            rep = query.filter(exercise_id = e+1).count()
            reps_sum.append(0 if rep is None else rep)
    else: 
        query = Exercise.objects.all().filter(user_name = 'test_user')
        # histogram total sets of individual exercises
        for e in range(len(Exercise_name.objects.all())):
            exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
            rep = query.filter(exercise_id = e+1).count()
            reps_sum.append(0 if rep is None else rep)


    # convert to df 
    df = pd.DataFrame({'exercise name': exercise,'sets': reps_sum})
    df.sort_values('sets', inplace=True, ascending = False)
    
    #plot and save 
    fig = px.bar(df, x='exercise name', y='sets', opacity = 0.7, title= 'Total number of sets per exercise:', hover_name = 'exercise name', hover_data = {'sets': False, 'exercise name': False}, text = 'sets')
    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       
       })
    fig.update_layout(margin=dict(l=0, r=0))
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_traces(cliponaxis=False)
    fig.update_layout(font=dict(family="Courier New, monospace",size=14))
    config = {'displayModeBar': False}
    plot_div = plot(fig, output_type='div', config = config )
    return plot_div


def plot_histograms_reps(request):
    exercise = []
    reps_sum = []
    request = request 
    if request.user.is_authenticated:
        # histogram for total of reps per exercise 
        query = Exercise.objects.filter(user_name = request.user)
        for e in range(len(Exercise_name.objects.all())):
            exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
            rep = query.filter(exercise = e+1).aggregate(Sum('reps'))['reps__sum']
            reps_sum.append(0 if rep is None else rep)   

    else:
        query = Exercise.objects.filter(user_name = 'test_user')
        for e in range(len(Exercise_name.objects.all())):
            exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
            rep = query.filter(exercise = e+1).aggregate(Sum('reps'))['reps__sum']
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
    fig.update_layout(margin=dict(l=0, r=0))
    config = {'displayModeBar': False}
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=8)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_traces(cliponaxis=False)
    fig.update_layout(font=dict(family="Courier New, monospace",size=14))
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config)
    return plot_div


def plot_histograms_reppset(request):
    exercise = []
    reps_sum = []
    request = request
    if request.user.is_authenticated:
    # # histogram for total of reps per exercise 
        query = Exercise.objects.filter(user_name = request.user)
        for e in range(len(Exercise_name.objects.all())):
            exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
            total_rep = query.filter(exercise = e+1).aggregate(Sum('reps'))['reps__sum']
            sets = query.filter(exercise_id = e+1).count()
            if sets == None or total_rep == None:
                reppset = 0
            else: 
                reppset = round(total_rep/sets, 2)
            reps_sum.append(0 if reppset is None else reppset)

    else: 
        query = Exercise.objects.filter(user_name = 'test_user')
        for e in range(len(Exercise_name.objects.all())):
            exercise.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
            total_rep = query.filter(exercise = e+1).aggregate(Sum('reps'))['reps__sum']
            sets = query.filter(exercise_id = e+1).count()
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
    fig.update_layout(margin=dict(l=0, r=0))
    config = {'displayModeBar': False}
    fig.update_traces(textposition='outside')
    fig.update_traces(cliponaxis=False)
    fig.update_layout(uniformtext_minsize=8)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_layout(font=dict(family="Courier New, monospace",size=14))
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config)
    return plot_div


# currently not in use 
def plot_histograms_days(request):
    request = request 
    # # histogram for total of reps per exercise 
    weekDays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dic = {weekDays[i]:0 for i in range(7)}
    if request.user.is_authenticated:
        query = Training.objects.all().filter(user_name=request.user)
        for i in range(len(query)):
            dic[f'{weekDays[query.values_list("training_date")[i][0].weekday()]}'] += 1

    else: 
        query = Training.objects.all().filter(user_name='test_user')
        for i in range(len(query)):
            dic[f'{weekDays[query.values_list("training_date")[i][0].weekday()]}'] += 1
            
    # convert to df 
    df = pd.DataFrame()
    df['weekdays']  = dic.keys()
    df['frequency'] = dic.values()

    #plot and save 
    fig = px.bar(df, x='weekdays', y='frequency', opacity = 0.7, cliponaxis=False ,title= 'Trainings per weekday:', hover_name = 'weekdays', hover_data = {'weekdays': False, 'frequency': False}, text = 'frequency')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
    fig.update_layout(margin=dict(l=0, r=0))
    config = {'displayModeBar': False}
    fig.update_layout(font=dict(family="Courier New, monospace",size=14))
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config )
    return plot_div




def plot_bar_types(request):
    training = []
    sum_ = []
    request = request

    # histogram - number training types 
    if request.user.is_authenticated:
        query = Training.objects.all().filter(user_name=request.user)
        for e in range(len(Training_type.objects.all())):
            training.append(Training_type.objects.values('training_type')[e]['training_type'])
            temp = query.filter(training_type = e+1).count()
            sum_.append(0 if temp is None else temp)

    else: 
        query = Training.objects.all().filter(user_name='test_user')
        for e in range(len(Training_type.objects.all())):
            training.append(Training_type.objects.values('training_type')[e]['training_type'])
            temp = query.filter(training_type = e+1).count()
            sum_.append(0 if temp is None else temp)

    # can be implemented in a better way. 
    sum_2 = []
    training_2 = []
    for i in range(len(sum_)): 
        if sum_[i] != 0:
            sum_2.append(sum_[i])
            training_2.append(training[i])
            


    # convert to df 
    df = pd.DataFrame({'training_type': training_2,'frequency': sum_2})
    df.sort_values('frequency', inplace=True, ascending = False)
    
    #plot and save 
    fig = px.pie(df, values='frequency', names='training_type', title= 'Frequency per training type:', 
                hover_name = 'frequency', hover_data = {'frequency': False, 'training_type': False})
    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       })
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0))
    fig.update_traces( textinfo='label+percent', textposition='outside', 
                     textfont_size=14, marker=dict(line=dict(color='rgba(0, 0, 0, 0)', width =2)))
    config = {'displayModeBar': False}
    
    fig.update_layout(
        hoverlabel=dict(bgcolor="black"),
        font=dict(family="Courier New, monospace", size=14 ))

    plot_div = plot(fig, output_type='div', config = config)
    return plot_div


# MIT LICENSE
def display_year(z,request,
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


    # first x days of year y belong to last week of year y-1. set to week 0 to have them at the start of the new year. 
    for j in range(7): 
        if weeknumber_of_dates[j] != 1: 
            weeknumber_of_dates[j] = 0 
        else: 
            break 



    request = request 
    types = []
    dates = []

    if request.user.is_authenticated:
        query = Training.objects.all().filter(user_name=request.user)
        for i in range(len(query)):
            types.append(Training.objects.values('training_type')[i]['training_type'])
            dates.append(query.values('training_date')[i]['training_date'])

    else: 
        query = Training.objects.all().filter(user_name='test_user')
        for i in range(len(query)):
            types.append(Training.objects.values('training_type')[i]['training_type'])
            dates.append(query.values('training_date')[i]['training_date'])


    text = ['' for i in dates_in_year] #gives something like list of strings like ‘2018-01-25’ for each date..


    for i in range(len(dates)): 
        if dates[i] in dates_in_year: 
            x = types[i] - 1

            text[dates_in_year.index(dates[i])] +=  Training_type.objects.values('training_type')[x]['training_type'] + ', ' + dates_in_year[dates_in_year.index(dates[i])].strftime(' %d.%m.%Y')


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
        height= 400,
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

        showlegend=False
    )
    
    if fig is None:
        fig = go.Figure(data=data, layout=layout)
    else:
        fig.add_traces(data, rows=[(row+1)]*len(data), cols=[1]*len(data))
        fig.update_layout(layout)
        fig.update_xaxes(layout['xaxis'])
        fig.update_yaxes(layout['yaxis'])
    fig.update_layout(margin=dict(l=0, r=0))
    fig['layout']['yaxis']['scaleanchor']='x'
     
    return fig


def display_years(z, years, request):
    request = request 
    fig = make_subplots(rows=len(years), cols=1, subplot_titles=years)
    for i, year in enumerate(years):
        data = z[i*365 : (i+1)*365]
        display_year(data, request, year=year, fig=fig, row=i)
        
    config = {'displayModeBar': False}

    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
    fig.update_layout(font=dict(family="Courier New, monospace",size=14))
    # fig.update_layout(margin=dict(l=0, r=0,  t= 200))
    plot_div = plot(fig, output_type='div', config=config)
    return plot_div




# Function that returns a 0-list with ones, when the day was a training day. 
def get_training_days(request):
    request = request  
    dates = []
    if request.user.is_authenticated:
        query = Training.objects.values('training_date').filter(user_name= request.user)
        
        for i in range(len(query)):
            dates.append(query.values('training_date')[i]['training_date'])

    else: 
        query = Training.objects.values('training_date').filter(user_name= 'test_user')
        for i in range(len(query)):
            dates.append(query.values('training_date')[i]['training_date'])

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






def plot_heatmap_week(request):
    # label 
    timeslot = [x for x in range(8,20)]
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # data
    frequency = [[0 for x in range(1,8)] for y in range(1,13)]
    
    
    if request.user.is_authenticated:
        query = Training.objects.all().filter(user_name = request.user)
    else: 
        query = Training.objects.all().filter(user_name = 'test_user')


    timeslots =  pd.date_range("07:30", "20:30", freq="60min").time

    for i in range(len(query)): 
        ind_day = query.values()[i]['training_date'].weekday()  # index tag erstes element 
        for j in range(len(timeslots)): 
            if query.values()[i]['training_time'] <= timeslots[j]: 
                frequency[j-1][ind_day] += 1 
                break

    # flatten list for colorscale 
    output = []
    def reemovNestings(l): 
        for i in l: 
            if type(i) == list: 
                reemovNestings(i) 
            else: 
                output.append(i)
        return set(output) 
    frequency_ = list(reemovNestings(frequency))

    
    text = [[0 for x in range(1,8)] for y in range(1,13)]
    for i in range(len(text)): 
        for j in range(len(text[i])):
            if frequency[i][j] == 0: 
                text[i][j] = ''
            else: 
                text[i][j] = f'{timeslot[i]}:00 -  {frequency[i][j]}'
    
    # color scale 
    blues = n_colors('rgb(200, 200, 255)', 'rgb(0, 0, 200)', np.max(frequency_)+1, colortype='rgb')
    blues[0] = 'rgba(0,0,0,0)'  # zero is transparent

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x = weekdays,
        y = timeslot,
        z = frequency,
        colorscale=np.array(blues)[frequency_],
        text=text,
        hoverinfo='text',

    ))
    fig.update_layout(
        #xaxis_title="Day of week",
        yaxis_title="Time of day",
        title = 'Workout times:',
        xaxis = {'showgrid': False },
        yaxis = {'showgrid': False },
        font=dict(
            family="Courier New, monospace",
            size=14, 
        )
    )
    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       })
    fig.update_layout(hovermode='x')

    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_xaxes(side="top")
    config = {'displayModeBar': False}
    fig.update_layout(margin=dict(l=0, r=0))
    plot_div = plot(fig, output_type='div', config = config)
    return plot_div