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
    request = request 
    if request.user.is_authenticated:
        # get exercise #'s and sum up reps by exercise name
        exc = Exercise.objects.values('exercise').filter(user_name = request.user)
        df_exc = pd.DataFrame.from_records(exc)['exercise'].value_counts().reset_index()


    else: 
        # get exercise #'s and sum up reps by exercise name
        exc = Exercise.objects.values('exercise').filter(user_name = 'test_user')
        df_exc = pd.DataFrame.from_records(exc)['exercise'].value_counts().reset_index()


    # get exercise names in textformat, merge to relate rep# to exercise name in text. 
    exc_name = pd.DataFrame.from_records(Exercise_name.objects.values('exercise_name'))
    exc_name['index'] = range(1, len(Exercise_name.objects.all())+1)
    new_df = exc_name.merge(df_exc, on='index', how='left').sort_values('exercise',ascending=False).dropna()
    

    #plot 
    fig = px.bar(new_df, x='exercise_name', y='exercise', opacity = 0.7, title= 'Total number of sets per exercise:', hover_name = 'exercise_name', hover_data = {'exercise': False, 'exercise_name': False}, text = 'exercise')
    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       })
    fig.layout.yaxis.fixedrange = True
    fig.update_traces(marker_color='grey')
    fig.update_layout(margin=dict(l=0, r=0))
    fig.update_traces(textposition='outside', hoverlabel=dict(bgcolor="black"))
    fig.update_layout(uniformtext_minsize=8)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_traces(cliponaxis=False)
    fig.update_xaxes( title=None)
    fig.update_layout(font=dict(family="Courier New, monospace"))
    config = {'displayModeBar': False}
    plot_div = plot(fig, output_type='div', config = config )
    return plot_div


def plot_histograms_reps(request):
    request = request 
    if request.user.is_authenticated:
        # get exercise #'s and sum up reps by exercise name
        exc = Exercise.objects.values('exercise', 'reps').filter(user_name = request.user)
        df_exc = pd.DataFrame.from_records(exc).groupby('exercise').sum().reset_index()

    else:
        # get exercise #'s and group reps by exercise name : sum
        exc = Exercise.objects.values('exercise', 'reps').filter(user_name = 'test_user')
        df_exc = pd.DataFrame.from_records(exc).groupby('exercise').sum().reset_index()

    # get exercise names in textformat, merge to relate rep# to exercise name in text. 
    exc_name = pd.DataFrame.from_records(Exercise_name.objects.values('exercise_name'))
    exc_name['exercise'] = range(1, len(Exercise_name.objects.all())+1)
    new_df = exc_name.merge(df_exc, on='exercise', how='left').sort_values('reps',ascending=False).dropna()


    #plot and save 
    fig = px.bar(new_df, x='exercise_name', y='reps', opacity = 0.7, title= 'Total number of reps per exercise:', hover_name = 'exercise_name', hover_data = {'reps': False, 'exercise_name': False}, text = 'reps')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        
        })
    fig.update_layout(xaxis_type='category')
    fig.layout.yaxis.fixedrange = True
    fig.update_layout(margin=dict(l=0, r=0))
    config = {'displayModeBar': False}
    fig.update_traces(textposition='outside', hoverlabel=dict(bgcolor="black"))
    fig.update_layout(uniformtext_minsize=8)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_traces(cliponaxis=False)
    fig.update_xaxes( title=None)
    fig.update_traces(marker_color='grey')
    fig.update_layout(font=dict(family="Courier New, monospace"))
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config)
    return plot_div


def plot_histograms_reppset(request):
    request = request 
    if request.user.is_authenticated:
        # get exercise #'s and get average rep/set per exercise 
        exc = Exercise.objects.values('exercise', 'reps').filter(user_name = request.user)
        df_exc = pd.DataFrame.from_records(exc)['exercise'].value_counts().reset_index()
        df_rep = pd.DataFrame.from_records(exc).groupby('exercise').sum().reset_index()
        df_rep.rename(columns={'exercise': 'index'}, inplace=True)
        df_repexc = df_exc.merge(df_rep, on='index', how='left') 
        df_repexc['average'] = round(df_repexc['reps']/df_repexc['exercise'],1)
        
    else: 
        # get exercise #'s and sum up reps by exercise name
        exc = Exercise.objects.values('exercise', 'reps').filter(user_name = 'test_user')
        df_exc = pd.DataFrame.from_records(exc)['exercise'].value_counts().reset_index()
        df_rep = pd.DataFrame.from_records(exc).groupby('exercise').sum().reset_index()
        df_rep.rename(columns={'exercise': 'index'}, inplace=True)
        df_repexc = df_exc.merge(df_rep, on='index', how='left')
        df_repexc['average'] = round(df_repexc['reps']/df_repexc['exercise'],1)
        
        
    # get exercise names in textformat, merge to relate rep# to exercise name in text. 
    exc_name = pd.DataFrame.from_records(Exercise_name.objects.values('exercise_name'))
    exc_name['index'] = range(1, len(Exercise_name.objects.all())+1)
    new_df = exc_name.merge(df_repexc, on='index', how='left').sort_values('average',ascending=False).dropna()



    #plot and save 
    fig = px.bar(new_df, x='exercise_name', y='average',opacity = 0.7, title= 'Average reps per set:', hover_name = 'exercise_name', hover_data = {'average': False, 'exercise_name': False}, text = 'average')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        
        })
    fig.layout.yaxis.fixedrange = True
    fig.update_traces(marker_color='#347c17')
    fig.update_layout(margin=dict(l=0, r=0))
    config = {'displayModeBar': False}
    fig.update_traces(textposition='outside', hoverlabel=dict(bgcolor="black"))
    fig.update_traces(cliponaxis=False)
    fig.update_layout(uniformtext_minsize=8)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_xaxes( title=None)
    fig.update_layout(font=dict(family="Courier New, monospace"))
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config)
    return plot_div


# currently not in use 
# def plot_histograms_days(request):
#     request = request 
#     # # histogram for total of reps per exercise 
#     weekDays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
#     dic = {weekDays[i]:0 for i in range(7)}
#     if request.user.is_authenticated:
#         query = Training.objects.all().filter(user_name=request.user)
#         for i in range(len(query)):
#             dic[f'{weekDays[query.values_list("training_date")[i][0].weekday()]}'] += 1

#     else: 
#         query = Training.objects.all().filter(user_name='test_user')
#         for i in range(len(query)):
#             dic[f'{weekDays[query.values_list("training_date")[i][0].weekday()]}'] += 1
            
#     # convert to df 
#     df = pd.DataFrame()
#     df['weekdays']  = dic.keys()
#     df['frequency'] = dic.values()

#     #plot and save 
#     fig = px.bar(df, x='weekdays', y='frequency', opacity = 0.7, cliponaxis=False ,title= 'Trainings per weekday:', hover_name = 'weekdays', hover_data = {'weekdays': False, 'frequency': False}, text = 'frequency')
#     fig.update_layout({
#         'plot_bgcolor': 'rgba(0, 0, 0, 0)',
#         'paper_bgcolor': 'rgba(0, 0, 0, 0)',
#         })
#     fig.update_layout(margin=dict(l=0, r=0))
#     config = {'displayModeBar': False}
#     fig.update_layout(font=dict(family="Courier New, monospace",size=14))
#     plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config )
#     return plot_div




def plot_pie_types(request):
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

    # colorscale 
    color_ind = [0,1,2,3,4,5]
    greys = n_colors('rgba(255,255,255)', 'rgb(0,0,0)', 6, colortype='rgb')
    colorscale = np.array(greys)[color_ind]


    # convert to df 
    df = pd.DataFrame({'training_type': training_2,'frequency': sum_2})
    df.sort_values('frequency', inplace=True, ascending = False)
    
    #plot and save 
    fig = px.pie(df, values='frequency', names='training_type', title= 'What kind of workouts:', 
                hover_name = 'frequency', hover_data = {'frequency': False, 'training_type': False}, 
                color_discrete_map = dict(zip(training_2, colorscale))) # geht irgendwie noch nicht. 


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
        font=dict(family="Courier New, monospace"))

    plot_div = plot(fig, output_type='div', config = config)
    return plot_div



# fig = px.pie(, , , color_discrete_map={'Thur':'lightcyan',
#                                  'Fri':'cyan',
#                                  'Sat':'royalblue',
#                                  'Sun':'darkblue'})



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
        font={ 'color':'black', 'family':"Courier New, monospace"},
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

    fig.layout.xaxis.fixedrange = False
    fig.layout.yaxis.fixedrange = True   

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

    fig.update_layout(font=dict(family="Courier New, monospace"))
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
        query = Training.objects.all().filter(user_name = request.user).exclude(training_time__isnull=True)
    else: 
        query = Training.objects.all().filter(user_name = 'test_user').exclude(training_time__isnull=True)


    timeslots =  pd.date_range("07:30", "20:30", freq="60min").time

    for i in range(len(query)): 
        ind_day = query.values()[i]['training_date'].weekday()  # index tag first element of query
        for j in range(len(timeslots)): 
            if query.values()[i]['training_time'] <= timeslots[j]: 
                frequency[j-1][ind_day] += 1 
                break

    # flatten list for colorscale    - max and min then delta in steps of 1. more beautiful
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
    blues = n_colors('rgba(255,255,255)', 'rgb(0,0,0)', np.max(frequency_)+1, colortype='rgb')
    blues[0] = 'rgba(0,0,0,0)'  # zero is transparent
    colorscale=np.array(blues)[frequency_]


    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x = weekdays,
        y = timeslot,
        z = frequency,
        colorscale=np.array(blues)[frequency_],
        text=text,
        hoverinfo='text',
        opacity = 0,
        showscale = True,

    ))
    fig.update_layout(
        #xaxis_title="Day of week",
        yaxis_title="Time of day",
        title = 'Workout days & times:',
        xaxis = {'showgrid': False },
        yaxis = {'showgrid': False },
        font=dict(
            family="Courier New, monospace",
             
        )
    )
    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       })
    fig.update_layout(hovermode='x')


    for i in range(len(frequency)): 
        for j in range(len(frequency[0])): 
            if frequency[i][j] > 0: 
                fig.add_shape(type="circle",
                    xref="x", yref="y",
                    x0=j-0.3, y0=timeslot[i]-0.5, x1=j+0.3, y1=timeslot[i]+0.5,
                    line_color=colorscale[frequency[i][j]],
                    fillcolor=colorscale[frequency[i][j]],
                )


    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_xaxes(side="top")
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    config = {'displayModeBar': False}
    fig.update_layout(margin=dict(l=0, r=0))
    plot_div = plot(fig, output_type='div', config = config)
    return plot_div
















################################################################################
#####                   Dashboard 2                                        #####
################################################################################



def exc_per_set(request): 
    tr = Training.objects.values('training_ID', 'training_date').filter(user_name = request.user)
    df_tr = pd.DataFrame.from_records(tr)
 
    ex = Exercise.objects.values('training_ID', 'exercise', 'reps').filter(user_name = request.user)
    df_ex = pd.DataFrame.from_records(ex)


    test = df_tr.merge(df_ex, on='training_ID', how='left')
    new = test.groupby(['training_date','exercise']).sum().reset_index()

    exercises = []
    for e in range(len(Exercise_name.objects.all())):
        exercises.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
    
    exc_num = list(Exercise_name.objects.all().values('id')[i]['id'] for i in range(len(exercises)))

    map_dict = dict(zip(exc_num, exercises))
    new['exercise'] = new['exercise'].map(map_dict)

    fig = px.line(new, x="training_date", y="reps", color='exercise')
    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       })
    plot_div = plot(fig, output_type='div')
    return plot_div




