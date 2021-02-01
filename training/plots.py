from numpy.lib.twodim_base import triu_indices
from .models import Exercise_name, Training, Exercise, Training_type

# graph test 
import pandas as pd


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
    fig.update_traces(textposition='outside', hoverlabel=dict(bgcolor="darkslategrey"))
    fig.update_layout(uniformtext_minsize=8)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_traces(cliponaxis=False)
    fig.update_xaxes( title=None)
    fig.update_yaxes( title='Number of Sets')
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
    fig = px.bar(new_df, x='exercise_name', y='reps', 
        opacity = 0.7, 
        title= 'Total number of reps per exercise:', 
        hover_name = 'exercise_name', 
        hover_data = {'reps': False, 'exercise_name': False}, 
        text = 'reps', )
        
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        
        })
    fig.update_layout(xaxis_type='category')
    fig.layout.yaxis.fixedrange = True
    fig.update_layout(margin=dict(l=0, r=0))
    config = {'displayModeBar': False}
    fig.update_traces(textposition='outside', hoverlabel=dict(bgcolor="darkslategrey"))
    fig.update_layout(uniformtext_minsize=8)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_traces(cliponaxis=False)
    fig.update_xaxes( title=None)
    fig.update_yaxes( title='Number of Reps')
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
    fig = px.bar(new_df, x='exercise_name', y='average',
        opacity = 0.7, 
        title= 'Average reps per set:',
        labels = {'average': 'Average Reps'},
        hover_name = 'exercise_name',
        hover_data = {'average': False, 'exercise_name': False}, 
        text = 'average')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        
        })
    fig.layout.yaxis.fixedrange = True
    fig.update_traces(marker_color=px.colors.sequential.Blugrn[-1])
    fig.update_layout(margin=dict(l=0, r=0))
    config = {'displayModeBar': False}
    fig.update_traces(textposition='outside', hoverlabel=dict(bgcolor="darkslategrey"))
    fig.update_traces(cliponaxis=False)
    fig.update_layout(uniformtext_minsize=8)
    fig.update_layout(xaxis_tickangle=45)
    fig.update_xaxes( title=None)
    fig.update_layout(font=dict(family="Courier New, monospace"))
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, config = config)
    return plot_div



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

    # convert to df 
    df = pd.DataFrame({'training_type': training_2,'frequency': sum_2})
    df.sort_values('frequency', inplace=True, ascending = False)
    
    #plot and save 
    fig = px.pie(df, values='frequency', names='training_type', title= 'What kind of workouts:', 
                hover_name = 'frequency', hover_data = {'frequency': False, 'training_type': False}, 
               color_discrete_sequence=px.colors.sequential.Blugrn[::-1]) 
                    # Mint, Darkmint, Greens

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
        hoverlabel=dict(bgcolor="darkslategrey"),
        font=dict(family="Courier New, monospace"))

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
            types.append(query.values('training_type')[i]['training_type'])
            dates.append(query.values('training_date')[i]['training_date'])
            
    else: 
        query = Training.objects.all().filter(user_name='test_user')
        for i in range(len(query)):
            types.append(query.values('training_type')[i]['training_type'])
            dates.append(query.values('training_date')[i]['training_date'])


    text = ['' for i in dates_in_year] #gives something like list of strings like ‘2018-01-25’ for each date..


    for i in range(len(dates)): 
        if dates[i] in dates_in_year: 
            x = types[i] - 1

            text[dates_in_year.index(dates[i])] +=  Training_type.objects.values('training_type')[x]['training_type'] + ', ' + dates_in_year[dates_in_year.index(dates[i])].strftime(' %d.%m.%Y') + ' '


    #4cc417 green #347c17 dark green
    colorscale=[[False, 'lightgrey'], [True, px.colors.sequential.Blugrn[-2]]]
    
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
        font={ 'color':'darkslategrey', 'family':"Courier New, monospace"},
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
    fig.update_traces( hoverlabel=dict(bgcolor="darkslategrey"))
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
    timeslot = [x for x in range(8,24)]
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # data
    frequency = [[0 for x in range(1,8)] for y in range(1,17)]

    
    if request.user.is_authenticated:
        query = Training.objects.all().filter(user_name = request.user).exclude(training_time__isnull=True)
    else: 
        query = Training.objects.all().filter(user_name = 'test_user').exclude(training_time__isnull=True)


    timeslots =  pd.date_range("07:30", "22:30", freq="60min").time
    for i in range(len(query)): 
        ind_day = query.values()[i]['training_date'].weekday()  # index tag first element of query
        for j in range(len(timeslots)): 
            if query.values()[i]['training_time'] <= timeslots[j]: 
                frequency[j][ind_day] += 1 
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
    greys = n_colors('rgba(255,255,255)', 'rgb(0,0,0)', np.max(frequency_)+1, colortype='rgb')
    greys[0] = 'rgba(0,0,0,0)'  # zero is transparent
    colorscale=np.array(greys)[frequency_]


    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x = weekdays,
        y = timeslot,
        z = frequency,
        colorscale=np.array(greys)[frequency_],
        text=text,
        hoverinfo='text',
        opacity = 0,
        showscale = True,

    ))
    fig.update_layout(
        yaxis_title="Time of Day",
        title = 'Workout Days & Times:',
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

    fig.update_traces( hoverlabel=dict(bgcolor="darkslategrey"))
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_xaxes(side="top")
    fig.layout.xaxis.fixedrange = True
    fig.layout.yaxis.fixedrange = True
    config = {'displayModeBar': False}
    fig.update_layout(margin=dict(l=0, r=0))

    plot_div = plot(fig, output_type='div', config = config)
    return plot_div



def reps_sets(request):
    if request.user.is_authenticated:
        tr = Training.objects.values('training_ID', 'training_date').filter(user_name = request.user)
        df_tr = pd.DataFrame.from_records(tr)
    
        ex = Exercise.objects.values('training_ID', 'exercise', 'reps').filter(user_name = request.user)
        df_ex = pd.DataFrame.from_records(ex)

        df_trex = df_tr.merge(df_ex, on='training_ID', how='left')

    else:
        tr = Training.objects.values('training_ID', 'training_date').filter(user_name = 'test_user')
        df_tr = pd.DataFrame.from_records(tr)
    
        ex = Exercise.objects.values('training_ID', 'exercise', 'reps').filter(user_name = 'test_user')
        df_ex = pd.DataFrame.from_records(ex)

        df_trex = df_tr.merge(df_ex, on='training_ID', how='left')

    # list of all exercises
    exercises = []
    for e in range(len(Exercise_name.objects.all())):
        exercises.append(Exercise_name.objects.values('exercise_name')[e]['exercise_name'])
    
    # list of exercises done by user 
    exc_num = list(Exercise_name.objects.all().values('id')[i]['id'] for i in range(len(exercises)))

    # replace exercise_id with actual name 
    map_dict = dict(zip(exc_num, exercises))
    df_trex['exercise'] = df_trex['exercise'].map(map_dict)
    tr_ids = df_trex['training_ID'].unique() # list with training_IDs 
    
    master = []
    for training_id in tr_ids: 
        ex_names = df_trex.where(df_trex['training_ID'] == training_id)['exercise'].dropna().unique()
        for e in ex_names:
            reps = df_trex.where(df_trex['training_ID'] == training_id).dropna()
            sets = reps.where(reps['exercise'] == e).dropna()
            rep_per = list(sets['reps'])
            rep_per.insert(0, e)
            rep_per.insert(0,Training.objects.values('training_date').filter(training_ID = training_id)[0]['training_date'])
            master.append(rep_per)
    
    #all unique exercises by the user 
    ex_names = df_trex['exercise'].dropna().unique()

    df = pd.DataFrame(master, columns = ['training_date','exercise','Set 1','Set 2','Set 3','Set 4','Set 5'])
    
    sets = ['Set 1','Set 2','Set 3','Set 4','Set 5',]
    dtest = pd.DataFrame(index = ex_names, columns =  sets)

    for set_ in sets: 
        for ex_name in ex_names:
            dtest[set_][ex_name] = round(float(df.loc[df['exercise'] == ex_name, [set_]].sum(axis=0) / df.groupby('exercise').count()[set_][ex_name]),1)
    
    dtest = dtest.rename(columns=dict(zip(sets, ['set1','set2','set3','set4','set5'])))
    dtest = dtest.reset_index() 
    dtest = dtest.rename(columns={'index': 'exercise'})
    df = df.merge(dtest, on='exercise', how='left')

    data1 = []
    for i,row in df.iterrows(): 
        data1.append(go.Scatter(x=df[['Set 1','Set 2','Set 3','Set 4','Set 5']].columns, y = df[['set1','set2','set3','set4','set5']].iloc[i],mode='markers', name = ''))  
                     # ,text = df[['set1','set2','set3','set4','set5']].iloc[i], hovertemplate = 'Average reps:'))

    # list of exercises, with duplicates
    dataex = list(df['exercise'])
    df['sum'] = df['Set 1'].fillna(0) + df['Set 2'].fillna(0) + df['Set 3'].fillna(0) + df['Set 4'].fillna(0) + df['Set 5'].fillna(0) 

    data2 = []
    # show total reps at that day
    for i,row in df.iterrows():
        x1 = [df['training_date'].iloc[i]]
        y1 = [df['sum'].iloc[i]]
        data2.append(go.Scatter(x=x1, y=y1,mode='markers' , name = ''))
       

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Average Repetitions per Set', 'Total Repetitions per Day'))

    for i in data1:
        fig.add_trace(i, row=1, col=1) 

    for i in data2: 
        fig.add_trace(i, row=2, col=1)


    button = []
    button.append(dict(
                        args=[{"visible": [True]},
                                {'yaxis.range': [0,max(df['Set 1']+5)], 
                                 'yaxis2.range': [0,max(df['sum']+5)]},   # if - show individual reps - replace 'sum' with 'set1'
                                ],
                        label='All',
                        method="update"
                    ))
    
    for ex in ex_names:     # if - show individual reps - replace 'sum' with 'set1'
        button.append(dict(
                            args=[ {"visible": [ex == dataex[i] for i in range(len(dataex))]*2}, 
                                {'yaxis.range': [0, max(df['Set 1'].where(df['exercise'] == ex).dropna() +5)], 
                                 'yaxis2.range':[0, max(df['sum'].where(df['exercise'] == ex).dropna() +5)]
                                 },
                                ],
                            label=ex,
                            method="update"
                        ))

    # Add dropdown
    updatemenus=[
            dict(
                buttons=button,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                xanchor="left",
                y=1.2,
                yanchor="top",
                bgcolor='white',
                bordercolor='darkslategrey'
            ),
        ]

    fig['layout']['updatemenus'] = updatemenus
    fig.update_layout({
       'plot_bgcolor': 'rgba(0, 0, 0, 0)',
       'paper_bgcolor': 'rgba(0, 0, 0, 0)',
       })
    fig.update_layout(height=600,
                     margin=dict(l=0, r=0), 
                     uniformtext_minsize=8,
                     showlegend=False,
                     font=dict(family="Courier New, monospace"),
                     hoverlabel=dict(bgcolor="darkslategrey"),
                     )
    
    fig.update_yaxes(title_text="Reps", row=1, col=1)
    fig.update_yaxes(title_text="Reps", row=2, col=1)
    fig.update_traces(marker_color=px.colors.sequential.Blugrn[-2], marker_size = 10)
    fig.update_traces(cliponaxis=False)

    config = {'displayModeBar': False}
    plot_div = plot(fig, output_type='div', config = config)
    return plot_div

# umso häufiger umso größer der Kreis ? 