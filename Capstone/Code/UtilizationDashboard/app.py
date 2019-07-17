import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Nutrients in the food supply, by source of nutritional equivalent and commodity
# https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3210033301&pickMembers%5B0%5D=3.1&pickMembers%5B1%5D=4.35


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
server = app.server

df= pd.read_csv('Nutrient.csv', low_memory = False)

#SELECTING unique cities,categories, and stars for dropdown menu and slider
Measures_indicators = df['Measures'].unique()
Sex_indicators = df['Sex'].unique()
Age_indicators =df['Age group'].unique()



# Groupby commodity and date for time series graph
groupbynutrients = df.groupby(['Measures','REF_DATE']).sum().unstack()['VALUE']

# Create empty trace to store data
traces= []

# traces
for i in range(0,len(groupbynutrients)):
    traces.append(go.Scatter(x = groupbynutrients.columns.tolist(), y =groupbynutrients.iloc[i], mode = 'lines',name = str(groupbynutrients.index.tolist()[i])))

# tab2 barplot with drop down
dropdown_choices = df.groupby(['Measures']).sum().unstack()['VALUE'].index.tolist()
dropdown_choice =[]

for i in dropdown_choices:
    dropdown_choice.append({'label':i, 'value':i})

colors = ['Bluered','Reds', 'Blues', 'Picnic', 'Rainbow', 'Hot', 'Blackbody', 'Earth', 'Viridis']

import random

random_color = random.choice(colors)


mapbox_access_token = 'pk.eyJ1IjoiYmlwYXJ0aXRlaHlwZXJjdWJlIiwiYSI6ImNqczFjZzUydjF0MGc0OW1sYWIwYW5lY2UifQ.l6OVeSa3poXp6S4s8km8kA'


tabs_styles = {
     'vertical-align': 'middle',
     'float': 'left'
}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '10px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

# Cummulative Piechart of total supply of milk Products from 1976 to 2019
# Piechart with year slider or some sort of cummulative slider by year since we can sum up all the records that are in 1976, 1977, etc.


app.layout = app.layout = html.Div(
    html.Div([
        html.Div(
            [
                html.H1(children='Nutrients in the food supply, by source of nutritional equivalent and commodity',className='nine columns'),
                html.H6(children='This table contains 12354 series, with data for years 1976 - 2009 (not all combinations necessarily have data for all years), and is no longer being released. This table contains data described by the following dimensions (Not all combinations are available): Geography (1 item: Canada); Nutrients (29 items: Calcium;Carbohydrates;Cholesterol;Copper; ...); Source of nutritional equivalent (2 items: Nutrients available;Nutrients available adjusted for losses); Commodity (213 items: All commodities;Cereal products, total;Breakfast food;Corn flour and meal; ...)',
                        className = 'nine columns'),
                html.Label(['Please click here to be directed to Statistics Canada', html.A(' Data link ', href='https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3210033301&pickMembers%5B0%5D=3.1&pickMembers%5B1%5D=4.35')])
            ], className="row"
        ),

        #Dropdown menu and slider
        html.Div(
            [dcc.Store(id='memory-output',storage_type='session'),
                #Category Dropdown
                html.Div(
                    [
                        html.P('Choose Province:',
                               style = {'fontWeight':600}
                        ),

                        dcc.Dropdown(id='measure-column',
                                     options=[{'label': i, 'value': i} for i in Measures_indicators],
                                     value='Vitamin D (nmol/L)',
                                     style={'position':'relative', 'zIndex':'999'}
                        ),
                    ],
                    className='six columns',
                    style={'margin-top': '0'}
                ),

             # Category Dropdown
             html.Div(
                 [
                     html.P('Choose Commodity: ',
                            style={'fontWeight': 600}
                            ),

                     dcc.Dropdown(id='age-column',
                                  options=[{'label': i, 'value': i} for i in Age_indicators],
                                  value='Ages 6 to 19',
                                  style={'position':'relative', 'zIndex':'999'}
                                  ),
                 ],
                 className='six columns',
                 style={'margin-top': '0'}
             ),
            ], className="row"
        ),


        html.Div ([
               #DataTable
               html.Div(
                    [
                    dt.DataTable(
                        id='datatable',
                        data=df.to_dict('rows'),
                        columns=[
                                {"name": i, "id": i, "deletable": True} for i in df[['REF_DATE','GEO','DGUID','Measures', 'Sex', 'Age group', 'Statistics', 'SCALAR_FACTOR', 'VALUE']].columns
                        ],
                        style_header={'backgroundColor':
                                      'rgb(192, 192, 192)',
                                      'color':'#333'},
                        style_cell={'textAlign': 'left',
                                'minWidth': '0px','maxWidth': '180px',
                                'whiteSpace': 'normal',
                                'backgroundColor': 'rgb(249, 249, 249)',
                                'color':'#333'
                            },
                        css=[{
                                'selector': '.dash-cell div.dash-cell-value',
                                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                            }],


                       style_table={
                            'maxHeight': '500',
                            'overflowY': 'scroll',
                            'overflowX': 'scroll'
                        },

                        pagination_mode="fe",
                                pagination_settings={
                                    "displayed_pages": 1,
                                    "current_page": 0,
                                    "page_size": 15,
                                },

                        navigation="page",
                        n_fixed_rows=1,
                        sorting=True,
                        sorting_type="multi",
                        selected_rows=[]
                    ),
                ], className="twelve columns", style={'margin-top': '10'}
                )
    ], className="row"
    ),
    html.Div(
    children=[
        dcc.Tabs(id='tabs', vertical=True, children=[
            dcc.Tab(label='Time Series', style=tab_style, selected_style=tab_selected_style,
                    children=[html.Div([dcc.Graph(id='nutrient utilization',
                                                  figure={'data': traces,
                                                          'layout': go.Layout(title='Nutrient Utilization',
                                                                              titlefont=dict(family='Arial', size=18,
                                                                                             color='black'), width=1300,
                                                                              height=500)})])]),

            dcc.Tab(label='Barplot', style=tab_style, selected_style=tab_selected_style,
                    children=[html.Div([dcc.Dropdown(id='my-dropdown', options=dropdown_choice,
                                                     value='Ferritin  (µg/L)',
                                                     style={'height': '40px', 'width': '1300px', 'font-size': "100%",
                                                            'min-height': '3px', }),
                                        html.Div(id='graph-with-dropdown')])]),

            dcc.Tab(label='Pie chart', style=tab_style, selected_style=tab_selected_style,
                    children = [html.Div([dcc.Slider(id='year-slider', min=2013, max=2017, step=1, value=2015),
                                           html.Div(id='graph-with-slider')])]),

            dcc.Tab(label='Geolocation', style=tab_style, selected_style=tab_selected_style,
                    children=[html.Div([dcc.Graph(id='geolocation-plot', config={'scrollZoom': True},
                                                  figure={'data': [],
                                                          'layout': go.Layout(autosize=False, width=1300, height=600,
                                                                              hovermode='closest',
                                                                              mapbox=dict(
                                                                                  accesstoken=mapbox_access_token,
                                                                                  bearing=0,
                                                                                  center=dict(lat=53, lon=-100),
                                                                                  pitch=35, zoom=3),
                                                                              title='No Geolocation Data')})])])],
                 parent_style={'float': 'left'}),
        html.Div(id='tab-output', style={'float': 'left', 'width': '400'})])
    ]))

@app.callback(dash.dependencies.Output('datatable', 'data'),
             [dash.dependencies.Input('measure-column', 'value'),
              dash.dependencies.Input('age-column', 'value')
              ])

def update_rows(selected_measure,selected_age):
    filtered = df.loc[( df['Measures'] == selected_measure) & (df['Age group'] == selected_age)]
    return filtered.to_dict('rows')


@app.callback(dash.dependencies.Output('graph-with-dropdown', 'children'),[dash.dependencies.Input('my-dropdown', 'value')])

def update_figure(value):
    data_milk = df[df['Measures'] == value].groupby(['Measures', 'Age group']).sum().unstack()['VALUE']
    y_values = data_milk.iloc[0].tolist()
    x_values = data_milk.columns.tolist()

    # return the dcc.Graph so it will graph out
    return html.Div([dcc.Graph(id='Graph-with-dropdown',
              figure={'data': [go.Bar(x=x_values, y=y_values, name=str(value), marker={
                  'color': y_values,
                  'colorscale':random_color,
                  'showscale': True,
                  'reversescale': True,
                  'line': dict(color='rgb(8,48,107)', width=2)})],
                  'layout':go.Layout(title = (str(value) + ' by Age groups'),width = 1300, height = 500,
                                     titlefont=dict(family='Arial',size=18,color='black'),
                                         yaxis=dict(title='Tonnes',titlefont=dict(family='Arial',size=18,color='black')),
                                         xaxis=dict(title='Category',titlefont=dict(family='Arial',size=18,color='black')))})])


@app.callback(dash.dependencies.Output('graph-with-slider', 'children'),[dash.dependencies.Input('year-slider', 'value')])

def update_figure1(value):
    data_pie = df.groupby(['REF_DATE', 'Measures']).sum()['VALUE'][value]
    pie_labels = data_pie.index.tolist()
    pie_values = data_pie.values.tolist()

    return html.Div([dcc.Graph(id='Graph-with-slider',
                           figure=go.Figure(
                               data=[go.Pie(labels=pie_labels,
                                            values=pie_values,
                                            hoverinfo='label+value',
                                            textinfo='percent',
                                            textfont=dict(size=10),
                                            marker=dict(colors=colors,
                                                        line=dict(color='#000000', width=2)))],
                               layout=go.Layout(
                                   title='Nutritional status of '+str(value)+'in the household population',
                                   height=600,
                                   width=1300
                               )))])


if __name__ == '__main__':
    app.run_server(debug=True, port =8000)
