import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import pandas as pd

from dash.dependencies import Input, Output
import plotly.graph_objs as go


# This dashboard is built for the Consumption data model
# Reported occasion of food consumption: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310047501


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
server = app.server

# Read data
data = pd.read_csv('Consumption.csv', low_memory = False)


Age_indicators = data['Age group'].unique()
sex_indicators = data['Sex'].unique()


# initialize dropdown tab2 bar plot
dropdown1 = data.groupby(['Sex','Age group','Reported occasion of food consumption' ]).sum().unstack()['VALUE'].index.unique().levels[0].tolist()
dropdown2 = data.groupby(['Sex','Age group','Reported occasion of food consumption' ]).sum().unstack()['VALUE'].index.unique().levels[1].tolist()
dropdown_choice1 =[] #sex
dropdown_choice2 = [] #age groups

for i in dropdown1:
    dropdown_choice1.append({'label':i, 'value':i})

for i in dropdown2:
    dropdown_choice2.append({'label':i, 'value':i})

colors = ['Bluered','Reds', 'Blues', 'Picnic', 'Rainbow', 'Hot', 'Blackbody', 'Earth', 'Viridis']

import random
random_color = random.choice(colors)

tabs_styles = {
     'vertical-align': 'middle',
     'float': 'left',
     'backgroundColor': 'rgb(249,249,249)'
}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '10px',
    'fontWeight': 'bold',
    'backgroundColor': 'rgb(249,249,249)'
}


tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',

}


mapbox_access_token = 'pk.eyJ1IjoiYmlwYXJ0aXRlaHlwZXJjdWJlIiwiYSI6ImNqczFjZzUydjF0MGc0OW1sYWIwYW5lY2UifQ.l6OVeSa3poXp6S4s8km8kA'

# Tabs}
app.layout = html.Div(
    html.Div([
        html.Div(
            [
                html.H1(children='Reported occasion of food consumption',
                        className='nine columns'),
                html.H6(children='Number and percentage of persons based on reported occasions of food consumption, by age group and sex, for 2004 only.',
                        className='nine columns'),
                html.Label(['Please click here to be directed to Statistics Canada', html.A(' Data link ', href='https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310047501')])
            ], className="row"
        ),

        # Dropdown menu and slider
        html.Div(
            [dcc.Store(id='memory-output', storage_type='session'),
             # Category Dropdown
             html.Div(
                 [
                     html.P('Age group',
                            style={'fontWeight': 600}
                            ),

                     dcc.Dropdown(id='measure-column',
                                  options=[{'label': i, 'value': i} for i in Age_indicators],
                                  value='1 to 18 years',
                                    style={'position':'relative', 'zIndex':'999'}
                                  ),
                 ],
                 className='six columns',
                 style={'margin-top': '0'}
             ),

             # Category Dropdown
             html.Div(
                 [
                     html.P('Sex',
                            style={'fontWeight': 600}
                            ),

                     dcc.Dropdown(id='age-column',
                                  options=[{'label': i, 'value': i} for i in sex_indicators],
                                  value='Females',
                                  style={'position':'relative', 'zIndex':'999'}
                                  ),
                 ],
                 className='six columns',
                 style={'margin-top': '0'}
             ),
             ], className="row"
        ),

        html.Div([
            # DataTable
            html.Div(
                [
                    dt.DataTable(
                        id='datatable',
                        data=data.to_dict('rows'),
                        columns=[
                            {"name": i, "id": i, "deletable": True} for i in data[
                                ['REF_DATE', 'GEO', 'DGUID', 'Age group','Sex', 'Reported occasion of food consumption', 'Characteristics', 'VALUE',
                                 'SCALAR_FACTOR', 'VALUE']].columns
                        ],
                        style_header={'backgroundColor':
                                          'rgb(192, 192, 192)',
                                      'color':'#333'
                                      },
                        style_cell={'textAlign': 'left',
                                    'minWidth': '0px', 'maxWidth': '180px',
                                    'whiteSpace': 'normal',
                                    'backgroundColor': 'rgb(249, 249, 249)',
                                    'color':'#333'

                                    },
                        css=[{
                            'selector': '.dash-cell div.dash-cell-value',
                            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;',

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
      dcc.Tabs(id='tabs',vertical = True, children=[
            dcc.Tab(label='Time Series', style=tab_style, selected_style=tab_selected_style,
                    children = [html.Div([html.H2(children = 'Error: No Time Series Data', style = {'textAlign': 'center'}),
                                          dcc.Graph(id='Time-series',
                                                    figure={'data': [],'layout': go.Layout(title='Reported occasion of food consumption',
                                                                                           titlefont=dict(family='Arial',size=18,color='black'),width=1300,height=500)})])]),

            dcc.Tab(label='Barplot', style=tab_style, selected_style=tab_selected_style,
                    children = [html.Div([dcc.Dropdown(id='dropdown1', options=dropdown_choice1,value='Both sexes',style={'height': '40px','width': '1300px','font-size': "100%",'min-height': '3px'}),
                       html.Div(id='graph-with-dropdown')])]),

            dcc.Tab(label='Pie chart', style=tab_style, selected_style=tab_selected_style,
                    children = [html.Div([dcc.Dropdown(id='dropdown2', options=dropdown_choice2, value='19 to 30 years', style={'height': '40px','width': '1300px','font-size': "100%",'min-height': '3px'}),
                                          html.Div(id='graph-with-dropdown2')])]),

            dcc.Tab(label = 'Geolocation', style=tab_style, selected_style=tab_selected_style,
                    children = [html.Div([html.H2(children = 'Error: No Geolocation Data', style = {'textAlign': 'center'}),
                                          dcc.Graph(id='geolocation-plot',config={'scrollZoom': True},
                                                    figure = {'data': [go.Scattermapbox()],
                                                              'layout': go.Layout(autosize=False, width=1300, height=600, hovermode='closest',
                                                                                  mapbox=dict(accesstoken=mapbox_access_token, bearing=0,center=dict(lat=53, lon=-100),
                                                                                              pitch=35, zoom=3))})])])], parent_style={'float': 'left'}),
       html.Div(id='tab-output', style={'float': 'left', 'width': '400'})
   ])],style={'backgroundColor': 'rgb(249,249,249)'}), style={'backgroundColor': 'rgb(249,249,249)'})

# call backs for data table
@app.callback(dash.dependencies.Output('datatable', 'data'),
             [dash.dependencies.Input('measure-column', 'value'),
              dash.dependencies.Input('age-column', 'value')])

def update_rows(selected_measure,selected_age):
    filtered = data.loc[(data['Age group'] == selected_measure) & (data['Sex'] == selected_age)]
    return filtered.to_dict('rows')


@app.callback(dash.dependencies.Output('graph-with-dropdown', 'children'),[dash.dependencies.Input('dropdown1', 'value')])

def update_figure(value):
    # update based on two conditions
    data_consump = data[(data['Sex'] == value)].groupby(['Sex', 'Age group', 'Reported occasion of food consumption']).sum().unstack()['VALUE']
    y_values = data_consump.iloc[0].tolist()
    x_values = data_consump.columns.tolist()

    return html.Div([dcc.Graph(id='Graph-with-dropdown',
              figure={'data': [go.Bar(x=x_values, y=y_values, marker={
                  'color': y_values,
                  'colorscale':random_color,
                  'showscale': True,
                  'reversescale': True,
                  'line': dict(color='rgb(8,48,107)', width=2)})],
                  'layout':go.Layout(title = ('Reported occasion of food consumption by ' +str(value)),width = 1300, height = 500,
                                     titlefont=dict(family='Arial',size=18,color='black'),
                                         yaxis=dict(title='Tonnes',titlefont=dict(family='Arial',size=18,color='black')),
                                         xaxis=dict(title='Category',titlefont=dict(family='Arial',size=18,color='black')))})])

@app.callback(dash.dependencies.Output('graph-with-dropdown2', 'children'),[dash.dependencies.Input('dropdown2', 'value')])

def update_figure(value):
    data_consump = data[data['Age group'] == value].groupby(['Sex', 'Age group', 'Reported occasion of food consumption']).sum().unstack()['VALUE']
    y_values = data_consump.iloc[0].tolist()
    x_values = data_consump.columns.tolist()
    return html.Div([dcc.Graph(id='Graph-with-dropdown2',
                           figure=go.Figure(data=[go.Pie(labels=x_values,
                                                         values=y_values,
                                                         hoverinfo='label+value',
                                                         textinfo='percent',
                                                         textfont=dict(size=10),
                                                         marker=dict(colors=colors,
                                                                     line=dict(color='#000000', width=2)))],
                                            layout=go.Layout(title='Reported occasion of food consumption by age: '+str(value),height=650,width=1400)))])


if __name__ == '__main__':
    app.run_server(debug=True, port =9000)
