import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# This dashboard is built for the supply data model for Stats Can website prototype
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets = external_stylesheets)
server = app.server

# https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3210011301
# Read CSV file for Dairy distribution
data = pd.read_csv('./milk3.csv', low_memory = False)


# initialize indicators
# Dairy distribution and GEO (PROVINCE
distribution_indicators = data['Dairy distribution'].unique()
GEO_indicators = data['GEO'].unique()

df = data[data['GEO'] == 'Canada'].copy()
# Groupby commodity and date for time series graph
groupbymilk = df.groupby(['Dairy distribution','REF_DATE']).sum().unstack()['VALUE']

# Create empty trace to store data
traces= []

# traces
for i in range(0,len(groupbymilk)):
    traces.append(go.Scatter(x = groupbymilk.columns.tolist(), y =groupbymilk.iloc[i], mode = 'lines',name = str(groupbymilk.index.tolist()[i])))


# tab2 barplot with drop down
dropdown_choices = data.groupby(['Dairy distribution','GEO']).sum().unstack()['VALUE'].index.tolist()
dropdown_choice =[]


for i in dropdown_choices:
    dropdown_choice.append({'label':i, 'value':i})

colors = ['Bluered','Reds', 'Blues', 'Picnic', 'Rainbow', 'Hot', 'Blackbody', 'Earth', 'Viridis']

import random

random_color = random.choice(colors)

df['YEAR'] = df['REF_DATE'].str.slice(0, 4)

mapbox_access_token = 'pk.eyJ1IjoiYmlwYXJ0aXRlaHlwZXJjdWJlIiwiYSI6ImNqczFjZzUydjF0MGc0OW1sYWIwYW5lY2UifQ.l6OVeSa3poXp6S4s8km8kA'

info = data.groupby(['GEO', 'longitude', 'latitude']).sum().index.tolist()
info.remove(('Canada', 1.0, 1.0))
lat = []
lon = []
province = []

for i in info:
    province.append(i[0])
    lon.append(i[1])
    lat.append(i[2])

unnormalized = data.groupby(['GEO', 'longitude', 'latitude']).sum()['VALUE'].tolist()
unnormalized.remove(961150236.0) # remove the total Canada
normalized = [(float(i)/sum(unnormalized))*500 for i in unnormalized]
str_unnorm = [str(int(i)) for i in unnormalized]
text_display = []

for i in zip(str_unnorm,province):
    text_display.append(i[1]+": "+i[0]+" Kilolitres")

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


app.layout = app.layout = html.Div(
    html.Div([
        html.Div(
            [
                html.H1(children='Milk production and utilization',
                        className='nine columns'),
                html.H6(
                    children='Milk production and utilization, Canada and provinces (in kilolitres). Data are available on a monthly basis.',
                    className='nine columns'),
                html.Label(['Please click here to be directed to Statistics Canada', html.A(' Data link',
                                                                                            href='https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3210011301')])
            ], className="row"
        ),

        # Dropdown menu and slider
        html.Div(
            [dcc.Store(id='memory-output', storage_type='session'),
             # Category Dropdown
             html.Div(
                 [
                     html.P('Province',
                            style={'fontWeight': 600}
                            ),

                     dcc.Dropdown(id='measure-column',
                                  options=[{'label': i, 'value': i} for i in GEO_indicators],
                                  value='Quebec',
                                  style={'position': 'relative', 'zIndex': '999'}
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
                                  options=[{'label': i, 'value': i} for i in distribution_indicators],
                                  value='Milk sold off farms, total',
                                  style={'position': 'relative', 'zIndex': '999'}
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
                                ['REF_DATE', 'GEO', 'DGUID', 'Dairy distribution', 'UOM',
                                 'VALUE']].columns
                        ],
                        style_header={'backgroundColor': 'rgb(192, 192, 192)',
                                      'color':'#333'},
                        style_cell={'textAlign': 'left',
                                    'minWidth': '0px', 'maxWidth': '180px',
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
      dcc.Tabs(id='tabs', vertical = True, children=[
            dcc.Tab(label='Time Series', style=tab_style, selected_style=tab_selected_style, children = [html.Div([
            dcc.Graph(id='Dairy-distribution',
                  figure={'data': traces,
                          'layout': go.Layout(title='Dairy Distribution in Canada (Kilolitres)',
                                              titlefont=dict(family='Arial',size=18,color='black'),
                                              width=1300,
                                              height=500)})])]),
            dcc.Tab(label='Barplot', style=tab_style, selected_style=tab_selected_style, children =[html.Div([dcc.Dropdown(id='my-dropdown', options=dropdown_choice, value='Delivered as cream',style={
    'height': '40px',
    'width': '1300px',
    'font-size': "100%",
    'min-height': '3px',
    }),
                  html.Div(id='graph-with-dropdown')])]),
            dcc.Tab(label='Pie chart', style=tab_style, selected_style=tab_selected_style, children = [html.Div([dcc.Slider(id='year-slider',
                                   min=1946,
                                   max=2019,
                                   step=1,
                                   value=1995), html.Div(id='graph-with-slider')])]),
            dcc.Tab(label = 'Geolocation', style=tab_style, selected_style=tab_selected_style, children = [html.Div([dcc.Graph(id='geolocation-plot',config={'scrollZoom': True}, figure = {'data': [go.Scattermapbox(lon = lon, lat = lat,
                                                                                                                        mode = 'markers',
                                                                                                                        text = text_display,
                                                                                                                        marker=go.scattermapbox.Marker(size=normalized))], 'layout': go.Layout(autosize=False, width=1300, height=600, hovermode='closest',
                                                       mapbox=dict(accesstoken=mapbox_access_token, bearing=0,
                                                                   center=dict(lat=53, lon=-100),
                                                                   pitch=35,
                                                                   zoom=3),
                                                       title='Total amount of Dairy products by Provinces' )})])])], parent_style={'float': 'left'}),
   html.Div(id='tab-output', style={'float': 'left', 'width': '400'})
   ])]))

# call backs for data table
@app.callback(dash.dependencies.Output('datatable', 'data'),
             [dash.dependencies.Input('measure-column', 'value'),
              dash.dependencies.Input('age-column', 'value')])

def update_rows(selected_measure,selected_age):
    filtered = data.loc[(data['GEO'] == selected_measure) & (data['Dairy distribution'] == selected_age)]
    return filtered.to_dict('rows')


@app.callback(dash.dependencies.Output('graph-with-dropdown', 'children'),[dash.dependencies.Input('my-dropdown', 'value')])


def update_figure(value):
    data_milk = data[data['Dairy distribution'] == value].groupby(['Dairy distribution','GEO']).sum().unstack()['VALUE']
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
                  'layout':go.Layout(title = ('Distribution by category for ' +str(value) + ' in Kilolitres'),width = 1300, height = 500,
                                     titlefont=dict(family='Arial',size=18,color='black'),
                                         yaxis=dict(title='Tonnes',titlefont=dict(family='Arial',size=18,color='black')),
                                         xaxis=dict(title='Category',titlefont=dict(family='Arial',size=18,color='black')))})])

# Cummulative Piechart of total supply of milk Products from 1976 to 2019
# Piechart with year slider or some sort of cummulative slider by year since we can sum up all the records that are in 1976, 1977, etc.

@app.callback(dash.dependencies.Output('graph-with-slider', 'children'), [dash.dependencies.Input('year-slider', 'value')])
def update_figure1(value):
    data_pie = df.groupby(['YEAR', 'Dairy distribution'])['VALUE'].sum().reset_index().copy()
    data_pie = data_pie[data_pie.YEAR == str(value)]
    pie_labels = data_pie['Dairy distribution'].tolist()
    pie_values = data_pie['VALUE'].tolist()

    return html.Div([ dcc.Graph(id='supply-with-slider',
                           figure=go.Figure(
                               data=[go.Pie(labels=pie_labels,
                                            values=pie_values,
                                            hoverinfo='label+value',
                                            textinfo='percent',
                                            textfont=dict(size=10),
                                            marker=dict(colors=colors,
                                                        line=dict(color='#000000', width=2)))],
                               layout=go.Layout(
                                   title='Distribution of dairy products '+str(value),
                                   height=650,
                                   width=1400
                               )))])


if __name__ == '__main__':
    app.run_server(debug=True)
