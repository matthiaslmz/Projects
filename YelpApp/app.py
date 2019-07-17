import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import pandas as pd
import plotly.graph_objs as go

#API KEYS AND DATASET
mapbox_access_token = 'pk.eyJ1IjoiZGVzY2l1aXR2IiwiYSI6ImNqcnZ1OWZ6MjA1eDYzeXBpOG5sZWd1NGcifQ.GQV-bvZFT62xzUBu2d4s6g'
map_style = "mapbox://styles/desciuitv/cjrwd4gn215au1ftfaphiuhhl"
df = pd.read_csv("mainplot.csv")

#SELECTING unique cities,categories, and stars for dropdown menu and slider
city_indicators = df['city'].unique()
star_rating_indicators = sorted(df['stars'].unique())
category_indicators = sorted(df['categories'].unique())

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div(
    html.Div([
        html.Div(
            [
                html.H1(children='Yelp Interactive Visualization',
                        className='twelve columns'),
                html.P('An interactive visualization for the top 15 cities with the most reviews from the Yelp dataset',
                        className='twelve columns')
            ], className="row"
        ),
            
        #Dropdown menu and slider
        html.Div(
            [
                #City Dropdown    
                html.Div(
                    [   
                        html.P('Choose City:',
                               style = {'fontWeight':600}
                        ),
                        
                        dcc.Dropdown(id='city-column',
                                     options=[{'label': i, 'value': i} for i in city_indicators],
                                     value='Greater Toronto'
                        ),
                    ],
                    className='six columns',
                    style={'margin-top': '10'}
                ),
                 
                #Star Ratings Slider 
                html.Div(
                    [
                        html.P('Star Ratings:',
                               style = {'fontWeight':600}
                        ),
                        
                        dcc.RangeSlider(id='star-slider',
                                     marks={i: str(i)+' Star' for i in star_rating_indicators},
                                     min = 1,
                                     max = 5,
                                     step = 0.5,
                                     updatemode="drag",
                                     value=[1, 5]
                        ),
                    ],
                    
                    className='two columns',
                    style={'margin-top': '0'}
                ),
                    
                #Category Dropdown 
                html.Div(
                    [
                        html.P('Category:',
                               style = {'fontWeight':600}
                        ),
                        
                        dcc.Input(id='category-column',
                                  placeholder= "Enter a category.. e.g., desserts",
                                  type="text",
                                  value="dessert"
                                     
                        ),
                    ],
                    className='two columns',
                    style={'margin-top': '0'}
                ),
            ], className="row"
        ),
        
     
        html.Div (
            [
                #Map
                html.Div(
                    [
                    html.Div([
                           dcc.Graph(
                                    id='map-figure',
                                    config={
                                            'scrollZoom': True
                                    }
                           )
                        ], className= 'six columns',  style={'margin-top': '0'})
                    ]
                ),
                    
               #DataTable
               html.Div(
                    [
                    dt.DataTable(
                        id='datatable',
                        data=df.to_dict('rows'),
                        columns=[
                                {"name": i, "id": i, "deletable": True} for i in df[['name','stars','review_count','address','categories']].columns
                        ],
                        
                        style_header={'backgroundColor': 'rgb(30, 30, 30)'},
                        
                        style_cell={'textAlign': 'left', 
                                'minWidth': '160px', 'width': '160px', 'maxWidth': '160px',
                                'whiteSpace': 'normal',
                                'backgroundColor': 'rgb(0, 0, 51)',
                                'color': 'white'
                            },
                        css=[{
                                'selector': '.dash-cell div.dash-cell-value',
                                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                            }],
                                    
                        style_cell_conditional=[
                            {'if': {'column_id': 'name'},
                             'width': '30%',
                             'textAlign': 'left'},
                            {'if': {'column_id': 'stars'},
                             'width': '15%',
                             'textAlign': 'left'},
                            {'if': {'column_id': 'review_counts'},
                             'width': '15%',
                             'textAlign': 'left'},
                            {'if': {'column_id': 'address'},
                             'width': '20%',
                             'textAlign': 'left'},
                            {'if': {'column_id': 'categories'},
                             'width': '20%',
                             'textAlign': 'left'},
                        ],
                                
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
                ],className="six columns", style={'margin-top': '20'}
                )
    ], className="row"
    )

]))
                    
@app.callback(dash.dependencies.Output('datatable', 'data'),
             [dash.dependencies.Input('city-column', 'value'),
              dash.dependencies.Input('category-column', 'value'),
              dash.dependencies.Input('star-slider', 'value')])
                    
def update_rows(selected_city,selected_category,star):
    filtered = df.loc[ ( df.city == selected_city ) &
                  ( df.categories.str.contains(selected_category)) &
                  ( df['stars'].between(star[0],star[1],inclusive=True) ) 
              ]
    return filtered.to_dict('rows')
                        
@app.callback(
    dash.dependencies.Output('map-figure', 'figure'),
    [dash.dependencies.Input('city-column', 'value'),
     dash.dependencies.Input('category-column', 'value'),
     dash.dependencies.Input('star-slider', 'value')])

def update_mapfigure(selected_city,selected_category, star):
    
    
    filtered_df = df.loc[ ( df.city == selected_city ) &
                          ( df.categories.str.contains(selected_category)) &
                          ( df['stars'].between(star[0],star[1],inclusive=True)) 
                  ]
    my_text=['Name:' + name +'<br>Stars:' + str(stars) +'<br>Number of Reviews:' + str(revs)
    for name, stars, revs in zip(list(filtered_df['name']), 
                                 list(filtered_df['stars']),
                                 list(filtered_df['review_count'])
                                )
            ] 
    
    trace = [go.Scattermapbox(lat=filtered_df['latitude'],
                              lon=filtered_df['longitude'],
                              text=my_text,
                              mode='markers',
                              marker=dict(size=6, color='gold', opacity=.5)
            )]
                
    layout_map = dict(
            #autosize=True,
            hovermode='closest',
            width=720,
            height=580,
            margin=dict(l=0, r=0,b=0, t=0
                   ),
            plot_bgcolor="#191A1A",
            paper_bgcolor="#020202",
            mapbox=dict(
                    accesstoken=mapbox_access_token,
                    bearing=50,
                    center=dict(
                            lat=float(filtered_df.city_lat.values[0]),
                            lon=float(filtered_df.city_long.values[0]),
                            ),
                    pitch=50,
                    zoom=12,
                    style=map_style
                    )
                )
    return {
        'data': trace,
        'layout': layout_map
    }

if __name__ == '__main__':
    app.run_server(debug=True,port=8020)