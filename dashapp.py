import plotly.graph_objects as go
import dash
#import dash_ui as dui
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import base64
import io
import os
import pickle
import dropbox
import cv2
#import time

#my_css = ['https://cdn.jsdelivr.net/npm/bootswatch@4.5.2/dist/cyborg/bootstrap.min.css']
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

DROPBOX_REFRESH_TOKEN = 'zXS2ed9DnA8AAAAAAAAAAUnaiyzwI7w4G30_VWxASZO07T2ScBgLNZZLQM3jWrE0'

dbx = dropbox.Dropbox(
    app_key='p0vdxr2klkwc89v',
    app_secret='dtujihdjolkv24m',
    oauth2_refresh_token = DROPBOX_REFRESH_TOKEN)

#folders = dropbox_listfolders()

testimg = Image.open("Muh.png")
routeimg = Image.open("RouteWall_Points2.png")

data = {}
metadata = {}

margin_dd = '20px'
defl = {'margin':{'l': 10, 'r': 25, 't': 40, 'b': 40}, 'height': 330, 'clickmode':'event'}    # Default layout for graph figures
styleimgN = {'margin':'20px','vertical-align':'center','display':'none',
                                'margin-left':'auto', 'margin-right':'auto'}    # Default layout for image figure no display
styleimgB = {'margin':'20px','vertical-align':'center','display':'block',
                                'margin-left':'auto', 'margin-right':'auto'}    # Default layout for image figure display block

#datapath = r'D:\Github\Speed-Advanced\SavedData'
#folders = os.listdir(datapath)

holds = np.cumsum(np.asarray([[0, 0], [125, 250], [-750, 750], [-750, 876], [500, 750], [375, 501], [-375, 750],
                                   [500, 876], [250, 750], [-1125, 626], [250, 625], [-500, 250], [625, 876],
                                   [-500, 625], [375, 501], [-125, 250], [-750, 625], [1250, 751], [500, 625],
                                   [-750, 375]]),axis=0) + np.array([2250, 1689])

scale = np.array([2.4/3000, 20/15000])
holdspos = holds*scale + np.array([0, -10])
holdstext = ['Hold %d'%i for i in range(20)]

rf1 = go.Figure()

#xmesh, ymesh = np.meshgrid(np.linspace(0,4.2,11),np.linspace(-30,10,11))

rf1.update_layout(
    margin=dict(l=0, r=0, t=0,b=0),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
    plot_bgcolor="white",
    clickmode='event',
    xaxis_range=[0,2.9],
    yaxis_range=[-10,10]
)

rf1.add_layout_image(
    dict(
        source=routeimg,
        xref="x",
        yref="y",
        x=0,
        y=10,
        sizex=2.9,
        sizey=19.99,
        sizing="contain",
        opacity=1,
        layer="below"
    )
)

progress1 = html.Div(
    [
        #dcc.Interval(id="progress-interval1", n_intervals=0, interval=100),
        #dbc.Progress(id="progress1", value=100, color='Gray', animated=False, striped=True),
        #dcc.Loading(id='loading1', type='default', children=html.Div(id='loading-output1'))
    ]
)

progress2 = html.Div(
    [
        #dcc.Interval(id="progress-interval2", n_intervals=0, interval=100),
        #dbc.Progress(id="progress2",  value=100, color='Gray', animated=False, striped=True),
        #dcc.Loading(id='loading2', type='default', children=html.Div(id='loading-output2'))
    ]
)

load1 = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id='dropdown_load1',
                        style={'color': 'black', 'text-align': 'center', 'margin':'1px'},
                        placeholder='Please select...',
                    ), md=4
                ),
                # dbc.Col(
                #         dcc.Upload(html.Button('Upload File'),
                #                id='upload-data1'), md=2
                # ),
                dbc.Col(progress1 ,align='center', md=2),
                dbc.Col(
                    html.Div(id='placeholderload1', style={'display':'none'}),md=1
                ),
                dbc.Col(
                        dcc.Input(id='input_ath1',
                               type='text',
                               placeholder='Athlete 1',
                               readOnly=True,
                               style={'textAlign':'center', 'margin':'4px'}), md=2
                ),
                dbc.Col(
                        dcc.Input(id='input_ath2',
                               type='text',
                               placeholder='Athlete 2',
                               readOnly=True,
                               style={'textAlign':'center', 'margin':'4px'}), md=2
                )
            ]
        )
    ],
    style={"height": "4vh"}
)

load2 = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                                 id='dropdown_load2',
                                 style={'color': 'black', 'text-align': 'center', 'margin':'1px'},
                                 placeholder='Please select...',
                                 ), md=4
                ),
                # dbc.Col(
                #         dcc.Upload(html.Button('Upload File'),
                #                id='upload-data2'), md=2
                # ),
                #dbc.Col(dbc.Progress(id='progressbar2', label="50%", value=50), align='center', md=2),
                dbc.Col(progress2, align='center', md=2),
                dbc.Col(
                    html.Div(id='placeholderload2', style={'display':'none'}),md=1
                ),
                dbc.Col(
                        dcc.Input(id='input_ath1_2',
                               type='text',
                               placeholder='Athlete 1',
                               readOnly=True,
                               style={'textAlign':'center', 'margin':'4px'}), md=2
                ),
                dbc.Col(
                        dcc.Input(id='input_ath2_2',
                               type='text',
                               placeholder='Athlete 2',
                               readOnly=True,
                               style={'textAlign':'center', 'margin':'4px'}), md=2
                )
            ]
        )
    ],
    style={"height": "4vh"}
)

controls1 = dbc.Card(
    [
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown_ath1',
                    style={'color':'black', 'text-align':'center','margin-bottom': margin_dd},
                    placeholder='Athlete',
                    disabled=True,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown_angle1',
                    style={'color': 'black', 'text-align':'center','margin-bottom': margin_dd},
                    placeholder='Angle',
                    maxHeight=140,
                    disabled=True,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown_COG1',
                    style={'color': 'black', 'text-align':'center','margin-bottom': margin_dd},
                    placeholder='Center of Gravity',
                    maxHeight=140,
                    disabled=True,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown_JVEL1',
                    style={'color': 'black', 'text-align':'center','margin-bottom': margin_dd},
                    placeholder='Joint Velocity',
                    maxHeight=140,
                    disabled=True,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown_IMG1',
                    style={'color': 'black', 'text-align':'center','margin-bottom': margin_dd},
                    placeholder='Images',
                    disabled=False,
                    options=['testimg']
                ),
            ]
        ),
    ],
    body=True,
    style={"height": "41vh"}
)

controls2 = dbc.Card(
    [
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown_ath2',
                    style={'color':'black', 'text-align':'center', 'margin-bottom': margin_dd},
                    placeholder='Athlete',
                    disabled=True,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown_angle2',
                    style={'color': 'black', 'text-align':'center', 'margin-bottom': margin_dd},
                    placeholder='Angle',
                    maxHeight=140,
                    disabled=True,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown_COG2',
                    style={'color': 'black', 'text-align':'center', 'margin-bottom': margin_dd},
                    placeholder='Center of Gravity',
                    maxHeight=140,
                    disabled=True,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown_JVEL2',
                    style={'color': 'black', 'text-align':'center','margin-bottom': margin_dd},
                    placeholder='Joint Velocity',
                    maxHeight=140,
                    disabled=True,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id='dropdown_IMG2',
                    style={'color': 'black', 'text-align':'center','margin-bottom': margin_dd},
                    placeholder='Images',
                    disabled=False,
                ),
            ]
        ),
    ],
    body=True,
    style={"height": "41vh"}
)

plot1 = dbc.Card(
    [
        html.Div(
            [
                html.Img(id='Image1', src=testimg, height='300px', style=styleimgN),
                dcc.Graph(id='Graph1', figure=go.Figure(), style={'display':'block'}),
            ]
        ),
        html.Div(
            [
                dcc.Slider(0, 100, 1, value=0, marks=None, id='Slider1',
                           tooltip={"placement": "bottom", "always_visible": True}),
            ]
        )
    ],
    style={"height": "41vh"}
)

plot2 = dbc.Card(
    [
        html.Div(
            [
                html.Img(id='Image2', src=testimg, height='300px', style=styleimgN),
                dcc.Graph(id='Graph2', figure=go.Figure(), style={'display': 'block'}),
            ]
        ),
        html.Div(
            [
                dcc.Slider(0, 100, 0, value=1, marks=None, id='Slider2',
                           tooltip={"placement": "bottom", "always_visible": True}),
            ]
        )
    ],
    style={"height": "41vh"}
)

route = dbc.Card(
    [
    dbc.Row(
        [
            dbc.Col(
                dcc.Graph(id='routemap1',figure=rf1,
                          style=dict(margin='3px',height='89vh'),
                          config=dict(displayModeBar=False)),md=6
            ),
            # dbc.Col(
            #     html.Div(id='placeholder', style={'display':'none'}),md=1
            # ),
            dbc.Col(
                dcc.Graph(id='routemap2', figure=rf1,
                          style=dict(margin='3px', height='89vh'),
                          config=dict(displayModeBar=False)), md=6
            )
        ], align='center'
    )
    ],
    style={"height": "90vh"}
)

app.layout = dbc.Container(
    [
        dcc.Store(id='Data'),
        dcc.Store(id='Metadata'),
        #html.Div(dcc.Dropdown(id='hidden', style={'display': 'none'})),
        dbc.Row([
            dbc.Col(html.H1("Speed Climbing Data Visualizer", style={"height": "5vh"}),md=11),
            dbc.Col(dcc.Loading(id='loading1', type='default', children=html.Div(id='loading-output1'), style={'margin-top':'70px'}),md=1)
        ]),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                    dbc.Col(load1),
                    dbc.Col(load2),
                    dbc.Row(
                        [
                            dbc.Col(controls1, md=3),
                            dbc.Col(plot1, md=9),
                            #dbc.Col(route, md=4),
                        ],
                        align="center",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(controls2, md=3),
                            dbc.Col(plot2, md=9),
                        ],
                        align="center",
                    ),
                    ], md=9
                ),
                dbc.Col(route, md=3)
            ]
        )
    ],
    fluid=True,
)

def parse_contents(contents, filename):
    global data, metadata
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'xlsx' in filename:
        xls = pd.ExcelFile(io.BytesIO(decoded))
        data = {}
        data[xls.sheet_names[0]] = xls.parse(xls.sheet_names[0])
        data[xls.sheet_names[1]] = xls.parse(xls.sheet_names[1])

    return data

@app.callback(
    Output('dropdown_load1', 'options'),
    Output('dropdown_load2', 'options'),
    Input('dropdown_load1', 'value'),
    Input('dropdown_load2', 'value'))
def toggle_animation(dpl1, dpl2):
    global data, metadata
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger = ''
    else:
        trigger = ctx.triggered_id
    if trigger == 'dropdown_load1':
        return dropbox_listfolders(), dropbox_listfolders()
    elif trigger == 'dropdown_load2':
        return dropbox_listfolders(), dropbox_listfolders()
    else:
        data = {}
        metadata = {}
        return dropbox_listfolders(), dropbox_listfolders()

@app.callback(Output('dropdown_angle1', 'disabled'),
              Output('dropdown_COG1', 'disabled'),
              Output('dropdown_JVEL1', 'disabled'),
              Input('dropdown_ath1', 'value'))
def dropdownath1(value):
    if value:
        return False, False, False
    else:
        return True, True, True

@app.callback(Output('dropdown_angle2', 'disabled'),
              Output('dropdown_COG2', 'disabled'),
              Output('dropdown_JVEL2', 'disabled'),
              Input('dropdown_ath2', 'value'))
def dropdownath2(value):
    if value:
        return False, False, False
    else:
        return True, True, True

@app.callback(Output('Graph1', 'figure'),
              Output('routemap1', 'figure'),
              Output('dropdown_angle1', 'value'),
              Output('dropdown_COG1', 'value'),
              Output('dropdown_JVEL1', 'value'),
              Output('Slider1', 'disabled'),
              Output('Slider1', 'max'),
              Output('Slider1', 'value'),
              Output('Graph1', 'style'),
              Output('Image1', 'style'),
              Input('dropdown_ath1', 'value'),
              Input('dropdown_angle1', 'value'),
              Input('dropdown_COG1', 'value'),
              Input('dropdown_JVEL1', 'value'),
              Input('dropdown_IMG1','value'),
              Input('Slider1', 'value'),
              Input('Graph1', 'clickData'),
              Input('Graph1', 'figure'),
              Input('routemap1', 'figure'),
              Input('routemap1', 'clickData'))
def graph1plot(athvalue, angvalue, cogvalue, jvelvalue, imgvalue, slidervalue, clickdata, fig, routefig, clickData):
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger = ''
    else:
        trigger = ctx.triggered_id
        if trigger != 'dropdown_IMG1':
            fig = go.Figure(data=fig['data'], layout=fig['layout'])
            if athvalue:
                px, py, vel = calcroute(athvalue)
            else:
                fig.data = []
                return fig, rf1, '', '', '', True, 0, 0, dash.no_update, dash.no_update
    if trigger == 'dropdown_angle1':
        ht = '%{y:.2f}°<extra></extra>'
        return plotData(athvalue, angvalue, trigger, ht)
    elif trigger == 'dropdown_COG1':
        if cogvalue == 'BC Velocity':
            ht = '%{y:.2f} m/s<extra></extra>'
        elif cogvalue == 'BC Acceleration':
            ht = '%{y:.2f} m/s^2<extra></extra>'
        else:
            ht = '%{y:.2f} m<extra></extra>'
        return plotData(athvalue, cogvalue, trigger, ht)
    elif trigger == 'dropdown_JVEL1':
        ht = '%{y:.2f} m/s<extra></extra>'
        return plotData(athvalue, jvelvalue, trigger, ht)
    elif trigger == 'dropdown_IMG1':
        return dash.no_update, dash.no_update, '', '', '', dash.no_update, dash.no_update, dash.no_update, {'display':'none'}, styleimgB
    elif trigger == 'Slider1' or trigger == 'Graph1' or trigger == 'routemap1':
        fig = go.Figure(data=fig['data'], layout=fig['layout'])
        if not len(fig.data):
            return [dash.no_update for i in range(10)]
        if len(fig.data) > 1:
            temp = [d for d in fig.data if d.name != 'SliderBar']
            remitem = [d for d in fig.data if d.name == 'SliderBar'][0]
            fig.data = temp
        x, y = fig.data[0].x, fig.data[0].y
        ylim = fig.layout.yaxis.range
        if trigger == 'Slider1':
            fig.add_trace(go.Scatter(x=[x[slidervalue], x[slidervalue]], y=ylim, mode='lines', hoverinfo='skip', name='SliderBar'))
        elif trigger == 'Graph1':
            cx = clickdata['points'][0]['x']
            dat = np.array(x)
            slidervalue = np.where(dat == min(dat, key=lambda var: abs(var - cx)))[0][0]
            fig.add_trace(go.Scatter(x=[cx, cx], y=ylim, mode='lines', hoverinfo='skip', name='SliderBar'))
        else:
            holdtext = clickData['points'][0]['text']
            hind = getHoldIndex(holdtext, athvalue)
            if hind is not None:
                slidervalue = int(hind)
                fig.add_trace(go.Scatter(x=[x[slidervalue], x[slidervalue]], y=ylim, mode='lines', hoverinfo='skip', name='SliderBar'))
            else:
                try:
                    fig.add_trace(remitem)
                except UnboundLocalError:
                    pass
        rfig = go.Figure(data=routefig['data'], layout=routefig['layout'])
        temp = [d for d in rfig.data if d.name in ['COGPath', 'holdspoints']]
        rfig.data = temp
        rfig = drawSkeleton(rfig, slidervalue, athvalue, px, py)
        fig.update_layout(showlegend=False)
        return fig, rfig, dash.no_update, dash.no_update, dash.no_update, False, int(len(y)) - 1, slidervalue, dash.no_update, dash.no_update
    else:
        fig = go.Figure(data=fig['data'], layout=defl)
        rfig = go.Figure(data=[], layout=routefig['layout'])
        try:
            rfig.add_trace(go.Scatter(x=px, y=py, mode='markers+lines', hovertemplate='%{text:.2f}<extra></extra>', text=['{}'.format(v) for v in vel],
                                      marker=dict(size=4, color=vel, colorscale='Jet', showscale=False), line=dict(width=4), name='COGPath'))
            rfig.add_trace(go.Scatter(x=list(holdspos[:, 0]), y=list(holdspos[:, 1]), opacity=0, mode='markers', hoverinfo='text',
                           text=holdstext, hoverlabel=dict(bgcolor='blue'), name='holdspoints'))
            rfig.update_layout(showlegend=False)
        except NameError:
            pass
        return fig, rfig, dash.no_update, dash.no_update, dash.no_update, True, 0, 0, dash.no_update, dash.no_update

@app.callback(Output('Graph2', 'figure'),
              Output('routemap2', 'figure'),
              Output('dropdown_angle2', 'value'),
              Output('dropdown_COG2', 'value'),
              Output('dropdown_JVEL2', 'value'),
              Output('Slider2', 'disabled'),
              Output('Slider2', 'max'),
              Output('Slider2', 'value'),
              Output('Graph2', 'style'),
              Output('Image2', 'style'),
              Input('dropdown_ath2', 'value'),
              Input('dropdown_angle2', 'value'),
              Input('dropdown_COG2', 'value'),
              Input('dropdown_JVEL2', 'value'),
              Input('Slider2', 'value'),
              Input('Graph2', 'clickData'),
              Input('Graph2', 'figure'),
              Input('routemap2', 'figure'),
              Input('routemap2', 'clickData'))
def graph2plot(athvalue, angvalue, cogvalue, jvelvalue, slidervalue, clickdata, fig, routefig, clickData):
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger = ''
    else:
        trigger = ctx.triggered_id
        fig = go.Figure(data=fig['data'], layout=fig['layout'])
        if athvalue:
            px, py, vel = calcroute(athvalue)
        else:
            fig.data = []
            return fig, rf1, '', '', '', True, 0, 0, dash.no_update, dash.no_update
    if trigger == 'dropdown_angle2':
        ht = '%{y:.2f}°<extra></extra>'
        return plotData(athvalue, angvalue, trigger, ht)
    elif trigger == 'dropdown_COG2':
        if cogvalue == 'BC Velocity':
            ht = '%{y:.2f} m/s<extra></extra>'
        elif cogvalue == 'BC Acceleration':
            ht = '%{y:.2f} m/s^2<extra></extra>'
        else:
            ht = '%{y:.2f} m<extra></extra>'
        return plotData(athvalue, cogvalue, trigger, ht)
    elif trigger == 'dropdown_JVEL2':
        ht = '%{y:.2f} m/s<extra></extra>'
        return plotData(athvalue, jvelvalue, trigger, ht)
    elif trigger == 'Slider2' or trigger == 'Graph2' or trigger == 'routemap2':
        fig = go.Figure(data=fig['data'], layout=fig['layout'])
        if not len(fig.data):
            return [dash.no_update for i in range(10)]
        if len(fig.data) > 1:
            temp = [d for d in fig.data if d.name != 'SliderBar']
            remitem = [d for d in fig.data if d.name == 'SliderBar'][0]
            fig.data = temp
        x, y = fig.data[0].x, fig.data[0].y
        ylim = fig.layout.yaxis.range
        if trigger == 'Slider2':
            fig.add_trace(go.Scatter(x=[x[slidervalue], x[slidervalue]], y=ylim, mode='lines', hoverinfo='skip', name='SliderBar'))
        elif trigger == 'Graph2':
            cx = clickdata['points'][0]['x']
            dat = np.array(x)
            slidervalue = np.where(dat == min(dat, key=lambda var: abs(var - cx)))[0][0]
            fig.add_trace(go.Scatter(x=[cx, cx], y=ylim, mode='lines', hoverinfo='skip', name='SliderBar'))
        else:
            holdtext = clickData['points'][0]['text']
            hind = getHoldIndex(holdtext, athvalue)
            if hind is not None:
                slidervalue = int(hind)
                fig.add_trace(go.Scatter(x=[x[slidervalue], x[slidervalue]], y=ylim, mode='lines', hoverinfo='skip', name='SliderBar'))
            else:
                try:
                    fig.add_trace(remitem)
                except UnboundLocalError:
                    pass
        rfig = go.Figure(data=routefig['data'], layout=routefig['layout'])
        temp = [d for d in rfig.data if d.name in ['COGPath', 'holdspoints']]
        rfig.data = temp
        rfig = drawSkeleton(rfig, slidervalue, athvalue, px, py)
        fig.update_layout(showlegend=False)
        return fig, rfig, dash.no_update, dash.no_update, dash.no_update, False, int(len(y)) - 1, slidervalue, dash.no_update, dash.no_update
    else:
        fig = go.Figure(data=fig['data'], layout=defl)
        rfig = go.Figure(data=[], layout=routefig['layout'])
        try:
            rfig.add_trace(go.Scatter(x=px, y=py, mode='markers+lines', hovertemplate='%{text:.2f}<extra></extra>',
                                      text=['{}'.format(v) for v in vel],
                                      marker=dict(size=4, color=vel, colorscale='Jet', showscale=False), line=dict(width=4), name='COGPath'))
            rfig.add_trace(go.Scatter(x=list(holdspos[:, 0]), y=list(holdspos[:, 1]), opacity=0, mode='markers', hoverinfo='text',
                           text=holdstext, hoverlabel=dict(bgcolor='blue'), name='holdspoints'))
            rfig.update_layout(showlegend=False)
        except NameError:
            pass
        return fig, rfig, dash.no_update, dash.no_update, dash.no_update, True, 0, 0, dash.no_update, dash.no_update

@app.callback(Output('input_ath1', 'value'),
              Output('input_ath2', 'value'),
              Output('input_ath1_2', 'value'),
              Output('input_ath2_2', 'value'),
              Output('dropdown_ath1', 'options'),
              Output('dropdown_angle1', 'options'),
              Output('dropdown_COG1', 'options'),
              Output('dropdown_JVEL1', 'options'),
              Output('dropdown_ath1', 'disabled'),
              Output('dropdown_ath2', 'options'),
              Output('dropdown_angle2', 'options'),
              Output('dropdown_COG2', 'options'),
              Output('dropdown_JVEL2', 'options'),
              Output('dropdown_ath2', 'disabled'),
              Output('loading-output1', 'children'),
              Input('dropdown_load1', 'value'),
              Input('dropdown_load2', 'value'),
              State('input_ath1', 'value'),
              State('input_ath2', 'value'),
              State('input_ath1_2', 'value'),
              State('input_ath2_2', 'value'), prevent_initial_call=True)
def loaddata(selected1, selected2, ath1value, ath2value, ath1value2, ath2value2):
    cid = dash.callback_context.triggered_id
    global data, metadata
    if not cid:
        return '', '', '', '', [], [], [], [], True, [], [], [], [], True, dash.no_update

    if cid == 'dropdown_load1':
        if not selected1:
            return '', '', '', '', [], [], [], [], True, [], [], [], [], True, dash.no_update
        _metadata, xls, frames = dropbox_download_files(selected1)
        if ath1value and ath2value:
            del metadata[ath1value]; del metadata[ath2value]
            del data[ath1value]; del data[ath2value]
    else:
        if not selected2:
            return '', '', '', '', [], [], [], [], True, [], [], [], [], True, dash.no_update
        _metadata, xls, frames = dropbox_download_files(selected2)
        if ath1value2 and ath2value2:
            del metadata[ath1value2]; del metadata[ath2value2]
            del data[ath1value2]; del data[ath2value2]
        # fpath = os.path.join(datapath, selected1)
        # for file in os.listdir(fpath):
        #     if file.endswith(".xlsx"):
        #         ath1, ath2 = file.split('.')[0].split('_')
        #         break
        #
        # xls = pd.ExcelFile(os.path.join(fpath,file))

    ath1, ath2 = xls.sheet_names
    metadata[ath1], metadata[ath2] = unpackmeta(_metadata, 'Left'), unpackmeta(_metadata, 'Right')
    data[ath1], data[ath2] = xls.parse(ath1), xls.parse(ath2)
    #data = {xls.sheet_names[0]: xls.parse(xls.sheet_names[0]), xls.sheet_names[1]: xls.parse(xls.sheet_names[1])}
    #metafile = open(fpath + '/metadata.pkl', 'rb')
    #metadata[0] = pickle.load(metafile)
    #metafile.close()
    cols = list(data[ath1].keys())
    try:
        cols.remove('RightHeel')
        cols.remove('LeftHeel')
    except ValueError:
        pass
    optionsath = [{'label':key, 'value':key} for key in data.keys()]

    optionsangle = [{'label': key, 'value': key} for key in cols[:6]]
    optionscog = [{'label': key, 'value': key} for key in cols[7:11]]
    optionsjvel = [{'label': key, 'value': key} for key in cols[12:]]

    if cid == 'dropdown_load1':
        return ath1, ath2, dash.no_update, dash.no_update, optionsath, optionsangle, optionscog, optionsjvel,\
               False, optionsath, optionsangle, optionscog, optionsjvel, False, ''
    else:
        return dash.no_update, dash.no_update, ath1, ath2, optionsath, optionsangle, optionscog, optionsjvel, \
               False, optionsath, optionsangle, optionscog, optionsjvel, False, ''

def removeOutliersHL(_data, outlierConstant):
    a = np.array(_data)[:, 2]
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    eles = [_data[i] for i, x in enumerate(a) if x <= quartileSet[0] or x >= quartileSet[1]]
    result = _data.copy()
    result = [x for x in result if x not in eles]

    return result

def getHoldIndex(holdtext, ath):
    try:
        holdslist = removeOutliersHL(metadata[ath]['Holds'][holdtext], 0.5)
    except KeyError:
        return None
    # fnrs = np.array(holdslist)[:, 0]
    # coords = metadata[ath]['KeyPoints']['RWrist'][fnrs, :]
    # xy = np.array(holdslist)[:, 1:3]
    # dist, _min = np.argmin(np.linalg.norm(coords - xy, axis=1)), np.min(np.linalg.norm(coords - xy, axis=1))
    # return fnrs[dist] if _min < 50 else None
    for hold in holdslist:
        fnr, hx, hy, _, _ = hold
        rx, ry = metadata[ath]['KeyPoints']['RWrist'][fnr]
        lx, ly = metadata[ath]['KeyPoints']['LWrist'][fnr]
        if np.linalg.norm(np.array([hx - rx, hy - ry])) < 40 or \
                np.linalg.norm(np.array([hx - lx, hy - ly])) < 40:
            return fnr
    return None

def plotData(athvalue, datavalue, trigger, ht):
    fig = go.Figure(data=[], layout=defl)
    if datavalue is None:
        return fig, dash.no_update, dash.no_update, dash.no_update, dash.no_update, True, 0, 0
    y = interNaNs(data[athvalue][datavalue].values)
    trace = go.Scatter(
        x=np.linspace(0, len(y) - 1, len(y)) / 24,
        y=y,
        name=datavalue,
        hovertemplate=ht
    )
    ran = np.max(y) - np.min(y)
    ylim = [np.min(y) - 0.1 * ran, np.max(y) + 0.1 * ran]
    fig.add_trace(trace)
    fig.update_layout(yaxis=dict(range=[ylim[0], ylim[1]]))
    if trigger[:-1] == 'dropdown_angle':
        return fig, dash.no_update, dash.no_update, '', '', False, int(len(y)) - 1, 0, {'display':'block'}, styleimgN
    elif trigger[:-1] == 'dropdown_COG':
        return fig, dash.no_update, '', dash.no_update, '', False, int(len(y)) - 1, 0, {'display':'block'}, styleimgN
    elif trigger[:-1] == 'dropdown_JVEL':
        return fig, dash.no_update, '', '', dash.no_update, False, int(len(y)) - 1, 0, {'display':'block'}, styleimgN

def drawSkeleton(fig, index, ath, x, y):
    kpts = metadata[ath]['KeyPoints']

    t = 1  # OpenPose 1, MMpose 0
    if len(kpts) > 18:
        bodyparts = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                     [7, 8], [8, 9], [9, 10], [10, 21], [21, 19],
                     [7, 11], [11, 12], [12, 13], [13, 18], [18, 16]]
        t = 1
    else:
        bodyparts = [[15, 3], [3, 5], [5, 7], [15, 2], [2, 4],
                    [4, 6], [3, 9], [2, 8], [16, 9], [9, 11], [11, 13], [16, 8],
                    [8, 10], [10, 12]]
        t = 0

    try:
        cond = (~np.isnan(data[ath]['RightKnee'][index]) and ~np.isnan(data[ath]['LeftKnee'][index])) \
               or (~np.isnan(data[ath]['LeftElbow'][index]) and ~np.isnan(data[ath]['RightElbow'][index]))
    except KeyError:
        cond = False

    ms = 5
    lw = 2

    if cond:
        k = []
        for kpt in kpts.keys():
            k.append(kpts[kpt][index])

        sx, sy = scale * 1000
        k2 = (np.array(k) - np.array(k[-1])) * metadata[ath]['Scale'][index]
        kx, ky = k2[:, 0] * sx + x[index], -k2[:, 1] * sy + y[index]
        kx, ky = np.clip(kx, 0, 2.9), np.clip(ky, -10, 10)
        if t:
            head = np.array([(kx[14] + kx[15]) / 2, (ky[14] + ky[15]) / 2])
            fig.add_trace(go.Scatter(x=[head[0], kx[0]], y=[head[1], ky[0]], mode='markers+lines', hoverinfo='skip',
                                     name='Head', marker=dict(color='Blue', size=ms), line=dict(color='Blue', width=lw)))
        else:
            head = np.array([kx[14], ky[14]])
            fig.add_trace(go.Scatter(x=[head[0], kx[15]], y=[head[1], ky[15]], mode='markers+lines', hoverinfo='skip',
                                     name='Head', marker=dict(color='Blue', size=ms), line=dict(color='Blue', width=lw)))

        i = 0
        for a, b in bodyparts:
            fig.add_trace(go.Scatter(x=[kx[a], kx[b]], y=[ky[a], ky[b]], mode='lines', hoverinfo='skip',
                                     name='Bodypart%d'%i, line=dict(color='Blue', width=lw)))
            i+=1
        if t:
            kx = np.delete(kx, [14, 15, 22])
            ky = np.delete(ky, [14, 15, 22])
        else:
            kx = np.delete(kx, [0, 1, 17])
            ky = np.delete(ky, [1, 0, 17])

        fig.add_trace(go.Scatter(x=kx, y=ky, mode='markers', hoverinfo='skip', name='Keypoints', marker=dict(color='Blue', size=ms)))
        #axes.plot(kx, ky, 'bo', linewidth=0.2, markersize=3)
    return fig

def dropbox_download_files(folder):
    _, result = dbx.files_download(path='/Test/'+folder+'/metadata.pkl')
    metafile = pickle.load(io.BytesIO(result.content))

    files = dbx.files_list_folder('/Test/'+folder).entries
    for file in files:
        _, ending = file.name.split('.')
        if ending == 'xlsx':
            xlsfile = file.name
        elif ending == 'avi':
            imgfile = file.name

    _, result = dbx.files_download(path='/Test/'+folder+'/'+xlsfile)
    xls = pd.ExcelFile(result.content)
    #_, response = dbx.files_download(path='/Test/'+folder+'/'+imgfile)
    dbx.files_download_to_file(download_path='output.avi', path='/Test/' + folder + '/' + imgfile)
    frames = readFrames(metafile=metafile)

    return metafile, xls, frames

def dropbox_listfolders():
    temp = dbx.files_list_folder('/Test').entries
    FOLDERS = []
    for file in temp:
        FOLDERS.append(file.name)

    return FOLDERS

def calcroute(ath):
    x, y = interNaNs(data[ath]['BC PositionX'].values), interNaNs(data[ath]['BC PositionY'].values)
    sx, sy = scale*1000

    startx, starty = metadata[ath]['StartPos'] * np.array([sx, sy])
    x = x * sx + holdspos[0, 0] + startx
    y = y * sy + holdspos[0, 1] - starty
    vel = interNaNs(data[ath]['BC Velocity'].values)
    vel = np.clip(vel, 0, 4)

    return x, y, list(vel)

def interNaNs(y):
    nans, x = np.isnan(y), lambda z: z.nonzero()[0]
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    return y

def getColors(vec, colormap=plt.cm.jet ,vmin=0, vmax=4):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(vec))

def unpackmeta(meta, side):
    cont = {}
    for k in meta.keys():
        if k not in ['WallPosition','FPS']:
            cont[k] = meta[k][side]
        else:
            cont[k] = meta[k]

    return cont

def readFrames(metafile, response=None):
    FILE_OUTPUT = 'output.avi'
    if response is not None:
        if os.path.isfile(FILE_OUTPUT):
            os.remove(FILE_OUTPUT)

        out_file = open(FILE_OUTPUT, "wb")  # open for [w]riting as [b]inary
        out_file.write(response.content)
        out_file.close()

    kps = metafile['KeyPoints']

    cap = cv2.VideoCapture(FILE_OUTPUT)
    frames = {'Left':[], 'Right':[]}
    i = 0
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            conv = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            for key in kps.keys():
                temp = [kps[key][k][i, :] for k in kps[key].keys()]
                mins, maxs = np.min(temp, axis=0), np.max(temp, axis=0)
                left, bottom = mins - 100
                right, top = maxs + 100
                frame = Image.fromarray(conv)
                crop = frame.crop((left, bottom, right, top))
                frames[key].append(crop)
            i+=1
        else:
            break
    cap.release()
    return frames

if __name__ == '__main__':
    app.run_server(debug=True, port=8000, dev_tools_hot_reload=False)
