import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import requests
import base64
import io
from PIL import Image
import numpy as np
import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("图像标定工具", style={'textAlign': 'center'}),
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                '拖拽或 ',
                html.A('选择图像')
            ]),
            style={
                'width': '98%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        dcc.Graph(id='graph', style={'height': '80vh'}),
        html.Button('提交', id='submit-button', n_clicks=0, style={'margin': '10px'}),
        dcc.Store(id='point-data'),
        dcc.Store(id='image-data'),
        html.Div(id='output-image', style={'marginTop': '20px'})
    ], style={'width': '70%', 'display': 'inline-block', 'padding': '20px'}),
    html.Div([
        html.H3('工具栏', style={'textAlign': 'center'}),
        daq.ToggleSwitch(
            id='draw-mode',
            label='绘制模式',
            value=False,
            style={'margin': '20px'}
        )
    ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '5px'})
])

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded))

@app.callback(
    Output('graph', 'figure'),
    Output('point-data', 'data'),
    Input('upload-image', 'contents'),
    Input('graph', 'clickData'),
    State('point-data', 'data')
)
def update_graph_and_points(contents, clickData, data):
    if contents is None:
        return go.Figure(), data
    
    image = parse_contents(contents)
    image_np = np.array(image)
    height, width, _ = image_np.shape

    if clickData is not None:
        if data is None:
            data = []
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        x_pixel = int(x)
        y_pixel = int(y)
        data.append({'x': x_pixel, 'y': y_pixel})

    fig = go.Figure()
    fig.add_trace(go.Image(z=image_np))

    if data is not None:
        fig.add_trace(go.Scatter(
            x=[point['x'] for point in data],
            y=[point['y'] for point in data],
            mode='markers',
            marker={'size': 12, 'color': 'red'}
        ))

    fig.update_layout(
        clickmode='event+select',
        dragmode=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, width], scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[height, 0])
    )
    return fig, data

@app.callback(
    Output('submit-button', 'n_clicks'),
    Output('output-image', 'children'),
    Input('submit-button', 'n_clicks'),
    State('point-data', 'data'),
    State('upload-image', 'contents')
)
def submit_points(n_clicks, data, contents):
    if n_clicks > 0 and data is not None and contents is not None:
        # 将点信息发送到C++程序
        # response = requests.post('http://localhost:5000/submit', json={'points': data})
        print("data: ",data)
    #     if response.status_code == 200:
    #         # 显示返回的结果图像
    #         result_image = response.json().get('result_image')
    #         return 0, html.Img(src='data:image/png;base64,{}'.format(result_image), style={'width': '100%'})
    return 0, None

if __name__ == '__main__':
    app.run_server(debug=True)