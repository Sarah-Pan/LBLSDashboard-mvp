import dash
from dash import html, dcc, Input, Output, State, DiskcacheManager, ALL
import diskcache
import os

from strategies import moral_licensing, reactance, anchor, social, collapse, stereo, rank, ambiguity

# Strategy mapping
STRATEGY_MAP = {
    'moral': moral_licensing,
    'reactance': reactance, 
    'anchor': anchor,
    'social': social,
    'collapse': collapse,
    'stereo': stereo,
    'rank': rank,
    'ambiguity': ambiguity
}

# Strategy descriptions
STRATEGY_INFO= {
    'moral':{
        'title': 'Concept: Moral Licensing',
        'description': r'''

        **Effect:** Student reduces effort when prediction exceeds the goal.

        **Example:** A student sees a predicted grade of 95 (above target 80) and relaxes, studying less.

        **Formula for Iteration t+1:**  
        $y_{t+1} = y_t - w [\hat{y}_t - y_{target}]_+  + \mathcal{E}_{noise}$

        '''
    },
    'reactance':{
        'title': 'Concept: Psychological Reactance',
        'description': r'''

        **Effect:** Student moves away from the prediction to assert autonomy.

        **Example:** A student predicted to score 70 feels controlled and works extra hard to prove the model wrong, aiming for 90.

        **Formula for Iteration t+1:**  
        $y_{t+1} = y_t - w(\hat{y}_t - y_t) + \mathcal{E}_{noise}$

        '''
    },
    'anchor':{
        'title': 'Concept: Anchoring Effect (+ Peer Effect)',
        'description': r'''

        **Effect:** Student subconsciously aligns actual performance with the prediction.

        **Example:** A student predicted to score 85 gradually adjusts effort to match that expectation, even if initially aiming for 80.

        **Formula for Iteration t+1:**  
        $y_{t+1} = y_t + w_{1}(\hat{y}_t - y_t) + w_{2}(\hat{y}_t - y_t) + \mathcal{E}_{noise}$
    '''
    },
    'social':{
        'title': 'Concept: Social Proof',
        'description': r'''

        **Effect:** Student adjusts performance to match predicted population average.

        **Example:** A student sees most peers predicted at 90 and increases effort to align with the group norm.

         **Formula for Iteration t+1:**  
        $y_{t+1} = y_t + w(\overline{\hat{y}}_t - y_t) + \mathcal{E}_{noise}$

        '''
    },
    'collapse':{
        'title': 'Concept: Collapse Effect',
        'description': r'''

        **Effect:** Student gives up when prediction falls below the critical threshold.

        **Example:** A student predicted to score 50 (below passing threshold 60) feels hopeless and stops trying.

        **Formula for Iteration t+1:**  
        $y_{t+1} = y_t - w [\hat{y}_t - \theta]_- + \mathcal{E}_{noise}$

        '''
    },
    'stereo':{
        'title': 'Concept: Stereotype Threat',
        'description': r'''

        **Effect:** Performance drops when minority agents feel underestimated.

        **Example:** A student from an underrepresented group sees a low predicted score and performs worse due to anxiety.

        **Formula for Iteration t+1:**  
        $y_{t+1} = y_t - w \cdot X_{bias} \cdot [y_t - \hat{y}_t]_+ + \mathcal{E}_{noise}$
        
        '''
    },
    'rank':{
        'title': 'Concept: Rank Anxiety',
        'description': r'''

        **Effect:** Student increases effort when predicted rank behind an expected threshold ($\tau$, fixed for all). 

        **Example:** A student predicted to rank 150th (threshold = 100) works harder to climb into the top 100.

        **Formula for Iteration t+1:**  
        $y_{t+1} = y_t + w [\hat{r} - \tau]_+ + \mathcal{E}_{noise}$

        '''
    },
    'ambiguity':{
        'title': 'Concept: Ambiguity Aversion',
        'description': r'''
        
        **Effect:** Student’s response decays as model uncertainty (variance) increases.

        **Example:** A student sees a prediction with high uncertainty (wide prediction interval) and hesitates to change study habits.

        **Formula for Iteration t+1:**  
        $y_{t+1} = y_t + w \cdot \exp(-\sigma^2) + \mathcal{E}_{noise}$


        '''
    }

}

# Strategy pattern
STRATEGY_DESCRIPTIONS = {
    'moral': (
        "High **predictions** (🔵) above the target cause the *nudged performance* (🔴) to drop significantly."
        "This drop drags down future **predictions**, creating a downward spiral for high-performing students."
    ),
    'reactance': (
        "*Nudged performance* (🔴) scatter toward the top and bottom extremes, moving away from the central **predictions** (🔵)."
        "Consequently, the **predictions**  also spread out widely over time."
    ),
    'anchor': (
        "*Nudged performance* (🔴) gravitate toward the **predictions** (🔵) as if pulled by a magnet."
        "Eventually, the student *nudged performance* and the **predictions** almost completely overlap."
    ),
    'social': (
        "*Nudged performance* (🔴) move toward the group average, causing the **predictions** (🔵) to flatten out."
        "Eventually, both *nudged performance* and prediction form a single horizontal line."
    ),
    'collapse': (
        "*Nudged performance* (🔴) that start below the threshold drop sharply as students give up."
        "This crash in *nudged performance* immediately pulls the subsequent **predictions** (🔵) down with them."
    ),
    'stereo': (
        "The Risk Group (▲) drops significantly, and so do their **predictions** (🔵)."
        "Meanwhile, the Safe Group (●) remains stable or rises slightly."
    ),
    'rank': (
        "Lower-ranked *nudged performance* (🔴) rise rapidly to catch up, pulling their **predictions** (🔵) upward."
        " In contrast, high-ranked students stay still and are eventually overtaken by the rising group."
    ),
    'ambiguity': (
        "Left side **predictions** (🔵) and *nudged performance* (🔴) (small variance) move quickly toward the top/bottom of the chart."
        "In contrast, right side **predictions** and *nudged performance* (large variance) get stuck and move much slower."
    ),

}

# Initialize Dash app with Diskcache for caching
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = dash.Dash(__name__, background_callback_manager=background_callback_manager)
app.config.suppress_callback_exceptions = True
app.title = "Nudging Simulation Lab"
# server = app.server

# Layout
app.layout = html.Div([
    # sidebar
    html.Div([
        html.H2("Nudging Simulation Lab", className = "sidebar-title"),
        html.Hr(),

        # Strategy Selection
        html.Label("Nudging Strategy:", className="section-title"),
        dcc.Dropdown(
            id = "strategy-dropdown",
            options=[
                {'label': 'Moral Licensing', 'value': 'moral'},
                {'label': 'Psychological Reactance', 'value': 'reactance'},
                {'label': 'Anchoring Effect', 'value': 'anchor'},
                {'label': 'Social Proof', 'value': 'social'},
                {'label': 'Collapse Effect', 'value': 'collapse'},
                {'label': 'Stereotype Threat', 'value': 'stereo'},
                {'label': 'Rank Anxiety', 'value': 'rank'},
                {'label': 'Ambiguity Aversion', 'value': 'ambiguity'}
            ],
            value='moral',
            clearable=False,
            className="dropdown-dark"
        ),

        html.Br(),
        html.Hr(),

        # Description Card
        html.Div([
            html.H3(id='desc-title', className="card-title"),
            html.Hr(),
            dcc.Markdown(
                id='desc-content', 
                className="markdown-content",
                mathjax = True)
        ], className="card description-card"),

        # Dynamic Parameter Inputs
        html.H4("Parameters Configuration", className="section-title"),
        html.Div(id='dynamic-controls-container'),

        html.Br(),
        html.Hr(),

        html.Label("Simulation Rounds (Iterations)", className="control-label"),
        dcc.Slider(
            id='slider-rounds',
            min=1,
            max=20,
            step=1,
            value=5,
            marks={1: '1', 5: '5', 10: '10', 15: '15', 20: '20'},
            tooltip={"placement": "bottom", "always_visible": True}
        ),

        html.Br(),

        # Run Button
        html.Button("Run Simulation", id="btn-run", n_clicks=0, className="btn-run"),

        # Progress Bar
        html.Div([
            html.Br(),
            html.Label("Simulation Progress:", className="progress-label"),
            html.Progress(id="progress-bar", value="0", max="100", style={'width': '100%'}, className="progress-bar"),
            html.Div(id="progress-text", className="progress-text"),
        ], id="progress-container", style={"display": "none"}),

        html.Br(),

        html.Div([
            html.H4("Current Experiment Settings", className="card-title"),
            html.Hr(),
            html.Div(id="experiment-info", className="experiment-info-text")
        ], 
        id="info-card",
        className="card experiment-settings-card",
        style={'display': 'none'}
        ),

        
    ], className="sidebar"),

    # main content
    html.Div([
        html.H1("Simulation Dashboard", className="main-title"),
        # Graph Container
        html.Div([
            dcc.Loading(
                id="loading-spinner",
                type="circle",
                children=html.Iframe(
                    id='animation-iframe', 
                    className="viz-iframe",
                    srcDoc=None
                )
            )
        ], className="card graph-card"), 
        dcc.Markdown(
        id='annotation-card',
        className='strategy-card',  
        children="Please select a strategy to see the pattern."
    )
    ], className="main-content", style={'padding': '50px'})
])

# Callback: Update description based on strategy
@app.callback(
    [Output('desc-title', 'children'),
     Output('desc-content', 'children')],
    [Input('strategy-dropdown', 'value')]
)
def update_description(selected_strategy):
    info = STRATEGY_INFO.get(selected_strategy, {})
    return info.get('title', 'Unknown Strategy'), info.get('description', 'No description available.')

# Callback: Generate dynamic parameters based on selected strategy
@app.callback(
    Output('dynamic-controls-container', 'children'),
    Input('strategy-dropdown', 'value')
)

def update_param_ui(strategy_name):
    module = STRATEGY_MAP.get(strategy_name)
    if not module:
        return html.Div("Module not found.", className="error-message")
    
    params_config = getattr(module, 'PARAMS', {})

    if not params_config:
        return html.Div("No adjustable parameters.", className="no-params-msg")
    
    controls = []
    for param_key, config in params_config.items():
        param_type = config.get('type', 'slider')
        label_component = dcc.Markdown(
            config.get('label', param_key), 
            className="control-label",
            mathjax=True,
            style={'marginBottom': '5px', 'fontWeight': 'bold'}
        )

        if param_type == 'dropdown':
            input_component = dcc.Dropdown(
                id={'type': 'param-slider', 'index': param_key}, 
                options=config.get('options', []),
                value=config.get('value'),
                clearable=False,
                style={'color': '#333'}
            )
        
        elif param_type == 'radio':
            input_component = dcc.RadioItems(
                id={'type': 'param-slider', 'index': param_key},
                options=config.get('options', []),
                value=config.get('value'),
                inline=True,
                labelStyle={'marginRight': '20px', 'cursor': 'pointer'},
                style={'paddingLeft': '12px','marginTop': '5px'}
            )

        else:
            input_component = dcc.Slider(
                id={'type': 'param-slider', 'index': param_key},
                min=config['min'],
                max=config['max'],
                step=config['step'],
                value=config['value'],
                marks={i: str(i) for i in range(int(config['min']), int(config['max'])+1, 10)} 
                      if config.get('max', 100) > 10 else None,
                tooltip={"placement": "bottom", "always_visible": True}
            )

        controls.append(
            html.Div([
                label_component,
                input_component,
                html.Br(),
                html.Br()
            ])
        )
    return controls

# Callback: Run simulation and update visualization
@app.callback(
    Output('animation-iframe', 'srcDoc'),
    Output("experiment-info", "children"),
    Output("info-card", "style"),
    Input("btn-run", "n_clicks"),
    State("strategy-dropdown", "value"),
    State("slider-rounds", "value"),
    State({'type': 'param-slider', 'index': ALL}, 'value'),
    State({'type': 'param-slider', 'index': ALL}, 'id'), 
    background=True,
    running=[
        (Output("btn-run", "disabled"), True, False),
        (Output("progress-container", "style"), {'display': 'block'}, {'display': 'none'}),
        (Output("info-card", "style"), {'display': 'none'}, {'display': 'none'}),
    ],
    progress=[
        Output("progress-bar", "value"), 
        Output("progress-bar", "max"), 
        Output("progress-text", "children")
    ],
    prevent_initial_call=True
)

def run_simulation_callback(set_progress, n_clicks, strategy_name, n_rounds, param_values, param_ids):
    if not n_clicks:
        return tuple([dash.no_update] * 5)
    
    module = STRATEGY_MAP.get(strategy_name)
    if not module:
        return "<h1>Error: Module not found.</h1>", "Error: Module not found."

    # unpack parameters into a dictionary
    kwargs = {}
    for pid, val in zip(param_ids, param_values):
        param_name = pid["index"]
        kwargs[param_name] = val

    # reporter function
    def _reporter(p):
        curr, total, msg = p
        set_progress((str(curr), str(total), f"Round {curr}/{total}: {msg}"))

    # run simulation
    y, nudged, pred, logs, info_dict = module.run_simulation(
        n_rounds=n_rounds,
        progress_callback=_reporter,
        **kwargs
    )

    html_content = module.generate_visualization(
        y=y,
        nudged_history = nudged,
        pred_history = pred,
        extra_data=info_dict,
        **kwargs
    )

    info_content = ""
    info_card_style = {'display': 'none'}

    if info_dict and 'construct_name' in info_dict:
        info_content = [
                html.Span("Construct (X): ", style={'font-weight': 'bold'}),
                html.Span(f"{info_dict.get('construct_name', 'Unknown')}", style={'margin-right': '25px', 'margin-bottom': '5px'}),
                html.Br(),
                html.Span("Threshold: ", style={'font-weight': 'bold'}),
                html.Span(f"{info_dict.get('risk_desc', '')} "),
                html.Span(f"(Cutoff: {info_dict.get('cutoff', 0):.2f})", style={'color': '#7f8c8d', 'font-size': '0.9em'})
            ]
    
        info_card_style = {'display': 'block', 'border-left': '5px solid #e67e22', 'margin-bottom': '20px'}

    return html_content, info_content, info_card_style

@app.callback (
    Output('annotation-card', 'children'),
    Input('strategy-dropdown', 'value')
)
def update_annotation_text(selected_strategy):
    default_text = "Please select a strategy to see the pattern."
    return STRATEGY_DESCRIPTIONS.get(selected_strategy, default_text)

@app.callback(
    [Output({'type': 'param-slider', 'index': 'sigma'}, 'max'),
     Output({'type': 'param-slider', 'index': 'sigma'}, 'value'),
     Output({'type': 'param-slider', 'index': 'sigma'}, 'marks')],
    [Input({'type': 'param-slider', 'index': 'w'}, 'value')],
    [State({'type': 'param-slider', 'index': 'sigma'}, 'value')],
)
def dynamic_sigma_constraint(w_value, current_sigma):
    if w_value is None:
        w_value = 0

    new_max = round(max(w_value / 3.0, 0.01), 2)
    new_max = round(new_max, 3)

    if current_sigma and current_sigma > new_max:
        new_value = new_max
    else:
        new_value = current_sigma
        
    new_marks = {0: '0', new_max: f'{new_max:.2f}'}

    return new_max, new_value, new_marks

if __name__ == '__main__':
    app.run(debug=True)
 