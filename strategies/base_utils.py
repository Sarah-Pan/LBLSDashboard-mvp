import numpy as np
import pandas as pd

def load_data(file_path="merged_data.csv", target_column="score"):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def get_animation_settings(total_steps, duration=1000, transition=800, slider_labels=None):
    
    if slider_labels is None:
        slider_labels = [str(k + 1) for k in range(total_steps)]
        prefix = "Iteration:"
    else:
        prefix = ""
    
    return dict(
        updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'direction': 'left',
                'x': 0.1,          
                'y': -0.15,          
                'pad': {'r': 10, 't': 10},
                'buttons': [{
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': duration, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': transition, 'easing': 'quadratic-in-out'}
                    }]
                }]
            }],

            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': prefix,
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': transition, 'easing': 'cubic-in-out'},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [
                    {
                        'args': [[str(k)], {
                            'frame': {'duration': duration, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': transition}
                        }],
                        'label': slider_labels[k],
                        'method': 'animate'
                    } for k in range(total_steps)
                ]
            }],


    )