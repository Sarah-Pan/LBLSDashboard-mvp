import numpy as np
import pandas as pd

def load_data(file_path="merged_class_data.csv", target_column="score", extra_drop=None):
    df = pd.read_csv(file_path)
    if extra_drop:
        df = df.drop(columns=extra_drop)
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
                'pad': {'r': 10, 't': 50},
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
                'y': -0.1,
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

def generate_social_network(class_series):
    classes = class_series.values if isinstance(class_series, pd.Series) else np.array(class_series)
    adj = (classes[:, None] == classes[None, :]).astype(int)
    np.fill_diagonal(adj, 0)

    return adj

def cal_peer_force(current_y, prediction, adj_matrix, w_peer):
    if w_peer == 0:
        return np.zeros(len(current_y))

    y_values = current_y.values if isinstance(current_y, pd.Series) else current_y
    pred_values = prediction.values if isinstance(prediction, pd.Series) else prediction

    friend_counts = adj_matrix.sum(axis=1)
    safe_counts = np.where(friend_counts == 0, 1, friend_counts) # if friend_counts == 0, make it 1 prevent DivisionZero

    neighbour_sum = np.dot(adj_matrix, pred_values)
    neighbour_avg = neighbour_sum / safe_counts

    no_friend_mask = (friend_counts == 0)
    neighbour_avg[no_friend_mask] = y_values[no_friend_mask] # gap = 0, their score stay the same

    gap = neighbour_avg - y_values
    peer_force = gap * w_peer

    return peer_force

def generate_noise(length, sigma):
    if sigma <= 0:
        return 0
    
    return np.random.normal(loc=0, scale=sigma, size=length)