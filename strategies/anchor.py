import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

from .base_utils import load_data, get_animation_settings, generate_noise, cal_peer_force, generate_social_network

PARAMS = {
     'sigma': {
        'label': r'Noise Level ($\sigma$)', 
        'min': 0.0, 
        'max': 1.7,
        'step': 0.01, 
        'value': 0.0
    },
    'w': {
        'label': 'Weight ($w_{1}$)',
        'min': 0.0, 'max': 1.0, 'step': 0.1, 'value': 0.1
    },
    'peer_weight': {
        'label': 'Peer Effect Weight ($w_{2}$)',
        'min': -0.5,
        'max': 0.5,   
        'step': 0.01,
        'value': 0.05 
    }
    
}

def run_simulation(n_rounds=5, n_splits=5, progress_callback=None, w=0.1, **kwargs):
    """
    Run iterative nudging simulation based on Anchoring Effect.
    Formula: y_{t+1} = y_t + w_1 * (y_hat_t - y_t) + w_2 * (y_hat_t - y_t)
    
    Parameters:
    - w_1 (float): Weight of the nudging effect (how much they slack off).
    - w_2 (float): Peer force weight
    """

    X, y, class_series = load_data()

    current_y = y.copy()
    pred_history = []
    nudged_history = []
    log_messages = []
    raw_sigma = kwargs.get('sigma', 0.0)
    max_allowed_sigma = w / 3.0
    actual_sigma = min(raw_sigma, max_allowed_sigma)
    adj_matrix = generate_social_network(class_series)
    
    # record initial y 
    nudged_history.append(current_y.copy())

    for t in range(n_rounds):
        # 1. prediction model
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_preds = cross_val_predict(rf, X, current_y, cv=kf)
        pred_history.append(all_preds.copy())
        
        # calculate RMSE
        mse = mean_squared_error(current_y, all_preds)
        mean_rmse = np.sqrt(mse)

        # Anchoring Effect Nudging
        # Formula: y_{t+1} = y_t + w * (y_hat_t - y_t)
        gap = all_preds - current_y

        # anchoring effect
        reaction = w * gap

        noise = generate_noise(len(y), actual_sigma)

        # peer force
        peer_weight = kwargs.get('peer_weight', 0.05)
        peer_force = cal_peer_force(current_y, all_preds, adj_matrix, peer_weight)

        # nudged y
        new_y_values = current_y + reaction + peer_force + noise

        new_y_values = np.clip(new_y_values, 1, 100)
        
        # update current_y
        current_y = pd.Series(new_y_values, index=y.index)
        nudged_history.append(current_y.copy())

        msg = (f"[Round {t+1}] CV RMSE={mean_rmse:.4f}, "
               f"Mean y={current_y.mean():.3f}")
        print(msg)
        log_messages.append(msg)

        if progress_callback:
                # tuple: (current_round, total_rounds, newest_log)
                progress_callback((str(t + 1), str(n_rounds), msg))
        

    return y, nudged_history, pred_history, log_messages, {}

def generate_visualization(y, nudged_history, pred_history, **kwargs):
    peer_weight = kwargs.get('peer_weight', 0)
    first_pred = np.asarray(pred_history[0]).ravel()

    tick_vals = []
    tick_text = []
    class_boundaries = []

     # setup data
    _, _, class_series = load_data()

    if  peer_weight != 0:
        df_temp = pd.DataFrame({
            'class': class_series.values,
            'pred': first_pred,
            'original_idx': np.arange(len(y))
        })

        # sorted by class
        df_sorted = df_temp.sort_values(by=['class', 'pred'])
        sort_idx = df_sorted['original_idx'].values
        sorted_classes = df_sorted['class'].values
        # class boundaries
        class_boundaries = np.where(sorted_classes[:-1] != sorted_classes[1:])[0]

        start_idx = 0
        for boundary in class_boundaries:
             end_idx = boundary
             midpoint = (start_idx + end_idx) / 2
             class_name = sorted_classes[start_idx]
             tick_vals.append(midpoint)
             tick_text.append(str(class_name))
             start_idx = boundary + 1
        
        # last class
        last_end_idx = len(y) - 1
        last_midpoint = (start_idx + last_end_idx) / 2
        last_class_name = sorted_classes[start_idx]
        tick_vals.append(last_midpoint)
        tick_text.append(str(last_class_name))
        x_axis_title = "Student Index (Grouped by Class)"
        
    else: # peer_weight == 0 
        sort_idx = np.argsort(first_pred)
        x_axis_title = "Student Index (Sorted by Prediction)"
        
    shapes = []
    for boundary in class_boundaries:
        shapes.append(dict(
            type="line",
            x0=boundary + 0.5, y0=0,
            x1=boundary + 0.5, y1=105,
            line=dict(color="rgba(0,0,0,0.2)", width=1, dash="dash")
        ))
    
    # Frames
    frames = []
    x_axis = np.arange(len(y))
    y_sorted = np.asarray(y)[sort_idx]
    n_preds = len(pred_history)
    total_steps = n_preds + 1

    for i in range(total_steps):
        curr_nudged = np.asarray(nudged_history[i]).ravel()[sort_idx]

        if i < n_preds:
            curr_pred = np.asarray(pred_history[i]).ravel()[sort_idx]
            title_text = f"Anchoring Effect: Iteration {i}"
        else:
            curr_pred = np.asarray(pred_history[-1]).ravel()[sort_idx]
            title_text = f"Anchoring Effect: Final Result (After {n_preds} Iterations)"


        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_axis, y=y_sorted, mode='markers', marker=dict(color='grey', size=8, opacity=0.8)),
                go.Scatter(x=x_axis, y=curr_pred, mode='markers', marker=dict(color='blue', size=8, opacity=0.8)),
                go.Scatter(x=x_axis, y=curr_nudged, mode='markers', marker=dict(color='red', size=8, opacity=0.8))
            ],
            name=str(i),
            layout=go.Layout(
                 title=title_text,)
        ))

        # Initial Data
    initial_pred = np.asarray(pred_history[0]).ravel()[sort_idx]
    initial_nudged = np.asarray(nudged_history[0]).ravel()[sort_idx]


    custom_labels = []
    for i in range(total_steps):
        custom_labels.append("Start" if i == 0 else (f"Final" if i == total_steps-1 else f"Iteration {i}"))

    common_layout = get_animation_settings(total_steps=total_steps, duration=1000, transition=800, slider_labels=custom_labels)

    fig = go.Figure(
        data=[
            go.Scatter(x=x_axis, y=y_sorted, mode='markers', name='Ground Truth', marker=dict(color='grey', size=6, opacity=0.3)),
            go.Scatter(x=x_axis, y=initial_pred, mode='markers', name='Prediction', marker=dict(color='blue', size=8, opacity=0.6)),
            go.Scatter(x=x_axis, y=initial_nudged, mode='markers', name='Nudged Performance', marker=dict(color='red', size=8, opacity=0.8))
        ],
        layout=go.Layout(
            width=1000,
            height=600,
            autosize=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=80, b=40),
            title="Anchoring Effect Iteration 0 (Initial State)",
            xaxis=dict(title=x_axis_title, 
                       range=[0, len(y)],
                       tickvals=tick_vals if tick_vals else None,
                       ticktext=tick_text if tick_text else None),
            yaxis=dict(title="Score", range=[0, 105]),
            shapes=shapes,

            **common_layout
        ),
        frames=frames
    )
    
    return fig.to_html(
        include_plotlyjs='cdn',
        auto_play=False, 
        config={'displayModeBar': False}
    )
