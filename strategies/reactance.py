import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

from .base_utils import load_data, get_animation_settings, generate_noise

PARAMS = {
     'sigma': {
        'label': r'Noise Level ($\sigma$)', 
        'min': 0.0, 
        'max': 1.7,
        'step': 0.01, 
        'value': 0.0
    },
    'w': {
        'label': 'Weight ($w$)',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'value': 0.5
    }
}

def run_simulation(n_rounds=5, n_splits=5, progress_callback=None, w=0.5, **kwargs):
    """
    Run iterative nudging simulation based on Psychological Reactance.
    Formula: y_{t+1} = y_t - w * (y_hat_t - y_t)
    
    Parameters:
    - w (float): Weight of the nudging effect (how much they slack off).
    """

    X, y = load_data(extra_drop=['class'])
    # features = [c for c in df.columns if c not in ['class', 'score']]
    # X = df[features]

    current_y = y.copy()
    pred_history = []
    nudged_history = []
    log_messages = []
    raw_sigma = kwargs.get('sigma', 0.0)
    max_allowed_sigma = w / 3.0
    actual_sigma = min(raw_sigma, max_allowed_sigma)
    
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

        # 2. Psychological Reactance
        # Formula: y_{t+1} = y_t - w * (y_hat_t - y_t)
        gap = all_preds - current_y

        # reactance effect
        reaction = w * gap

        noise = generate_noise(len(y), actual_sigma)
        
        # nudged y 
        new_y_values = current_y - reaction + noise

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
     # setup data
    first_pred = np.asarray(pred_history[0]).ravel()
    sort_idx = np.argsort(first_pred)
    x_axis = np.arange(len(y))
    y_sorted = np.asarray(y)[sort_idx]
    
    frames = []
    n_preds = len(pred_history)
    total_steps = n_preds + 1

    for i in range(total_steps):
        curr_nudged = np.asarray(nudged_history[i]).ravel()[sort_idx]

        if i < n_preds:
            curr_pred = np.asarray(pred_history[i]).ravel()[sort_idx]
            title_text = f"Psychological Reactance: Iteration {i}"
        else:
            curr_pred = np.asarray(pred_history[-1]).ravel()[sort_idx]
            title_text = f"Psychological Reactance: Final Result (After {n_preds} Iterations)"

        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_axis, y=y_sorted, mode='markers', marker=dict(color='grey', size=8, opacity=0.8)),
                go.Scatter(x=x_axis, y=curr_pred, mode='markers', marker=dict(color='blue', size=8, opacity=0.8)),
                go.Scatter(x=x_axis, y=curr_nudged, mode='markers', marker=dict(color='red', size=8, opacity=0.8))
            ],
            name=str(i),
            layout=go.Layout(
                 title=title_text)
        ))

        # Initial Data
    initial_pred = np.asarray(pred_history[0]).ravel()[sort_idx]
    initial_nudged = np.asarray(nudged_history[0]).ravel()[sort_idx]


    custom_labels = []
    for i in range(total_steps):
        if i  == 0:
              custom_labels.append("Start")
        elif i == total_steps -1:
            custom_labels.append(f"Final(Iteration {i})")
        else:
              custom_labels.append(f"Iteration {i}")

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
            title="Psychological Reactance Iteration 0 (Initial State)",
            xaxis=dict(title="Student Index (Sorted by Prediction)", range=[0, len(y)]),
            yaxis=dict(title="Score", range=[0, 105]),

            **common_layout
        ),
        frames=frames
    )
    
    return fig.to_html(
        include_plotlyjs='cdn',
        auto_play=False, 
        config={'displayModeBar': False}
    )
