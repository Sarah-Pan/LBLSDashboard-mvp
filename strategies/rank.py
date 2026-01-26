import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

from .base_utils import load_data, get_animation_settings

PARAMS = {
    'use_noise': {
        'type': 'radio',
        'label': 'Random Variation (Noise)',
        'value': False, 
        'options': [
            {'label': 'Off', 'value': False},
            {'label': 'On', 'value': True}
        ]
    },
    'w': {
        'label': r'Weight ($w$)',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'value': 0.01
    },
    'threshold':{
        'label': r'Threshold Rank ($\tau$)',
        'min': 100, 'max': 200, 'step': 10, 'value': 100,
    }
}

def run_simulation(n_rounds=5, n_splits=5, w=0.5, threshold=100, progress_callback=None, **kwargs):
    """
    Run iterative nudging simulation based on rank anxiety.
    Formula: y_{t+1} = y_t + w * [rank_hat - threshold]_+
    
    Parameters:
    - threshold_rank: The rank cutoff (tau).
    - w (float): Weight of the nudging effect (how much they slack off).
    """

    X, y = load_data()

    current_y = y.copy()
    pred_history = []
    nudged_history = []
    log_messages = []
    use_noise = kwargs.get('use_noise', False)
    
    # record initial y 
    nudged_history.append(current_y.copy())

    for t in range(n_rounds):
        # prediction model
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_preds = cross_val_predict(rf, X, current_y, cv=kf)
        pred_history.append(all_preds.copy())
        
        # calculate RMSE
        mse = mean_squared_error(current_y, all_preds)
        mean_rmse = np.sqrt(mse)

        pred_series = pd.Series(all_preds, index=y.index)
        predicted_ranks = pred_series.rank(ascending=False, method='min')

        # Rank Anxiety
        # Formula: y_{t+1} = y_t + w * [rank_hat - threshold]_+
        # [rank_hat - threshold]_+ : calculate the part below the threshold

        gap = np.maximum(0, predicted_ranks - threshold)
        
        # Boost amount
        original_boost = w * gap

        if use_noise:
            noisy_boost = np.random.normal(loc=original_boost, scale=0.5, size=len(y))
            mask_effect = (original_boost > 0)
            boost = np.zeros_like(original_boost)
            boost[mask_effect] = np.maximum(0, noisy_boost[mask_effect])
        else:
            boost = original_boost


        # nudged y
        new_y_values = current_y + boost
        new_y_values = np.clip(new_y_values, 1, 100)
        
        # update current_y
        current_y = pd.Series(new_y_values, index=y.index)
        nudged_history.append(current_y.copy())

        msg = (f"[Round {t+1}] CV RMSE={mean_rmse:.4f}, "
               f"Mean y={current_y.mean():.3f}")
        print(msg)
        log_messages.append(msg)

        if progress_callback:# tuple: (current_round, total_rounds, newest_log)
                progress_callback((str(t + 1), str(n_rounds), msg))

    return y, nudged_history, pred_history, log_messages, {}

def generate_visualization(y, nudged_history, pred_history, **kwargs):

    threshold_rank = int(kwargs.get('threshold', 100))

    # setup data
    first_pred = np.asarray(pred_history[0]).ravel()
    sort_idx = np.argsort(first_pred)
    x_axis = np.arange(len(y))
    y_sorted = np.asarray(y)[sort_idx]

    # Frames
    frames = []
    n_preds = len(pred_history)
    total_steps = n_preds + 1
    frame_annotations = []

    for i in range(total_steps):
        curr_nudged = np.asarray(nudged_history[i]).ravel()[sort_idx] 
        if i < n_preds:
            curr_pred = np.asarray(pred_history[i]).ravel()[sort_idx]
            title_text = f"Rank Anxiety: Iteration {i}"
        else:
            curr_pred = np.asarray(pred_history[-1]).ravel()[sort_idx]
            title_text = f"Rank Anxiety: Final Result (After {n_preds} Iterations)"

            frame_annotations = [
                dict(
                    x=0.5, y=0.05,
                    yanchor="bottom",
                    xref="paper", yref="paper",
                    text="<b>Pattern Detected:</b><br>"+
                    "Predicted lower-rank students (red dots) rise rapidly to catch up, pulling their blue predictions upward.<br>" + 
                    "In contrast, Predicted top 100 students (yellow dots) stay still and are eventually overtaken by the rising group.",
                    showarrow=False,
                    font=dict(size=16, color="darkblue")
                )
            ]

        ranks = pd.Series(curr_pred).rank(ascending=False, method='min').values
        mask_safe = (ranks <= threshold_rank)
        mask_danger = ~mask_safe

        nudged_safe = curr_nudged[mask_safe]
        nudged_danger = curr_nudged[mask_danger]

        x_safe = x_axis[mask_safe]
        x_danger = x_axis[mask_danger]

        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_axis, y=y_sorted),
                go.Scatter(x=x_axis, y=curr_pred),
                go.Scatter(x=x_safe, y=nudged_safe),
                go.Scatter(x=x_danger, y=nudged_danger)
            ],
            name=str(i),
            layout=go.Layout(
                title=title_text,
                annotations=frame_annotations)
        ))

    initial_pred = np.asarray(pred_history[0]).ravel()[sort_idx]
    initial_nudged = np.asarray(nudged_history[0]).ravel()[sort_idx]
    
    ranks_init = pd.Series(initial_pred).rank(ascending=False, method='min').values
    mask_safe_init = (ranks_init <= threshold_rank)
    mask_danger_init = ~mask_safe_init


    x_safe_init = x_axis[mask_safe_init]
    x_danger_init = x_axis[mask_danger_init]
    nudged_safe_init = initial_nudged[mask_safe_init]
    nudged_danger_init = initial_nudged[mask_danger_init]

    custom_labels = []
    for i in range(total_steps):
        if i == 0:
            custom_labels.append("Start")
        elif i == total_steps - 1:
            custom_labels.append(f"Final(Iteration {i})")
        else:
            custom_labels.append(f"Iteration {i}")

    common_layout = get_animation_settings(total_steps=total_steps, duration=800, slider_labels=custom_labels)


    fig = go.Figure(
        data=[
            go.Scatter(x=x_axis, y=y_sorted, mode='markers', name='Original', marker=dict(color='lightgrey', size=6, opacity=0.8)),
            go.Scatter(x=x_axis, y=initial_pred, mode='markers', name='Prediction', marker=dict(color='blue', size=8, opacity=0.3)),
            go.Scatter(x=x_safe_init, y=nudged_safe_init, mode='markers', name=f'Predicted Top {threshold_rank} (Safe)', marker=dict(    color='gold',     size=10,     opacity=1.0,    line=dict(width=1, color='yellow'))),
            go.Scatter(x=x_danger_init, y=nudged_danger_init, mode='markers', name='Predicted Behind (Boosting)', marker=dict(color='red', size=8, opacity=0.9))
        ],
        layout=go.Layout(
            width=1000, height=650, autosize=True,
            title="Rank Anxiety Iteration 0 (Initial State)",
            xaxis=dict(title="Student Index (Sorted by Prediction)", range=[0, len(y)]),
            yaxis=dict(title="Score", range=[0, 105]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            **common_layout
        ),
        frames=frames
    )

    return fig.to_html(include_plotlyjs='cdn', 
                       auto_play=False, 
                       config={'displayModeBar': False})
     
