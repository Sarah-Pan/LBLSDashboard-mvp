import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

from .base_utils import load_data, get_animation_settings

PARAMS = {
    'w': {
        'label': 'Weight ($w$)',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'value': 0.5
    },
    'threshold':{
        'label': 'Threshold (Top/Bottom %)',
        'min': 10, 'max': 90, 'step': 10, 'value': 30,
    }
}

CONSTRUCTS_DB ={
    'test_anxiety':{
        'name':'Test Anxiety',
        'columns':['srl_m_27', 'srl_m_28', 'srl_m_29', 'srl_m_30', 'srl_m_31'],
        'higher_is_risk': True
    }
}

ACTIVE_KEY = 'test_anxiety'

# Get current construct configuration
def get_current_config():
    return CONSTRUCTS_DB.get(ACTIVE_KEY, list(CONSTRUCTS_DB.values())[0])

def cal_construct_score(X, config):
    cols = config['columns']
    available_cols = [c for c in cols if c in X.columns]

    if not available_cols:
        raise ValueError("No valid columns found for the construct.")
    
    return X[available_cols].mean(axis=1)

def is_risk_group(scores, threshold, higher_is_risk=True):
    if higher_is_risk:
        cutoff = np.percentile(scores, 100 - threshold)
        x_bias = (scores >= cutoff).astype(int)
        return x_bias, cutoff
    else:
        cutoff = np.percentile(scores, threshold)
        x_bias = (scores <= cutoff).astype(int)
        return x_bias, cutoff

def run_simulation(n_rounds=5, n_splits=5, progress_callback=None, w=0.5, threshold=30, **kwargs):
    """
    Run iterative nudging simulation based on stereotype effect.
    Formula: y_{t+1} = y_t - w * X_bias * [y_t - y_hat]_+
    
    Parameters:
    - w (float): Weight of the nudging effect (how much they slack off).
    """

    X, y = load_data()

    # Get construct configuration
    config = get_current_config()

    # calculate construct scores
    construct_scores = cal_construct_score(X, config)

    # is risky?
    x_bias, cutoff = is_risk_group(construct_scores, threshold, config['higher_is_risk'])

    current_y = y.copy()
    pred_history = []
    nudged_history = []
    log_messages = []

    if config['higher_is_risk']:
        risk_desc = f"Top {threshold}%"
    else:
        risk_desc = f"Bottom {threshold}%"
    
    experiment_info = {
        "construct_name": config['name'],
        "risk_desc": risk_desc,
        "cutoff": cutoff
    }

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

        # 2. stereotype effect nudging
        # Formula: y_{t+1} = y_t - w * X_bias * [y_t - y_hat]_+
        gap = np.maximum(0, current_y - all_preds)

        # Slacking off amount
        decay = w * x_bias * gap

        # nudged y
        new_y_values = current_y - decay

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
        
    return y, nudged_history, pred_history, log_messages, experiment_info

def generate_visualization(y, nudged_history, pred_history, **kwargs):
    X, _ = load_data()
    config = get_current_config()
    threshold = kwargs.get('threshold', 30)
    
    construct_scores = cal_construct_score(X, config)
    x_bias, _ = is_risk_group(construct_scores, threshold, config['higher_is_risk'])
    
    first_pred = np.asarray(pred_history[0]).ravel()
    sort_idx = np.argsort(first_pred)
    x_axis = np.arange(len(y))
    y_sorted = np.asarray(y)[sort_idx]
    x_bias_sorted = np.asarray(x_bias)[sort_idx]

    mask_risk = (x_bias_sorted == 1)
    mask_safe = (x_bias_sorted == 0)

    x_risk = x_axis[mask_risk]
    x_safe = x_axis[mask_safe]

    frames = []
    n_preds = len(pred_history)
    total_steps = n_preds + 1
    
    for i in range(total_steps):
        curr_nudged = np.asarray(nudged_history[i]).ravel()[sort_idx]

        frame_annotations = []

        if i < n_preds:
            curr_pred = np.asarray(pred_history[i]).ravel()[sort_idx]
            title_text = f"Stereotype Threat: Iteration {i}"
        else:
            curr_pred = np.asarray(pred_history[-1]).ravel()[sort_idx]
            title_text = f"Stereotype Threat: Final Result (After {n_preds} Iterations)"

            frame_annotations = [
                dict(
                    x=0.5, y=0.05,
                    yanchor="bottom",
                    xref="paper", yref="paper",
                    text="<b>Pattern Detected:</b><br>The Risk Group (triangles) drops significantly below their blue prediction markers.<br>" +
                    "Meanwhile, the Safe Group (circles) remains stable or rises slightly.<br>",
                    showarrow=False,
                    font=dict(size=16, color="darkblue")
                )
            ]

        pred_risk = curr_pred[mask_risk]
        pred_safe = curr_pred[mask_safe]
        
        nudged_risk = curr_nudged[mask_risk]
        nudged_safe = curr_nudged[mask_safe]

        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_axis, y=y_sorted), 
                go.Scatter(x=x_safe, y=pred_safe),
                go.Scatter(x=x_risk, y=pred_risk),
                go.Scatter(x=x_safe, y=nudged_safe),
                go.Scatter(x=x_risk, y=nudged_risk)
            ],
            name=str(i),
            layout=go.Layout(
                title=title_text,
                annotations=frame_annotations)
        ))

    initial_pred = np.asarray(pred_history[0]).ravel()[sort_idx]
    initial_nudged = np.asarray(nudged_history[0]).ravel()[sort_idx]
    
    initial_pred_risk = initial_pred[mask_risk]
    initial_pred_safe = initial_pred[mask_safe]
    
    initial_nudged_risk = initial_nudged[mask_risk]
    initial_nudged_safe = initial_nudged[mask_safe]
    x_risk = x_axis[mask_risk]
    x_safe = x_axis[mask_safe]

    custom_labels = []
    for i in range(total_steps):
        if i  == 0:
            custom_labels.append("Start")
        elif i == total_steps -1:
            custom_labels.append(f"Final(Iteration {i})")
        else:
            custom_labels.append(f"Iteration {i}")
    common_layout = get_animation_settings(total_steps=total_steps, duration=800, slider_labels=custom_labels)

    fig = go.Figure(
        data=[
            go.Scatter(x=x_axis, y=y_sorted, mode='markers', name='Original Score', marker=dict(color='lightgrey', size=6, opacity=0.5)),
            go.Scatter(x=x_safe, y=initial_pred_safe, mode='markers', name='Prediction (Safe)', marker=dict(color='blue', size=8, opacity=0.3)),
            go.Scatter(x=x_risk, y=initial_pred_risk, mode='markers', name='Prediction (Risk)', marker=dict(color='blue', symbol='triangle-up', size=10, opacity=0.6, line=dict(width=1, color='darkblue'))),
            go.Scatter(x=x_safe, y=initial_nudged_safe, mode='markers', name='Nudged y (Safe)', marker=dict(color='orange', size=6, opacity=0.3)),
            go.Scatter(x=x_risk, y=initial_nudged_risk, mode='markers', name='Nudged y (Risk)', marker=dict(color='red', symbol='triangle-up', size=10, opacity=1.0, line=dict(width=1, color='darkred')))
        ],
        layout=go.Layout(
            width=1000, 
            height=650,
            autosize = True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=80, b=40),
            title="Stereotype Threat Iteration 0 (Initial State)",
            xaxis=dict(title="Student Index (Sorted by Prediction)", range=[0, len(y)]),
            yaxis=dict(title="Score", range=[0, 105]),

            **common_layout
        ),
        frames=frames
    )
    
    return fig.to_html(
        include_plotlyjs='cdn', 
        auto_play=False, 
        config={'displayModeBar': False})