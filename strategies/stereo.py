import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

from .base_utils import load_data, get_animation_settings, generate_noise


CONSTRUCTS_DB = {
    # === SILL (Language Learning Strategies) ===
    # correlation r: -0.004 (Negative) -> Risk
    'memory': {
        'name': 'Memory Strategies',
        'columns': ['s_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8'],
        'higher_is_risk': True
    },
    # correlation r: 0.089 (Positive) -> Safe
    'cognitive': {
        'name': 'Cognitive Strategies',
        'columns': ['s_9', 's_10', 's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19', 's_20', 's_21'],
        'higher_is_risk': False
    },
    # correlation r: 0.062 (Positive) -> Safe
    'compensation': {
        'name': 'Compensation Strategies',
        'columns': ['s_22', 's_23', 's_24', 's_25', 's_26', 's_27'],
        'higher_is_risk': False
    },
    # correlation r: -0.046 (Negative) -> Risk
    'metacognitive': {
        'name': 'Metacognitive Strategies',
        'columns': ['s_28', 's_29', 's_30', 's_31', 's_32', 's_33', 's_34', 's_35', 's_36'],
        'higher_is_risk': True
    },
    # correlation r: -0.116 (Negative) -> Risk
    'affective': {
        'name': 'Affective Strategies',
        'columns': ['s_37', 's_38', 's_39', 's_40', 's_41', 's_42'],
        'higher_is_risk': True
    },
    # correlation r: -0.096 (Negative) -> Risk
    'social': {
        'name': 'Social Strategies',
        'columns': ['s_43', 's_44', 's_45', 's_46', 's_47', 's_48'],
        'higher_is_risk': True
    },

    # === MSLQ (Motivated Strategies for Learning) ===
    # correlation r: 0.122 (Positive) -> Safe
    'rehearsal': {
        'name': 'Rehearsal',
        'columns': ['srl_s_1', 'srl_s_2', 'srl_s_3', 'srl_s_4'],
        'higher_is_risk': False
    },
    # correlation r: 0.253 (Positive) -> Safe
    'elaboration': {
        'name': 'Elaboration',
        'columns': ['srl_s_5', 'srl_s_6', 'srl_s_7', 'srl_s_8', 'srl_s_9', 'srl_s_10'],
        'higher_is_risk': False
    },
    # correlation r: 0.118 (Positive) -> Safe
    'organization': {
        'name': 'Organization',
        'columns': ['srl_s_11', 'srl_s_12', 'srl_s_13', 'srl_s_14'],
        'higher_is_risk': False
    },
    # correlation r: 0.139 (Positive) -> Safe
    'critical_thinking': {
        'name': 'Critical Thinking',
        'columns': ['srl_s_15', 'srl_s_16', 'srl_s_17', 'srl_s_18', 'srl_s_19'],
        'higher_is_risk': False
    },
    # correlation r: 0.188 (Positive) -> Safe
    'metacognitive_self_regulation': {
        'name': 'Metacognitive Self-Regulation',
        'columns': ['srl_s_20','srl_s_21','srl_s_22','srl_s_23','srl_s_24','srl_s_25','srl_s_26','srl_s_27','srl_s_28','srl_s_29','srl_s_30','srl_s_31','srl_s_32','srl_s_33','srl_s_34','srl_s_35','srl_s_36','srl_s_37','srl_s_38','srl_s_39'],
        'higher_is_risk': False
    },
    # correlation r: 0.034 (Positive) -> Safe
    'effort_regulation': {
        'name': 'Effort Regulation',
        'columns': ['srl_s_40', 'srl_s_41', 'srl_s_42', 'srl_s_43'],
        'higher_is_risk': False
    },
    # correlation r: 0.107 (Positive) -> Safe
    'peer_learning': {
        'name': 'Peer Learning',
        'columns': ['srl_s_44', 'srl_s_45', 'srl_s_46'],
        'higher_is_risk': False
    },
    # correlation r: 0.054 (Positive) -> Safe
    'help_seeking': {
        'name': 'Help Seeking',
        'columns': ['srl_s_47','srl_s_48', 'srl_s_49', 'srl_s_50'],
        'higher_is_risk': False
    },
    # correlation r: 0.140 (Positive) -> Safe
    'intrinsic_goal': {
        'name': 'Intrinsic Goal Orientation',
        'columns': ['srl_m_1', 'srl_m_2', 'srl_m_3', 'srl_m_4'],
        'higher_is_risk': False
    },
    # correlation r: 0.172 (Positive) -> Safe
    'extrinsic_goal': {
        'name': 'Extrinsic Goal Orientation',
        'columns': ['srl_m_5', 'srl_m_6', 'srl_m_7', 'srl_m_8'],
        'higher_is_risk': False
    },
    # correlation r: 0.254 (Positive) -> Safe
    'task_value': {
        'name': 'Task Value',
        'columns': ['srl_m_9', 'srl_m_10', 'srl_m_11', 'srl_m_12', 'srl_m_13', 'srl_m_14'],
        'higher_is_risk': False
    },
    # correlation r: 0.222 (Positive) -> Safe
    'control_beliefs': {
        'name': 'Control Beliefs',
        'columns': ['srl_m_15', 'srl_m_16', 'srl_m_17', 'srl_m_18'],
        'higher_is_risk': False
    },
    # correlation r: 0.401 (Positive) -> Safe
    'self_efficacy': {
        'name': 'Self-Efficacy',
        'columns': ['srl_m_19','srl_m_20','srl_m_21','srl_m_22','srl_m_23','srl_m_24','srl_m_25','srl_m_26'],
        'higher_is_risk': False
    },
    # correlation r: -0.230 (Negative) -> Risk
    'test_anxiety': {
        'name': 'Test Anxiety',
        'columns': ['srl_m_27', 'srl_m_28', 'srl_m_29', 'srl_m_30', 'srl_m_31'],
        'higher_is_risk': True
    }
}

PARAMS = {'sigma': {
        'label': r'Noise Level ($\sigma$)', 
        'min': 0.0, 
        'max': 1.7,
        'step': 0.01, 
        'value': 0.0
    },
    'construct_selector': {
        'type': 'dropdown',           
        'label': 'Select Risk Factor',
        'value': 'test_anxiety',       
        'options': [              
            {'label': info['name'], 'value': key} 
            for key, info in CONSTRUCTS_DB.items()
        ]
    },
    'w': {
        'label': 'Weight ($w$)',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'value': 0.5
    },
    'threshold':{
        'label': 'Threshold (Top/Bottom %)',
        'min': 10, 'max': 90, 'step': 10, 'value': 30,
    }
}

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

    X, y = load_data(extra_drop=['class'])
    # features = [c for c in df.columns if c not in ['class', 'score']]
    # X = df[features]

    # Get construct configuration
    construct_key = kwargs.get('construct_selector', 'test_anxiety')
    config = CONSTRUCTS_DB.get(construct_key, CONSTRUCTS_DB['test_anxiety'])

    # calculate construct scores
    construct_scores = cal_construct_score(X, config)

    # is risky?
    x_bias, cutoff = is_risk_group(construct_scores, threshold, config['higher_is_risk'])

    current_y = y.copy()
    pred_history = []
    nudged_history = []
    log_messages = []
    raw_sigma = kwargs.get('sigma', 0.0)
    max_allowed_sigma = w / 3.0
    actual_sigma = min(raw_sigma, max_allowed_sigma)

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

        noise = generate_noise(len(y), actual_sigma)

        # nudged y
        new_y_values = current_y - decay + noise

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
    construct_key = kwargs.get('construct_selector', 'test_anxiety')
    config = CONSTRUCTS_DB.get(construct_key, CONSTRUCTS_DB['test_anxiety'])
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

        if i < n_preds:
            curr_pred = np.asarray(pred_history[i]).ravel()[sort_idx]
            title_text = f"Stereotype Threat: Iteration {i}"
        else:
            curr_pred = np.asarray(pred_history[-1]).ravel()[sort_idx]
            title_text = f"Stereotype Threat: Final Result (After {n_preds} Iterations)"

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
                title=title_text)
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