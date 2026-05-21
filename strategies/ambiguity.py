import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
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
        'label': 'Weight ($w$)',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'value': 0.1
    },
    'peer_weight': {
        'label': 'Peer Effect Weight ($w_{2}$)',
        'min': -0.5,
        'max': 0.5,   
        'step': 0.01,
        'value': 0.05 
    }
}

def run_simulation(n_rounds=5, n_splits=5, progress_callback=None, w=5, **kwargs):
    """
    Run iterative nudging simulation based on Ambiguity Aversion using MANUAL CV LOOP.
    
    Why Manual CV? 
    Standard `cross_val_predict` discards the individual trees, making it impossible 
    to calculate sigma (variance). We must iterate folds manually to capture 
    prediction variance for the validation set.
    """

    X, y, class_series = load_data()
    
    current_y = y.copy()
    nudged_history = [current_y.copy()]
    pred_history = []
    log_messages = []
    var_history = []
    raw_sigma = kwargs.get('sigma', 0.0)
    max_allowed_sigma = w / 3.0
    actual_sigma = min(raw_sigma, max_allowed_sigma)
    adj_matrix = generate_social_network(class_series)

    # we need manually implement CV to get variance from individual trees
    for t in range(n_rounds):
        cv_preds_mean = np.zeros(len(y))
        cv_preds_var = np.zeros(len(y))
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + t)
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = current_y.iloc[train_idx]
            
            rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
            rf.fit(X_train, y_train)
            
            tree_preds = np.array([tree.predict(X_val.values) for tree in rf.estimators_])
            
            cv_preds_mean[val_idx] = np.mean(tree_preds, axis=0)
            cv_preds_var[val_idx] = np.var(tree_preds, axis=0)

        pred_history.append(cv_preds_mean.copy())
        var_history.append(cv_preds_var.copy())
        
        # calculate RMSE
        mse = mean_squared_error(current_y, cv_preds_mean)
        mean_rmse = np.sqrt(mse)

        # Ambiguity Nudging
        # Formula: y_{t+1} = y_t + w * exp(- normalized variance)
        # Normalize variance to [0,1]
        v_min, v_max = cv_preds_var.min(), cv_preds_var.max()
        if v_max - v_min > 0:
            variance_norm = (cv_preds_var - v_min) / (v_max - v_min)
        else:
            variance_norm = np.zeros_like(cv_preds_var)
        

        # boost = w * 0.05*np.exp(-0.05*variance_norm)
        eps = 1e-9
        boost = w / np.maximum(variance_norm, eps)

        noise = generate_noise(len(y), actual_sigma)

        # peer force
        peer_weight = kwargs.get('peer_weight', 0.05)
        peer_force = cal_peer_force(current_y, cv_preds_mean, adj_matrix, peer_weight)

        # nudged y
        new_y_values = np.clip(current_y + boost + peer_force + noise, 1, 100)
        
        current_y = pd.Series(new_y_values, index=y.index)
        nudged_history.append(current_y.copy())
        
        msg = (f"[Round {t+1}] RMSE={mean_rmse:.2f}, Mean y={current_y.mean():.2f}\n")
        
        log_messages.append(msg)
        print(msg)

        if progress_callback:
            progress_callback((str(t + 1), str(n_rounds), msg))

    experiment_info = {
        "var_history": var_history
    }

    return y, nudged_history, pred_history, log_messages, experiment_info

def generate_visualization(y, nudged_history, pred_history, **kwargs):
    peer_weight = kwargs.get('peer_weight', 0)
    extra_data = kwargs.get('extra_data', {})
    var_history = extra_data.get('var_history', [])
    _, _, class_series = load_data()

    if not var_history:
        return "<div>No variance data available. Run simulation first.</div>"

    first_var = np.asarray(var_history[0]).ravel()
    sort_idx = np.argsort(first_var)
    tick_vals = []
    tick_text = []
    class_boundaries = []

    if peer_weight != 0:

        df_temp = pd.DataFrame({
            'class': class_series.values,
            'var': first_var,
            'original_idx': np.arange(len(y))
        })

        # sorted by class
        df_sorted = df_temp.sort_values(by=['class', 'var'])
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
        sort_idx = np.argsort(first_var)
        x_axis_title = "Student Index (Sorted by Variance)"

    shapes = []
    for boundary in class_boundaries:
        shapes.append(dict(
            type="line", x0=boundary + 0.5, y0=0, x1=boundary + 0.5, y1=105,
            line=dict(color="rgba(0,0,0,0.2)", width=1, dash="dash")
        ))
    
    frames = []
    x_axis = np.arange(len(y))
    y_sorted = np.asarray(y)[sort_idx]
    n_preds = len(pred_history)
    total_steps = n_preds + 1

    for i in range(total_steps):
        if i < n_preds:
            idx = i
            title_text = f"Ambiguity Aversion: Iteration {i}"
        else:
            idx = -1 
            title_text = f"Ambiguity Aversion: Final Result (After {n_preds} Iterations)"

        curr_pred = np.asarray(pred_history[idx]).ravel()[sort_idx]
        curr_nudged = np.asarray(nudged_history[i]).ravel()[sort_idx]
        
        v_idx = idx if idx != -1 else len(var_history) - 1
        curr_var = np.asarray(var_history[v_idx]).ravel()[sort_idx]
        
        v_min, v_max = curr_var.min(), curr_var.max()
        if v_max - v_min > 0:
            var_norm = (curr_var - v_min) / (v_max - v_min)
        else:
            var_norm = np.zeros_like(curr_var)
            
        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_axis, y=y_sorted),
                go.Scatter(x=x_axis, y=curr_pred,
                    marker=dict(
                        color=var_norm,
                        reversescale=True, 
                        colorscale='Blues',
                        cmin=0, cmax=1,
                        size=8, 
                        line=dict(width=1, color='black'),
                        colorbar=dict(
                                      orientation='h',
                                      y=-0.1,
                                      x=0.5,
                                      xanchor='center', 
                                      yanchor='top',
                                      ticks="",
                                      tickvals=[0, 1],
                                      ticktext=['Low Variance', 'High Variance'],
                                      title="",
                                      thickness=15,
                                      len=0.6))),
                go.Scatter(x=x_axis, y=curr_nudged)
            ],
            name=str(i),
            layout=go.Layout(title=title_text)
        ))

    init_pred = np.asarray(pred_history[0]).ravel()[sort_idx]
    init_nudged = np.asarray(nudged_history[0]).ravel()[sort_idx]
    
    init_var = np.asarray(var_history[0]).ravel()[sort_idx]
    v_min, v_max = init_var.min(), init_var.max()
    if v_max - v_min > 0:
        v_norm_init = (init_var - v_min) / (v_max - v_min)
    else:
        v_norm_init = np.zeros_like(init_var)

    custom_labels = []
    for i in range(total_steps):
        custom_labels.append("Start" if i == 0 else (f"Final" if i == total_steps-1 else f"Iteration {i}"))   

    common_layout = get_animation_settings(total_steps=total_steps, duration=800, slider_labels=custom_labels)

    fig = go.Figure(
        data=[
            go.Scatter(x=x_axis, y=y_sorted, mode='markers', name='Original Score', marker=dict(color='lightgrey', size=8, opacity=0.8)),
            go.Scatter(x=x_axis, y=init_pred, mode='markers', name='Prediction', marker=dict(
                    color=v_norm_init, 
                    reversescale=True,
                    colorscale='Blues',
                    cmin=0, cmax=1,
                    size=8, 
                    line=dict(width=1, color='black'),
                    showscale=True,
                    colorbar=dict(
                                      orientation='h',
                                      y=-0.125,
                                      x=0.5,
                                      xanchor='center', 
                                      yanchor='top',
                                      ticks="",
                                      tickvals=[0, 1],
                                      ticktext=['Low Variance', 'High Variance'],
                                      title="",
                                      thickness=15,
                                      len=0.6)
                )),
            go.Scatter(x=x_axis, y=init_nudged, mode='markers', name='Student Performance', marker=dict(color='red', size=8, opacity=1)),
        ],
        layout=go.Layout(
            width=1000, height=650, autosize=True,
            title="Ambiguity Iteration 0 (Initial State)",
            xaxis=dict(title=x_axis_title, 
                       range=[0, len(y)],
                       tickvals=tick_vals if tick_vals else None,
                       ticktext=tick_text if tick_text else None),
            yaxis=dict(title="Score", range=[0, 105]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                yanchor="bottom",  
                y=0.01,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            shapes=shapes,
            **common_layout
        ),
        frames=frames
    )

    return fig.to_html(include_plotlyjs='cdn', auto_play=False, config={'displayModeBar': False})