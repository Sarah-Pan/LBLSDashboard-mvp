import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error

from .base_utils import load_data, get_animation_settings

PARAMS = {
    'w': {
        'label': 'Weight (w)',
        'min': 0.0, 'max': 5.0, 'step': 0.1, 'value': 1
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

    X, y = load_data()
    
    current_y = y.copy()
    nudged_history = [current_y.copy()]
    pred_history = []
    log_messages = []
    var_history = [] 

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
            
        boost = w * np.exp(-variance_norm)
        new_y_values = np.clip(current_y + boost, 1, 100)
        
        current_y = pd.Series(new_y_values, index=y.index)
        nudged_history.append(current_y.copy())
        
        avg_var = np.mean(cv_preds_var)
        avg_boost = np.mean(boost)
        
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

    extra_data = kwargs.get('extra_data', {})
    var_history = extra_data.get('var_history', [])

    if not var_history:
        return "<div>No variance data available. Run simulation first.</div>"

    first_var = np.asarray(var_history[0]).ravel()
    sort_idx = np.argsort(first_var)
    x_axis = np.arange(len(y))
    y_sorted = np.asarray(y)[sort_idx]

    frames = []
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
            
        speed_factor = np.exp(-var_norm * 3)

        frames.append(go.Frame(
            data=[
                go.Scatter(x=x_axis, y=y_sorted),
                go.Scatter(x=x_axis, y=curr_pred),
                go.Scatter(
                    x=x_axis, 
                    y=curr_nudged,
                    marker=dict(
                        color=speed_factor, 
                        colorscale='RdYlGn',
                        cmin=0, cmax=1,
                        size=10, 
                        line=dict(width=1, color='black'),
                        colorbar=dict(title="Clarity Speed")
                    )
                )
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
    init_speed = np.exp(-v_norm_init * 3)

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
            go.Scatter(x=x_axis, y=y_sorted, mode='markers', name='Original Score', marker=dict(color='lightgrey', size=6, opacity=0.3)),
            go.Scatter(x=x_axis, y=init_pred, mode='markers', name='Prediction', marker=dict(color='blue', size=8, opacity=0.4)),
            go.Scatter(
                x=x_axis, y=init_nudged, 
                mode='markers', 
                name='Student Performance', 
                marker=dict(
                    color=init_speed, 
                    colorscale='RdYlGn', 
                    cmin=0, cmax=1,
                    size=10, 
                    line=dict(width=1, color='black'),
                    showscale=True,
                    colorbar=dict(title="Clarity Speed")
                )
            )
        ],
        layout=go.Layout(
            width=1000, height=650, autosize=True,
            title="Ambiguity Iteration 0 (Initial State)",
            xaxis=dict(title="Student Index (Sorted by variance)", range=[0, len(y)]),
            yaxis=dict(title="Score", range=[0, 105]),
            legend=dict(
                yanchor="bottom",  
                y=0.01,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            annotations=[
                dict(
                    x=0.02, y=0.12, xref="paper", yref="paper",
                    text="Deep Green = High Clarity (Fast)",
                    showarrow=False,
                    font=dict(color="forestgreen", size=14, family="Arial Black")
                ),
                dict(
                    x=0.02, y=0.07, xref="paper", yref="paper",
                    text="Deep Red = High Ambiguity (Slow)",
                    showarrow=False,
                    font=dict(color="firebrick", size=14, family="Arial Black")
                )
            ],
            **common_layout
        ),
        frames=frames
    )

    return fig.to_html(include_plotlyjs='cdn', auto_play=False, config={'displayModeBar': False})