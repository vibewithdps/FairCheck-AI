import pandas as pd

def calculate_fairness(df, target_col, group_col, privileged_value, unprivileged_value):
    # Success rate for privileged group
    priv_df = df[df[group_col] == privileged_value]
    priv_success = priv_df[target_col].mean()
    
    # Success rate for unprivileged group
    unpriv_df = df[df[group_col] == unprivileged_value]
    unpriv_success = unpriv_df[target_col].mean()
    
    # Disparate Impact Ratio
    di_ratio = unpriv_success / priv_success if priv_success > 0 else 1.0
    
    status = "Fair" if 0.8 <= di_ratio <= 1.25 else "Biased"
    
    return {
        "disparate_impact": round(di_ratio, 2),
        "status": status,
        "priv_success": round(priv_success * 100, 2),
        "unpriv_success": round(unpriv_success * 100, 2)
    }