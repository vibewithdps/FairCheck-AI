import pandas as pd
from firebase_admin import credentials, db, initialize_app, _apps
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import math  # New import for Advance-level numerical validation

def setup_firebase():
    if not _apps:
        try:
            cred = credentials.Certificate("serviceAccountKey.json")
            initialize_app(cred, {
                'databaseURL': 'https://faircheck-ffe01-default-rtdb.firebaseio.com/'
            })
            return True
        except Exception as e:
            return False
    return True

def run_fairness_audit(df, target, group_col, priv_val, unprivileged_value):
    try:
        # Convert to numeric and handle missing values to avoid 'NaN' errors
        df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)
        
        priv_rate = df[df[group_col] == priv_val][target].mean()
        unpriv_rate = df[df[group_col] == unprivileged_value][target].mean()
        
        # --- Advance Level Robustness: Handling Edge Cases (NaN/Inf) ---
        # 1. Check if priv_rate is 0 to avoid division by zero (Infinity)
        if priv_rate == 0 or math.isnan(priv_rate):
            di_ratio = 0.0
        else:
            di_ratio = unpriv_rate / priv_rate
            
        # 2. Final safety check for JSON compliance
        if math.isinf(di_ratio) or math.isnan(di_ratio):
            di_ratio = 0.0
            
        status = "✅ Fair" if 0.8 <= di_ratio <= 1.25 else "⚠️ Biased"
        
        # Advance Level Addition: Statistical Parity Difference
        stat_parity = unpriv_rate - priv_rate
        if math.isnan(stat_parity):
            stat_parity = 0.0
        
        results = {
            "score": round(float(di_ratio), 3),
            "stat_parity": round(float(stat_parity), 3),
            "status": status,
            "priv_group_success": f"{round(float(priv_rate or 0) * 100, 1)}%",
            "unpriv_group_success": f"{round(float(unpriv_rate or 0) * 100, 1)}%",
            "target_column": str(target),
            "group_audited": str(group_col)
        }
        
        if _apps:
            # Wrap Firebase call in a separate try-except to ensure UI still works
            try:
                db.reference('audit_logs').push(results)
            except Exception as fe:
                print(f"Firebase Sync Error: {fe}")
            
        return results
    except Exception as e:
        return {"error": str(e)}

# --- NEW ADVANCE LEVEL RESEARCH FUNCTIONS ---

def generate_shap_explanation(df, target_col):
    """
    Advance Level: Uses SHAP (Explainable AI) to identify feature importance.
    This helps identify 'Proxy Variables' causing the bias.
    """
    try:
        # Prepare data (dropping non-numeric for simple SHAP calculation)
        X = df.drop(columns=[target_col]).select_dtypes(include=['number'])
        y = df[target_col]
        
        # Train a fast diagnostic model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Generate Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # shap_values[1] represents the positive outcome (e.g., Approved)
        shap.summary_plot(shap_values[1], X, show=False)
        plt.title("Explainable AI: Feature Impact on Decisions")
        plt.tight_layout()
        
        return fig
    except Exception as e:
        print(f"SHAP Error: {e}")
        return None