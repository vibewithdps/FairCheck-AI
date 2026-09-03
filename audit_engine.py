import pandas as pd
from firebase_admin import credentials, db, initialize_app, _apps
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import math
import numpy as np
import streamlit as st 

def setup_firebase():
    """
    Professional Cloud-Ready Initialization.
    Automatically switches between local JSON and Streamlit Secrets.
    """
    if not _apps:
        try:
            # Check if running on Streamlit Cloud with secrets
            if "firebase" in st.secrets:
                key_dict = dict(st.secrets["firebase"])
                if "private_key" in key_dict:
                    key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")
                cred = credentials.Certificate(key_dict)
            else:
                # Local development path
                cred = credentials.Certificate("serviceAccountKey.json")

            initialize_app(cred, {
                'databaseURL': 'https://faircheck-ai-default-rtdb.firebaseio.com/'
            })
            return True
        except Exception as e:
            st.sidebar.error(f"Firebase Config Error: {str(e)}")
            return False
    return True

def run_fairness_audit(df, target, group_col, priv_val, unprivileged_value):
    """
    Calculates Disparate Impact Ratio and Statistical Parity.
    Ensures 'Reduced Inequalities' (SDG 10) by auditing AI bias.
    """
    try:
        df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)
        
        # Calculate rates for the two groups
        priv_df = df[df[group_col] == priv_val]
        unpriv_df = df[df[group_col] == unprivileged_value]
        
        priv_rate = priv_df[target].mean()
        unpriv_rate = unpriv_df[target].mean()
        
        # --- Robustness: Handling Edge Cases ---
        if priv_rate == 0 or math.isnan(priv_rate):
            di_ratio = 0.0
        else:
            di_ratio = unpriv_rate / priv_rate
            
        if math.isinf(di_ratio) or math.isnan(di_ratio):
            di_ratio = 0.0
            
        # Status based on the 80% Rule (0.8 threshold)
        status = "✅ Fair" if 0.8 <= di_ratio <= 1.25 else "⚠️ Biased"
        
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
        
        if setup_firebase(): 
            try:
                db.reference('audit_logs').push(results)
            except Exception as fe:
                print(f"Firebase Sync Error: {fe}")
            
        return results
    except Exception as e:
        return {"error": str(e)}

def fetch_audit_logs():
    """
    Fetches past audit logs from Firebase.
    """
    try:
        if setup_firebase():
            ref = db.reference('audit_logs')
            return ref.get()
    except Exception as e:
        print(f"Firebase Fetch Error: {e}")
    return None

def calculate_reweighing_weights(df, target, group_col):
    """
    PhD-Level Mitigation Technique: 
    Generates weights to balance the dataset before training.
    """
    try:
        # Create a working copy to avoid modifying the original dataframe in memory
        df_work = df.copy()
        n = len(df_work)
        n_pos = len(df_work[df_work[target] == 1])
        n_neg = len(df_work[df_work[target] == 0])
        
        df_work['fairness_weight'] = 1.0 # Default weight
        groups = df_work[group_col].unique()
        
        for g in groups:
            n_g = len(df_work[df_work[group_col] == g])
            # Observed counts
            n_g_pos = len(df_work[(df_work[group_col] == g) & (df_work[target] == 1)])
            n_g_neg = len(df_work[(df_work[group_col] == g) & (df_work[target] == 0)])
            
            # Expected counts for Statistical Parity
            expected_pos = (n_g * n_pos) / n
            expected_neg = (n_g * n_neg) / n
            
            # Calculate and assign weights (Expected / Observed)
            if n_g_pos > 0:
                df_work.loc[(df_work[group_col] == g) & (df_work[target] == 1), 'fairness_weight'] = expected_pos / n_g_pos
            if n_g_neg > 0:
                df_work.loc[(df_work[group_col] == g) & (df_work[target] == 0), 'fairness_weight'] = expected_neg / n_g_neg
                
        return df_work[['fairness_weight']]
    except Exception as e:
        print(f"Mitigation Calculation Error: {e}")
        return None

def generate_shap_explanation(df, target_col):
    """
    Explainable AI (XAI) using SHAP values.
    Identifies 'Proxy Variables' that might hide bias.
    """
    try:
        # Optimization: Filter to numeric for Random Forest processing
        X = df.drop(columns=[target_col]).select_dtypes(include=['number'])
        y = df[target_col]
        
        if X.empty:
            return None

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use class 1 results for binary classification visualization
        shap.summary_plot(shap_values[1], X, show=False)
        plt.title("Explainable AI: Feature Impact on Decisions")
        plt.tight_layout()
        
        return fig
    except Exception as e:
        print(f"SHAP Error: {e}")
        return None