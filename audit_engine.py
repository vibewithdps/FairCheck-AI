import pandas as pd
from firebase_admin import credentials, db, initialize_app, _apps
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import math
import streamlit as st  # Added to access Cloud Secrets

def setup_firebase():
    """
    Professional Cloud-Ready Initialization.
    Automatically switches between local JSON and Streamlit Secrets.
    """
    if not _apps:
        try:
            # 1. Check if running on Streamlit Cloud (using Secrets)
            if "firebase" in st.secrets:
                # Convert the Secret TOML data into a dictionary
                key_dict = dict(st.secrets["firebase"])
                
                # CRITICAL: Fix the multi-line private key format for Google Auth
                if "private_key" in key_dict:
                    key_dict["private_key"] = key_dict["private_key"].replace("\\n", "\n")
                
                cred = credentials.Certificate(key_dict)
            
            # 2. Fallback for local testing on your MacBook
            else:
                cred = credentials.Certificate("serviceAccountKey.json")

            initialize_app(cred, {
                'databaseURL': 'https://faircheck-ffe01-default-rtdb.firebaseio.com/'
            })
            return True
        except Exception as e:
            # Displays the specific error in the sidebar for easier debugging
            st.sidebar.error(f"Firebase Config Error: {str(e)}")
            return False
    return True

def run_fairness_audit(df, target, group_col, priv_val, unprivileged_value):
    try:
        # Convert to numeric and handle missing values to avoid 'NaN' errors
        df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0)
        
        priv_rate = df[df[group_col] == priv_val][target].mean()
        unpriv_rate = df[df[group_col] == unprivileged_value][target].mean()
        
        # --- Advance Level Robustness: Handling Edge Cases (NaN/Inf) ---
        if priv_rate == 0 or math.isnan(priv_rate):
            di_ratio = 0.0
        else:
            di_ratio = unpriv_rate / priv_rate
            
        if math.isinf(di_ratio) or math.isnan(di_ratio):
            di_ratio = 0.0
            
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
        
        # Secure Firebase Logging
        if setup_firebase(): 
            try:
                db.reference('audit_logs').push(results)
            except Exception as fe:
                print(f"Firebase Sync Error: {fe}")
            
        return results
    except Exception as e:
        return {"error": str(e)}

def generate_shap_explanation(df, target_col):
    """
    Explainable AI (XAI) using SHAP values.
    Identifies if 'Proxy Variables' are influencing biased outcomes.
    """
    try:
        # Prepare data: keep only numeric features for SHAP calculation
        X = df.drop(columns=[target_col]).select_dtypes(include=['number'])
        y = df[target_col]
        
        if X.empty:
            return None

        # Train diagnostic model (Random Forest)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Generate Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # Use index [1] for the positive outcome (e.g., 'Approved')
        shap.summary_plot(shap_values[1], X, show=False)
        plt.title("Explainable AI: Feature Impact on Decisions")
        plt.tight_layout()
        
        return fig
    except Exception as e:
        print(f"SHAP Error: {e}")
        return None