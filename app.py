import streamlit as st
import pandas as pd
from audit_engine import setup_firebase, run_fairness_audit, generate_shap_explanation 
import datetime
import os  

# 1. Page Configuration
st.set_page_config(page_title="FairCheck AI", layout="wide")

# --- SIDEBAR CONTENT WITH LOGO AND UPDATED TEAM ---
with st.sidebar:
    # Adding the Project Logo at the very top
    if os.path.exists("logo.jpg"):
        st.image("logo.jpg", use_container_width=True)
    else:
        st.title(" FairCheck AI")
    
    st.markdown("### 👥 Project Team")
    
    # Function to display team member with photo and updated formatting
    def display_team_member(name, role, img_file):
        col1, col2 = st.columns([1, 3])
        with col1:
            if os.path.exists(img_file):
                st.image(img_file, width=50)
            else:
                st.write("👤") 
        with col2:
            st.markdown(f"**{name}** \n*{role}*")

    # Updated Team Members and Roles
    display_team_member("Dipendra Pratap Singh", "(Team Leader)", "dipendra.jpg")
    display_team_member("Sakshi Chauhan", "(Data Science)", "sakshi.jpg")
    display_team_member("Neha Yadav", "(Frontend/UI)", "neha.jpg")
    
    st.divider()
    
    # Firebase Status
    st.markdown("### 🔌 System Status")
    if setup_firebase():
        st.success("📡 Firebase: Connected")
    else:
        st.error("❌ Firebase: Disconnected")
        
    st.divider()
    
    # How It Works Guide for Judges
    st.markdown("### 📖 How it Works")
    st.caption("""
    1. **Upload**: Provide model prediction data (CSV).
    2. **Config**: Select target and protected groups.
    3. **Audit**: We calculate the *Disparate Impact Ratio*.
    4. **Report**: Get a legal-ready transparency report synced to the cloud.
    """)
    
    st.divider()

    # --- NEW PROFESSIONAL CONTACT & FOOTER SECTION ---
    st.markdown("### 📞 Get Help & Support")
    st.caption("For technical queries or audit support:")
    
    # Professional Contact Layout
    cont_col1, cont_col2 = st.columns([1, 5])
    with cont_col1:
        st.write("📱")
        st.write("✉️")
    with cont_col2:
        st.markdown("**+91 7037788052**")
        st.markdown("**thakurdps795@gmail.com**")

    st.divider()
    
    # Branded Footer (Website Style)
    st.markdown(
        """
        <div style="text-align: center; color: #808495; font-size: 13px; font-family: sans-serif;">
            <b>FairCheck AI v1.0</b><br>
            © 2026 Atmiya University, Rajkot<br>
            <i>Innovation Hub - Gujarat, India</i>
        </div>
        """, 
        unsafe_allow_html=True
    )
# --- END OF SIDEBAR ---

# Main UI Header
st.title("FairCheck: AI Transparency Dashboard")
st.subheader("Ensuring Ethical & Unbiased Automated Decisions") 
st.markdown(f"**Team Members:** Dipendra Pratap Singh, Sakshi Chauhan, Neha Yadav")

# 3. File Upload Section
uploaded_file = st.file_uploader("Upload your Dataset (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Dataset Overview")
    st.dataframe(data.head(10), use_container_width=True)

    st.divider()
    
    # 4. Configuration Columns
    st.subheader("⚙️ Audit Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target = st.selectbox("Select Result Column (e.g., 'Approved')", data.columns)
    with col2:
        group = st.selectbox("Select Protected Group (e.g., 'Gender')", data.columns)
    with col3:
        all_vals = data[group].unique().tolist()
        priv_val = st.selectbox("Privileged Value", all_vals)
        remaining_vals = [v for v in all_vals if v != priv_val]
        unpriv_val = st.selectbox("Unprivileged Value", remaining_vals)

    # 5. Run Button
    if st.button("🔍 Run Fairness Audit"):
        with st.spinner("Analyzing decisions for bias..."):
            report = run_fairness_audit(data, target, group, priv_val, unpriv_val)
            
            if "error" in report:
                st.error(f"Analysis Error: {report['error']}")
            else:
                st.divider()
                # 1. Metrics Display
                res1, res2, res3 = st.columns(3)
                res1.metric("Fairness Score (DI Ratio)", report['score'])
                res2.metric("Conclusion", report['status'])
                res3.metric("Success Gap", f"{report['unpriv_group_success']} vs {report['priv_group_success']}")
                
                # 2. Visualization
                chart_data = pd.DataFrame({
                    "Group": [priv_val, unpriv_val],
                    "Success Rate (%)": [
                        float(report['priv_group_success'].strip('%')), 
                        float(report['unpriv_group_success'].strip('%'))
                    ]
                })
                st.bar_chart(chart_data, x="Group", y="Success Rate (%)")

                # --- PHD-LEVEL RESEARCH SECTION ---
                st.divider()
                st.subheader("🔬 Advanced Research Metrics")
                
                exp_col1, exp_col2 = st.columns([2, 1])
                
                with exp_col1:
                    with st.expander("🔍 Feature Importance & XAI (SHAP Analysis)", expanded=False):
                        st.write("This section identifies 'Proxy Variables'. Even if gender is removed, the AI might use other variables to guess it.")
                        fig = generate_shap_explanation(data, target)
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.warning("Ensure the dataset has numeric features to generate SHAP explanations.")
                
                with exp_col2:
                    st.write("**Statistical Parity Difference**")
                    st.info(f"SPD Score: {report.get('stat_parity', 'N/A')}")
                    st.caption("PhD Note: SPD < 0 indicates bias against the unprivileged group.")

                # --- REPORT DOWNLOAD ---
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                is_fair = report['score'] >= 0.8
                compliance_status = "Likely Compliant (Pass)" if is_fair else "Non-Compliant (Risk Found)"
                action_plan = "✅ Maintain current monitoring." if is_fair else f"❌ URGENT: Re-examine training weights for the '{unpriv_val}' group."

                report_text = f"""
====================================================
🛡️ FAIRCHECK AI: OFFICIAL TRANSPARENCY AUDIT 🛡️
====================================================
Generated on: {timestamp}
Team: Dipendra Pratap Singh, Sakshi Chauhan, Neha Yadav

1. AUDIT SUMMARY
----------------------------------------------------
Target Column: {target}
Audited Attribute: {group}
Privileged Group: {priv_val}
Unprivileged Group: {unpriv_val}

2. CORE METRICS
----------------------------------------------------
- Fairness Score (DI Ratio): {report['score']}
- Statistical Parity Diff: {report.get('stat_parity', 'N/A')}
- Result Status: {report['status']}
- Success Rate ({priv_val}): {report['priv_group_success']}
- Success Rate ({unpriv_val}): {report['unpriv_group_success']}

3. LEGAL & ETHICAL COMPLIANCE (India DPDP Act Ref)
----------------------------------------------------
Status: {compliance_status}
Summary: This tool evaluates algorithmic discrimination as per 
emerging global AI safety standards.

4. MITIGATION & ACTION PLAN
----------------------------------------------------
Recommendation: {action_plan}

Generated via FairCheck AI Dashboard.
====================================================
                """

                st.download_button(
                    label="📥 Download Professional Audit Report (.txt)",
                    data=report_text,
                    file_name=f"FairCheck_Audit_{group}.txt",
                    mime="text/plain"
                )
                
                st.success("✅ Audit results synced to Firebase and professional report is ready!")
else:
    st.warning("Please upload a CSV file to begin the audit.")