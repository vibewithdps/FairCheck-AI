from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
from audit_engine import run_fairness_audit, calculate_reweighing_weights, generate_shap_explanation, fetch_audit_logs

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/ping")
def ping():
    return {"status": "ok", "message": "FairCheck AI Backend is running."}

@app.post("/api/audit")
async def audit_endpoint(
    file: UploadFile = File(...),
    target: str = Form(...),
    group: str = Form(...),
    priv_val: str = Form(...),
    unpriv_val: str = Form(...)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # 1. Run Fairness Audit
        report = run_fairness_audit(df, target, group, priv_val, unpriv_val)
        if "error" in report:
            raise HTTPException(status_code=400, detail=report["error"])
            
        # 2. SHAP Explanation
        shap_base64 = generate_shap_explanation(df, target)
        
        # 3. Reweighing Weights
        weights_df = calculate_reweighing_weights(df, target, group)
        weights_json = None
        if weights_df is not None:
            weights_json = weights_df.head(10).to_dict(orient="records")
            
        # 4. Bias Map data (histogram distribution)
        # Instead of returning plotly chart, return aggregate data so frontend can plot it
        group_dist = df.groupby([group, target]).size().reset_index(name='count')
        
        return {
            "report": report,
            "shap_base64": shap_base64,
            "weights_sample": weights_json,
            "distribution": group_dist.to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
def get_history():
    try:
        logs = fetch_audit_logs()
        if not logs:
            return {"history": []}
            
        records = []
        for key, val in logs.items():
            if isinstance(val, dict):
                val['id'] = key
                records.append(val)
        return {"history": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
