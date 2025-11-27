# flask app placeholder
import os
import json
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from config import Config
from utils import ensure_upload_folder, read_csv_file, standardize_household, standardize_industrial
from ml_model import Forecaster, AnomalyDetector
from llm_client import LLMClient

ensure_upload_folder(Config.UPLOAD_FOLDER)
app = Flask(__name__)
CORS(app)

@app.route("/health")
def health():
    return jsonify({"ok": True})

@app.route("/upload", methods=["POST"])
def upload():
    """
    Expects multipart/form-data with file under 'file' and optional form field 'data_type' = 'household'|'industrial'
    Returns: cleaned_filename and preview rows
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "no file"}), 400
    f = request.files["file"]
    data_type = request.form.get("data_type", "household").lower()
    raw = read_csv_file(f)
    if data_type == "household":
        cleaned = standardize_household(raw)
        cleaned_name = (f.filename or "upload") + "_cleaned_household.csv"
    else:
        cleaned = standardize_industrial(raw)
        cleaned_name = (f.filename or "upload") + "_cleaned_industrial.csv"
    path = os.path.join(Config.UPLOAD_FOLDER, cleaned_name)
    cleaned.to_csv(path, index=False)
    preview = cleaned.head(50).to_dict(orient="records")
    return jsonify({"ok": True, "cleaned_filename": cleaned_name, "preview": preview})

@app.route("/download/<path:fname>")
def download(fname):
    path = os.path.join(Config.UPLOAD_FOLDER, fname)
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "not found"}), 404
    return send_file(path, as_attachment=True, download_name=fname)

@app.route("/forecast", methods=["POST"])
def forecast():
    """
    JSON: { "cleaned_filename": "<file>", "horizon": 24 }
    Response: per-type or aggregated forecasts
    """
    data = request.get_json(force=True)
    fname = data.get("cleaned_filename")
    n = int(data.get("horizon", 24))
    if not fname:
        return jsonify({"ok": False, "error": "cleaned_filename required"}), 400
    path = os.path.join(Config.UPLOAD_FOLDER, fname)
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "file not found"}), 404
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # household appliance-level
    if "appliance" in df.columns:
        agg = df.groupby("timestamp")["energy_kwh"].sum().reset_index()
        fore = Forecaster()
        r2 = fore.train(agg, "energy_kwh")
        preds = fore.predict_next_n(agg, n, "energy_kwh")
        return jsonify({"ok": True, "r2": r2, "predictions": preds.to_dict(orient="records")})
    # industrial multi-energy types
    if "energy_type" in df.columns:
        out = {}
        for etype, group in df.groupby("energy_type"):
            grp = group.rename(columns={"energy_value": "energy_kwh"})
            fore = Forecaster()
            r2 = fore.train(grp, "energy_kwh")
            preds = fore.predict_next_n(grp, n, "energy_kwh")
            out[etype] = preds.to_dict(orient="records")
        return jsonify({"ok": True, "per_type": out})
    # single series
    col = "energy_kwh" if "energy_kwh" in df.columns else df.columns[1]
    df = df.rename(columns={col: "energy_kwh"})
    fore = Forecaster()
    r2 = fore.train(df, "energy_kwh")
    preds = fore.predict_next_n(df, n, "energy_kwh")
    return jsonify({"ok": True, "r2": r2, "predictions": preds.to_dict(orient="records")})

@app.route("/anomalies", methods=["POST"])
def anomalies():
    """
    JSON: { "cleaned_filename": "<file>" }
    Returns anomalies as list of rows flagged
    """
    data = request.get_json(force=True)
    fname = data.get("cleaned_filename")
    if not fname:
        return jsonify({"ok": False, "error": "cleaned_filename required"}), 400
    path = os.path.join(Config.UPLOAD_FOLDER, fname)
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "file not found"}), 404
    df = pd.read_csv(path, parse_dates=["timestamp"])
    ad = AnomalyDetector()
    if "energy_kwh" in df.columns:
        ad.fit(df, "energy_kwh")
        out = ad.detect(df, "energy_kwh")
        return jsonify({"ok": True, "anomalies": out[out["anomaly"]].to_dict(orient="records")})
    if "energy_value" in df.columns and "energy_type" in df.columns:
        results = []
        for etype, group in df.groupby("energy_type"):
            grp = group.rename(columns={"energy_value": "energy_kwh"})
            ad.fit(grp, "energy_kwh")
            res = ad.detect(grp, "energy_kwh")
            res["energy_type"] = etype
            results.extend(res[res["anomaly"]].to_dict(orient="records"))
        return jsonify({"ok": True, "anomalies": results})
    return jsonify({"ok": False, "error": "no recognizable energy column"}), 400

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    JSON: { "cleaned_filename": "<file>", "price_per_kwh": 0.15, "horizon": 24, "context": "" }
    Returns JSON recommendations (or raw text)
    """
    data = request.get_json(force=True)
    fname = data.get("cleaned_filename")
    price = float(data.get("price_per_kwh", 0.15))
    horizon = int(data.get("horizon", 24))
    context = data.get("context", "")
    if not fname:
        return jsonify({"ok": False, "error": "cleaned_filename required"}), 400
    path = os.path.join(Config.UPLOAD_FOLDER, fname)
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "file not found"}), 404
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Build brief metrics for the LLM prompt
    if "appliance" in df.columns:
        total = float(df["energy_kwh"].sum())
        peak = int(df.assign(hour=df["timestamp"].dt.hour).groupby("hour")["energy_kwh"].sum().idxmax())
        summary = f"total_kwh={total:.2f}, peak_hour={peak}"
    elif "energy_type" in df.columns:
        totals = df.groupby("energy_type")["energy_value"].sum().to_dict()
        summary = " ".join([f"{k}={v:.2f}" for k, v in totals.items()])
    else:
        col = "energy_kwh" if "energy_kwh" in df.columns else df.columns[1]
        total = float(df[col].sum())
        peak = int(df.assign(hour=df["timestamp"].dt.hour).groupby("hour")[col].sum().idxmax())
        summary = f"total_kwh={total:.2f}, peak_hour={peak}"

    system_prompt = "You are an energy advisor that produces JSON recommendations."
    user_prompt = f"Metrics: {summary} price_per_kwh={price} context={context}"

    llm = LLMClient()
    resp = llm.generate(system_prompt, user_prompt)
    try:
        parsed = json.loads(resp)
        return jsonify({"ok": True, "recommendations": parsed})
    except Exception:
        return jsonify({"ok": True, "recommendations_text": resp})

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT)
