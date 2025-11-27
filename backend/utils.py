import io
import pandas as pd
import numpy as np
from datetime import datetime
import os

def ensure_upload_folder(path: str):
    os.makedirs(path, exist_ok=True)

def read_csv_file(file_storage):
    """
    Accepts Werkzeug FileStorage and returns a pandas DataFrame.
    Tries common delimiters and handles encoding issues.
    """
    content = file_storage.read()
    try:
        text = content.decode("utf-8")
    except Exception:
        text = content.decode("latin1")
    # try delimiters
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    # fallback
    return pd.read_csv(io.StringIO(text))

def standardize_household(df):
    """
    Normalize household appliance-level CSVs to a long form:
    columns: timestamp, appliance, energy_kwh
    Supports either appliance_* columns (wide) or rows with 'appliance' + energy column.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # find timestamp column
    ts_cols = [c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower() or 'date' in c.lower()]
    if len(ts_cols) == 0:
        df['timestamp'] = pd.date_range(datetime.utcnow() - pd.Timedelta(hours=len(df)-1), periods=len(df), freq='H')
    else:
        df = df.rename(columns={ts_cols[0]: 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # detect appliance columns like fridge_kwh, hvac_kwh, etc.
    appliance_cols = [c for c in df.columns if c.lower().endswith('_kwh') or ('appliance' in c.lower() and c != 'appliances')]
    if appliance_cols:
        records = []
        for _, row in df.iterrows():
            for c in appliance_cols:
                records.append({'timestamp': row['timestamp'], 'appliance': c.replace('_kwh', ''), 'energy_kwh': row[c]})
        out = pd.DataFrame(records)
        out = out.dropna(subset=['timestamp'])
        return out

    # If there's explicit 'appliance' and an energy column
    energy_cols = [c for c in df.columns if 'kwh' in c.lower() or 'energy' in c.lower()]
    if 'appliance' in df.columns and energy_cols:
        return df[['timestamp', 'appliance', energy_cols[0]]].rename(columns={energy_cols[0]: 'energy_kwh'}).dropna(subset=['timestamp'])

    # If a single total energy column
    if energy_cols:
        return df[['timestamp', energy_cols[0]]].rename(columns={energy_cols[0]: 'energy_kwh'}).dropna(subset=['timestamp'])

    # fallback - produce a zero series
    df['energy_kwh'] = 0.0
    return df[['timestamp', 'energy_kwh']]

def standardize_industrial(df):
    """
    Normalize industrial CSVs into long form:
    timestamp, energy_type, energy_value
    Recognizes columns with names: electricity_kwh, gas_therms, diesel_liters, etc.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    ts_cols = [c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower() or 'date' in c.lower()]
    if len(ts_cols) == 0:
        df['timestamp'] = pd.date_range(datetime.utcnow() - pd.Timedelta(hours=len(df)-1), periods=len(df), freq='H')
    else:
        df = df.rename(columns={ts_cols[0]: 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    energy_cols = [c for c in df.columns if any(x in c.lower() for x in ['kwh', 'energy', 'gas', 'fuel', 'therm', 'liters', 'mwh'])]
    if energy_cols:
        records = []
        for _, row in df.iterrows():
            for c in energy_cols:
                records.append({'timestamp': row['timestamp'], 'energy_type': c, 'energy_value': row[c]})
        out = pd.DataFrame(records)
        out = out.dropna(subset=['timestamp'])
        return out

    # fallback
    df['energy_value'] = 0.0
    return df[['timestamp', 'energy_value']]
