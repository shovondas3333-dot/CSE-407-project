from flask import Flask, jsonify, render_template
from flask import request
from flask_sqlalchemy import SQLAlchemy
import tinytuya
import threading
import time
from datetime import datetime
from sqlalchemy.sql import func
from sqlalchemy import text

from flask import send_file
import pandas as pd
from sqlalchemy import text
import io
import os

app = Flask(__name__)

# Simple in-memory cache for Excel reads keyed by file mtime
excel_cache = {
    'mtime': None,
    'minutely_df': None
}

# Tuya Device Info
DEVICE_ID = "bf035aef5b8c5240dbykne"
LOCAL_KEY = "}=rHhdU-JWFeL3CB"
DEVICE_IP = "192.168.10.171"
PROTOCOL_VERSION = "3.5"

device = tinytuya.OutletDevice(DEVICE_ID, DEVICE_IP, LOCAL_KEY)
device.set_version(float(PROTOCOL_VERSION))

# SQLite Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///energy_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model
class EnergyData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(20))
    watt = db.Column(db.Float)
    voltage = db.Column(db.Float)
    current = db.Column(db.Float)
    kwh = db.Column(db.Float)

with app.app_context():
    db.create_all()
#render
# Polling Function (safe with app context)
def poll_device(interval=10):
    while True:
        try:
            data = device.status()
            dp = data.get("dps", {})

            raw_watt = dp.get("19", 0)
            voltage = dp.get("20", 0)
            current = dp.get("18", 0)

            # Corrected watt value
            watt = raw_watt / 10
            
            # store timestamp with millisecond precision in ISO format
            try:
                # Python 3.6+ supports timespec
                timestamp = datetime.now().isoformat(timespec='milliseconds')
            except TypeError:
                # fallback: format manually to milliseconds
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            kwh = watt * (interval / 3600) / 1000  # convert to kWh

            with app.app_context():
                entry = EnergyData(
                    timestamp=timestamp,
                    watt=watt,
                    voltage=voltage,
                    current=current,
                    kwh=kwh
                )
                db.session.add(entry)
                db.session.commit()

            print(f"[{timestamp}] W: {watt}W, V: {voltage}V, A: {current}mA, kWh: {kwh:.6f}")
        except Exception as e:
            print("Error fetching data:", e)
        time.sleep(interval)

# Flag to avoid multiple thread start
polling_started = False

# Routes
@app.route('/')
def dashboard():
    global polling_started
    if not polling_started:
        threading.Thread(target=poll_device, daemon=True).start()
        polling_started = True
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    entries = EnergyData.query.order_by(EnergyData.id.desc()).limit(60).all()
    entries.reverse()
    return jsonify([
        {
            "timestamp": e.timestamp,
            "watt": e.watt,
            "voltage": e.voltage,
            "current": e.current
        } for e in entries
    ])

@app.route('/api/total-kwh')
def total_kwh():
    total = db.session.query(db.func.sum(EnergyData.kwh)).scalar() or 0
    return jsonify({"total_kwh": round(total, 4)})

@app.route('/api/stats')
def energy_stats():
    daily = db.session.execute(
        """
        SELECT SUBSTR(timestamp, 1, 10) AS day, SUM(kwh)
        FROM energy_data
        GROUP BY day
        ORDER BY day DESC
        LIMIT 7
        """
    ).fetchall()

    hourly = db.session.execute(
        """
        SELECT SUBSTR(timestamp, 1, 13) AS hour, SUM(kwh)
        FROM energy_data
        GROUP BY hour
        ORDER BY hour DESC
        LIMIT 24
        """
    ).fetchall()

    return jsonify({
        "daily": [{"day": d[0], "kwh": round(d[1], 4)} for d in daily],
        "hourly": [{"hour": h[0], "kwh": round(h[1], 4)} for h in hourly]
    })

@app.route('/api/stats/minutely')
def minutely_stats():
    # Group by minute (first 16 chars of timestamp: 'YYYY-MM-DD HH:MM')
    results = db.session.execute(text("""
    SELECT SUBSTR(timestamp, 1, 16) AS minute, SUM(kwh) AS total_kwh
    FROM energy_data
    GROUP BY minute
    ORDER BY minute DESC
    LIMIT 60
    """)).fetchall()

    # Reverse so oldest to newest
    results = list(reversed(results))

    return jsonify([
        {"minute": r[0], "total_kwh": round(r[1], 6)} for r in results
    ])

@app.route('/export/full-energy-report')
def export_full_energy_report():
    # Fetch full raw energy data
    raw_data = db.session.execute(text("""
        SELECT timestamp, watt, current, voltage, kwh
        FROM energy_data
        ORDER BY timestamp ASC
    """)).fetchall()

    # Scale voltage for the export
    scaled_data = []
    for row in raw_data:
        timestamp, watt, current, voltage, kwh = row
        voltage = voltage / 10  # scale down voltage
        scaled_data.append((timestamp, watt, current, voltage, kwh))

    df_raw = pd.DataFrame(scaled_data, columns=['Timestamp', 'Watt', 'Current', 'Voltage', 'kWh'])
    df_raw['kWh'] = df_raw['kWh'].round(6)

    # Group by each minute for minutely total kWh
    df_raw['Minute'] = df_raw['Timestamp'].astype(str).str.slice(0, 16)
    df_minutely = df_raw.groupby('Minute', as_index=False)['kWh'].sum()
    df_minutely.rename(columns={'kWh': 'Total_kWh'}, inplace=True)
    df_minutely['Total_kWh'] = df_minutely['Total_kWh'].round(6)

    # Write both sheets into one Excel file
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_raw.drop(columns='Minute').to_excel(writer, index=False, sheet_name='Raw Data')
        df_minutely.to_excel(writer, index=False, sheet_name='Minutely Report')

    output.seek(0)

    return send_file(output,
                     as_attachment=True,
                     download_name='full_energy_report.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

#http://localhost:5000/export/full-energy-report
@app.route('/api/graph-data')
def api_graph_data():
    results = db.session.execute(text("""
        SELECT timestamp, current, watt, voltage
        FROM energy_data
        ORDER BY timestamp DESC
        LIMIT 100
    """)).fetchall()
    # results may have timestamp stored as string (ISO) or as datetime
    data = []
    for row in results[::-1]:  # reverse for chronological order
        ts = row.timestamp
        if hasattr(ts, 'strftime'):
            # datetime-like
            ts_val = ts.isoformat()
        else:
            ts_val = str(ts)

        data.append({
            'timestamp': ts_val,
            'current': round(row.current, 6) if row.current is not None else None,
            'watt': round(row.watt, 3) if row.watt is not None else None,
            'voltage': round(row.voltage, 3) if row.voltage is not None else None
        })

    return jsonify(data)


@app.route('/api/device/state')
def device_state():
    """Return the raw DPS dictionary from the Tuya device and a best-effort inferred power state.

    Response: { dps: {...}, power_dps: '<id>' or null, power: true/false/null }
    """
    try:
        data = device.status()
        dp = data.get('dps', {}) if isinstance(data, dict) else {}
        power_dps = None
        power = None
        # try to find a DPS key that looks like on/off (bool or 0/1)
        for k, v in dp.items():
            if isinstance(v, bool):
                power_dps = k
                power = v
                break
            if isinstance(v, (int, float)) and v in (0, 1):
                power_dps = k
                power = bool(int(v))
                break
            if isinstance(v, str) and v in ('0', '1'):
                power_dps = k
                power = (v == '1')
                break

        return jsonify({'dps': dp, 'power_dps': power_dps, 'power': power})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/device/on', methods=['POST'])
def device_on():
    try:
        # try high-level API
        if hasattr(device, 'turn_on'):
            device.turn_on()
        elif hasattr(device, 'set_status'):
            device.set_status(True)
        else:
            # fallback: set dps 1 to True (common for many plugs)
            device.set_dps({'1': True})
        return jsonify({'result': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/device/off', methods=['POST'])
def device_off():
    try:
        if hasattr(device, 'turn_off'):
            device.turn_off()
        elif hasattr(device, 'set_status'):
            device.set_status(False)
        else:
            device.set_dps({'1': False})
        return jsonify({'result': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/excel-graph-data')
def excel_graph_data():
    """Read `full_energy_report.xlsx` (Minutely Report sheet) and return JSON for plotting.

    Returns:
        JSON list of {minute: 'HH:MM', total_kwh: float}
    """
    excel_path = os.path.join(app.root_path, 'full_energy_report.xlsx')
    if not os.path.exists(excel_path):
        return jsonify({"error": "full_energy_report.xlsx not found"}), 404

    try:
        # The export creates a sheet named 'Minutely Report' with columns ['Minute', 'Total_kWh']
        df = pd.read_excel(excel_path, sheet_name='Minutely Report')
    except Exception:
        # Fallback: try reading first sheet
        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            return jsonify({"error": f"Failed to read excel: {e}"}), 500

    # Normalize column names and ensure expected columns exist
    cols = [c.lower() for c in df.columns]
    minute_col = None
    kwh_col = None
    for c in df.columns:
        lc = c.lower()
        if 'minute' in lc or 'timestamp' in lc or 'time' in lc:
            minute_col = c
        if 'kwh' in lc or 'total' in lc:
            kwh_col = c

    if minute_col is None or kwh_col is None:
        return jsonify({"error": "Expected columns (Minute, Total_kWh) not found in Excel"}), 500

    # Prepare output: show only HH:MM for labels
    out = []
    for _, row in df.iterrows():
        minute_val = str(row[minute_col])
        # try to extract HH:MM
        hhmm = minute_val[-5:] if len(minute_val) >= 5 else minute_val
        try:
            total = float(row[kwh_col])
        except Exception:
            total = None
        out.append({"minute": hhmm, "total_kwh": total})

    return jsonify(out)


def _load_minutely_from_excel(excel_path):
    """Load or reload the minutely report DataFrame from the Excel file.

    Returns (df_minutely, mtime) or (None, None) on error.
    Caches the DataFrame until the file mtime changes.
    """
    try:
        mtime = os.path.getmtime(excel_path)
    except Exception:
        return None, None

    if excel_cache.get('mtime') == mtime and excel_cache.get('minutely_df') is not None:
        return excel_cache['minutely_df'], mtime

    try:
        # Try reading Minutely Report sheet first
        xls = pd.read_excel(excel_path, sheet_name=None)
        if 'Minutely Report' in xls:
            df_min = pd.read_excel(excel_path, sheet_name='Minutely Report')
            # find minute and kwh columns heuristically
            minute_col = next((c for c in df_min.columns if 'minute' in c.lower() or 'time' in c.lower()), df_min.columns[0])
            kwh_col = next((c for c in df_min.columns if 'kwh' in c.lower() or 'total' in c.lower()), df_min.columns[1] if len(df_min.columns) > 1 else df_min.columns[-1])
            df_min = df_min.rename(columns={minute_col: 'Minute', kwh_col: 'Total_kwh'})
            # parse Minute as datetime where possible
            try:
                df_min['Minute'] = pd.to_datetime(df_min['Minute'])
            except Exception:
                # try parsing HH:MM only by attaching today's date
                df_min['Minute'] = pd.to_datetime(df_min['Minute'], errors='coerce')
            df_min['Total_kwh'] = pd.to_numeric(df_min['Total_kwh'], errors='coerce').fillna(0)
        else:
            # Fallback: build minutely from Raw Data (or first sheet)
            if 'Raw Data' in xls:
                df_raw = pd.read_excel(excel_path, sheet_name='Raw Data')
            else:
                # read first sheet
                first_sheet = list(xls.keys())[0]
                df_raw = pd.read_excel(excel_path, sheet_name=first_sheet)

            # find timestamp and kwh columns
            ts_col = next((c for c in df_raw.columns if 'timestamp' in c.lower() or 'time' in c.lower()), df_raw.columns[0])
            kwh_col = next((c for c in df_raw.columns if 'kwh' in c.lower()), df_raw.columns[-1])
            df_raw = df_raw.rename(columns={ts_col: 'Timestamp', kwh_col: 'kWh'})
            df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], errors='coerce')
            df_raw = df_raw.dropna(subset=['Timestamp'])
            df_raw['Minute'] = df_raw['Timestamp'].dt.floor('T')
            df_min = df_raw.groupby('Minute', as_index=False)['kWh'].sum().rename(columns={'kWh': 'Total_kwh'})

        # Cache and return
        excel_cache['mtime'] = mtime
        excel_cache['minutely_df'] = df_min
        return df_min, mtime
    except Exception as e:
        print('Error reading excel for minutely:', e)
        return None, None


@app.route('/api/excel-aggregates')
def excel_aggregates():
    """Return aggregate totals from the Excel minutely report.

    Response JSON:
      { file_mtime, total_kwh, last_hour_kwh, last_24h_kwh, last_7d_kwh }
    """
    excel_path = os.path.join(app.root_path, 'full_energy_report.xlsx')
    if not os.path.exists(excel_path):
        return jsonify({"error": "full_energy_report.xlsx not found"}), 404

    df_min, mtime = _load_minutely_from_excel(excel_path)
    if df_min is None:
        return jsonify({"error": "Failed to read minutely data from excel"}), 500

    # ensure Minute is datetime
    if 'Minute' not in df_min.columns:
        return jsonify({"error": "Minute column missing in minutely data"}), 500

    df = df_min.copy()
    try:
        df['Minute'] = pd.to_datetime(df['Minute'], errors='coerce')
    except Exception:
        pass

    now = pd.Timestamp.now()
    total = float(df['Total_kwh'].sum()) if not df['Total_kwh'].empty else 0.0
    last_hour = float(df.loc[df['Minute'] >= (now - pd.Timedelta(hours=1)), 'Total_kwh'].sum()) if 'Minute' in df.columns else 0.0
    last_24h = float(df.loc[df['Minute'] >= (now - pd.Timedelta(days=1)), 'Total_kwh'].sum()) if 'Minute' in df.columns else 0.0
    last_7d = float(df.loc[df['Minute'] >= (now - pd.Timedelta(days=7)), 'Total_kwh'].sum()) if 'Minute' in df.columns else 0.0

    return jsonify({
        'file_mtime': mtime,
        'total_kwh': round(total, 6),
        'last_hour_kwh': round(last_hour, 6),
        'last_24h_kwh': round(last_24h, 6),
        'last_7d_kwh': round(last_7d, 6)
    })


@app.route('/api/excel-latest')
def excel_latest():
    """Return the latest raw data row from `full_energy_report.xlsx` (sheet 'Raw Data').

    Response JSON: { timestamp, watt, current, voltage, kwh }
    """
    excel_path = os.path.join(app.root_path, 'full_energy_report.xlsx')
    if not os.path.exists(excel_path):
        return jsonify({"error": "full_energy_report.xlsx not found"}), 404

    try:
        df = pd.read_excel(excel_path, sheet_name='Raw Data')
    except Exception:
        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            return jsonify({"error": f"Failed to read excel: {e}"}), 500

    if df.empty:
        return jsonify({"error": "No data in Excel"}), 404

    # Use the last row as the latest
    last = df.iloc[-1]
    # Normalize column name matching (case-insensitive)
    def find_col(df, names):
        for n in df.columns:
            ln = n.lower()
            for candidate in names:
                if candidate in ln:
                    return n
        return None

    ts_col = find_col(df, ['timestamp', 'time', 'minute'])
    watt_col = find_col(df, ['watt'])
    current_col = find_col(df, ['current'])
    volt_col = find_col(df, ['volt', 'voltage'])
    kwh_col = find_col(df, ['kwh'])

    def safe_get(row, col):
        try:
            return row[col]
        except Exception:
            return None

    timestamp = str(safe_get(last, ts_col)) if ts_col else None
    watt = safe_get(last, watt_col)
    current = safe_get(last, current_col)
    voltage = safe_get(last, volt_col)
    kwh = safe_get(last, kwh_col)

    # Ensure numeric where possible
    try:
        watt = float(watt) if watt is not None else None
    except Exception:
        watt = None
    try:
        current = float(current) if current is not None else None
    except Exception:
        current = None
    try:
        voltage = float(voltage) if voltage is not None else None
    except Exception:
        voltage = None
    try:
        kwh = float(kwh) if kwh is not None else None
    except Exception:
        kwh = None

    return jsonify({
        'timestamp': timestamp,
        'watt': round(watt, 3) if watt is not None else None,
        'current': round(current, 6) if current is not None else None,
        'voltage': round(voltage, 3) if voltage is not None else None,
        'kwh': round(kwh, 6) if kwh is not None else None
    })

if __name__ == '__main__':
    app.run(debug=True)
