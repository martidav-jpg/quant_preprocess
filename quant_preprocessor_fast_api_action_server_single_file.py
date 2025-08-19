# quant_preprocessor_action_server.py
# Single-file FastAPI wrapper that ingests CSV URLs from a GPT Action,
# runs the file-pure preprocessor, and serves the Excel + meta JSON.
# Non-programmer friendly: deploy this file + requirements.txt.

import os, uuid, asyncio, re, json, hashlib
from math import sqrt
from typing import List, Dict, Tuple, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ------------ Config ------------
FILES_DIR = os.environ.get("FILES_DIR", "files")
API_KEY = os.environ.get("API_KEY", None)  # optional shared key
os.makedirs(FILES_DIR, exist_ok=True)

app = FastAPI(title="CSV Quant Preprocessor — Action Server",
              version="1.0.0",
              description="Outer-join vendor CSVs; run 18 file-pure strategies; export Excel workbook + meta JSON.")

#app.servers = [{"url": os.environ.get("BASE_URL", "http://localhost:8000")}]
app.servers = [{"url": "https://quant-preprocess.onrender.com"}]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

app.mount("/files", StaticFiles(directory=FILES_DIR), name="files")

# ------------ Models ------------
class FileRef(BaseModel):
    file_url: str = Field(..., description="Time-limited URL provided by GPT Actions")
    filename: str = Field(..., description="Original filename for labeling")

class ProcessRequest(BaseModel):
    name: str = Field("ASSET", description="Asset label")
    with_alerts: bool = Field(True, description="Include Alerts sheets")
    files: List[FileRef] = Field(..., description="One or more CSV URLs from the GPT conversation")
    api_key: Optional[str] = Field(None, description="Optional shared key (if set on server)")

# ------------ Utils ------------
STD_MAP: Dict[str, List[str]] = {
    'Date': [r'^Date$'],
    'Open': [r'^Open(_f[12])?$'],
    'High': [r'^High(_f[12])?$'],
    'Low':  [r'^Low(_f[12])?$'],
    'Close': [r'^(Adj(\.|usted)? )?Close(_f[12])?$', r'^Close(_f[12])?$'],
    'Volume': [r'^Volume(_f[12])?$'],
    'SMA50':  [r'^MA ma \(50,C,ma,0\)(_f[12])?$', r'^SMA ?50(_f[12])?$'],
    'SMA200': [r'^MA ma \(200,C,ma,0\)(_f[12])?$', r'^SMA ?200(_f[12])?$'],
    'EMA8':   [r'^MA ma \(8,C,ema,0\)(_f[12])?$', r'^EMA ?8(_f[12])?$'],
    'EMA9':   [r'^MA ma \(9,C,ema,0\)(_f[12])?$', r'^EMA ?9(_f[12])?$'],
    'EMA12':  [r'^MA ma \(12,C,ema,0\)(_f[12])?$', r'^EMA ?12(_f[12])?$'],
    'EMA25':  [r'^MA ma \(25,C,ema,0\)(_f[12])?$', r'^EMA ?25(_f[12])?$'],
    'BB_Top': [r'^Bollinger Bands Top .*\(20,C,2,ma,y\)(_f[12])?$'],
    'BB_Mid': [r'^Bollinger Bands (Median|Middle).*\(20,C,2,ma,y\)(_f[12])?$'],
    'Donch_High': [r'^Donchian High Donchian Channel \(20,20,y\)(_f[12])?$'],
    'Donch_Low':  [r'^Donchian Low Donchian Channel \(20,20,y\)(_f[12])?$'],
    'KC_Top': [r'^Keltner Top Keltner \(20,2\.00,ema,y\)(_f[12])?$'],
    'KC_Mid': [r'^Keltner Median Keltner \(20,2\.00,ema,y\)(_f[12])?$'],
    'Supertrend': [r'^Trend Supertrend \(14,3\)(_f[12])?$'],
    'ATR14': [r'^ATR ATR \(14\)(_f[12])?$'],
    'HMA55': [r'^MA ma \(55,C,hma,0\)(_f[12])?$'],
    'Ichi_Conv': [r'^Conversion Line Ichimoku Clouds \(9,26,52,26\)(_f[12])?$'],
    'Ichi_Base': [r'^Base Line Ichimoku Clouds \(9,26,52,26\)(_f[12])?$'],
    'Ichi_SpanA':[r'^Leading Span A Ichimoku Clouds \(9,26,52,26\)(_f[12])?$'],
    'Ichi_SpanB':[r'^Leading Span B Ichimoku Clouds \(9,26,52,26\)(_f[12])?$'],
    'RSI14': [r'^RSI rsi \(14\)(_f[12])?$'],
    'Stoch_%K': [r'^%K Stochastics \(C,14,n,3,3\)(_f[12])?$'],
    'Stoch_%D': [r'^%D Stochastics \(C,14,n,3,3\)(_f[12])?$'],
    'WilliamsR': [r'^Result Williams %R \(14\)(_f[12])?$'],
    'MFI14': [r'^Result M Flow \(14\)(_f[12])?$'],
    'CMF20': [r'^Result Chaikin MF \(20\)(_f[12])?$'],
    'Aroon_Up': [r'^Aroon Up \(25\)(_f[12])?$', r'^Aroon Up Aroon \(25\)(_f[12])?$'],
    'Aroon_Down':[r'^Aroon Down \(25\)(_f[12])?$', r'^Aroon Down Aroon \(25\)(_f[12])?$'],
    'ROC12': [r'^Result Price ROC \(12,C\)(_f[12])?$'],
    'Coppock': [r'^Result Coppock \(10,C,11,14\)(_f[12])?$'],
    'MACD_Line':[r'^MACD macd \(12,26,9\)(_f[12])?$'],
    'MACD_Signal':[r'^Signal macd \(12,26,9\)(_f[12])?$'],
    'VI_Plus': [r'^\+VI Vortex \(14\)(_f[12])?$'],
    'VI_Minus':[r'^-VI Vortex \(14\)(_f[12])?$'],
}

async def download_csvs(urls: List[FileRef], dest_dir: str) -> List[str]:
    paths = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for ref in urls:
            r = await client.get(ref.file_url)
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Failed to download {ref.filename}")
            p = os.path.join(dest_dir, f"{uuid.uuid4()}_{ref.filename}")
            with open(p, 'wb') as f:
                f.write(r.content)
            paths.append(p)
    return paths

def find_col(df: pd.DataFrame, patterns: List[str]) -> Tuple[Optional[str], List[str]]:
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        matches = [c for c in df.columns if rx.fullmatch(c) or rx.search(c)]
        if matches:
            def score(name):
                if name.endswith('_f1'): return (0, name)
                if name.endswith('_f2'): return (2, name)
                return (1, name)
            ms = sorted(matches, key=score)
            return ms[0], ms
    return None, []

def outer_join(files: List[str]) -> pd.DataFrame:
    dfs = []
    for i, path in enumerate(files, start=1):
        df = pd.read_csv(path)
        date_col = next((c for c in df.columns if c.strip().lower()=="date"), None)
        if date_col and date_col != 'Date':
            df = df.rename(columns={date_col:'Date'})
        if 'Date' not in df.columns:
            raise ValueError(f"No Date column in {path}")
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.rename(columns={c:(f"{c}_f{i}" if c!='Date' else c) for c in df.columns})
        dfs.append(df)
    m = dfs[0]
    for df in dfs[1:]:
        m = pd.merge(m, df, on='Date', how='outer', sort=True)
    for c in m.columns:
        if c!='Date':
            m[c] = pd.to_numeric(m[c], errors='ignore')
    return m.sort_values('Date').reset_index(drop=True)

def map_columns(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    cmap = {}
    for logical, pats in STD_MAP.items():
        found, candidates = find_col(df, pats)
        cmap[logical] = {'found': found, 'candidates': candidates[:12]}
    return cmap

# Strategy helpers

def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    a1, b1 = a.shift(1), b.shift(1)
    return (a > b) & (a1 <= b1)

def run_state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    entry = entry.fillna(False); exit_ = exit_.fillna(False)
    pos = pd.Series(0, index=entry.index, dtype=int)
    in_pos=False
    for i in range(len(entry)):
        if not in_pos and bool(entry.iat[i]): in_pos=True
        elif in_pos and bool(exit_.iat[i]): in_pos=False
        pos.iat[i] = 1 if in_pos else 0
    return pos

def evaluate(name: str, pos_raw: pd.Series, req_cols: List[str], merged: pd.DataFrame, close_col: str):
    pos = pos_raw.shift(1).fillna(0).astype(int)
    close = merged[close_col].astype(float)
    ret = close.pct_change().fillna(0.0)
    strat_ret = ret * pos
    eq = (1+strat_ret).cumprod()
    dr = strat_ret.replace([np.inf,-np.inf], np.nan).dropna()
    sharpe = (sqrt(252)*dr.mean()/dr.std(ddof=0)) if (dr.std(ddof=0) and len(dr)>1) else np.nan
    max_dd = (eq/eq.cummax()-1).min() if len(eq) else np.nan
    entries = (pos.diff()==1); exits=(pos.diff()==-1)
    e_idx=list(np.where(entries)[0]); x_idx=list(np.where(exits)[0]); pairs=[]; j=0
    for i_idx in e_idx:
        while j<len(x_idx) and x_idx[j]<i_idx: j+=1
        if j<len(x_idx): pairs.append((i_idx,x_idx[j])); j+=1
    wins=sum(1 for i,o in pairs if (eq.iloc[o]/eq.iloc[i])-1>0)
    win_rate = wins/len(pairs) if pairs else np.nan
    last_na = any(pd.isna(merged.iloc[-1][c]) for c in req_cols if c and c in merged.columns)
    current = 'Unavailable' if last_na else ('Long (enter next bar)' if bool(pos_raw.iloc[-1]) else 'Cash')
    return {
        'name':name,'total_return':float(eq.iloc[-1]-1 if len(eq) else np.nan),'win_rate':float(win_rate) if not np.isnan(win_rate) else None,
        'sharpe':float(round(sharpe,6)) if not np.isnan(sharpe) else None,'max_dd':float(max_dd) if not np.isnan(max_dd) else None,
        'trades_used':len(pairs),'current_signal':current,'eq':eq,'pos_raw':pos_raw,'pos':pos,'trades':pairs
    }

def build_strategies(merged: pd.DataFrame, cmap: Dict[str, Dict[str, object]]):
    cols = {k:v['found'] for k,v in cmap.items()}
    def c(k): return cols.get(k)
    series = {k: merged[c(k)] if c(k) in merged.columns and c(k) else pd.Series(index=merged.index,dtype=float) for k in cols}
    required_close = c('Close')
    if not required_close: raise ValueError('Close column not found.')
    s = []
    s.append(evaluate('Buy & Hold', pd.Series(1, index=merged.index, dtype=int), [required_close], merged, required_close))
    if c('SMA50') and c('SMA200'):
        s.append(evaluate('Momentum (SMA50>200)', (series['SMA50']>series['SMA200']).astype(int), [c('SMA50'),c('SMA200')], merged, required_close))
    if c('MACD_Line') and c('MACD_Signal'):
        s.append(evaluate('MACD Crossover', (series['MACD_Line']>series['MACD_Signal']).astype(int), [c('MACD_Line'),c('MACD_Signal')], merged, required_close))
    if c('Ichi_SpanA') and c('Ichi_SpanB') and c('Ichi_Conv') and c('Ichi_Base'):
        top = np.maximum(series['Ichi_SpanA'].astype(float), series['Ichi_SpanB'].astype(float))
        ichi = ((series['Close']>top)&(series['Ichi_Conv']>series['Ichi_Base'])).astype(int)
        s.append(evaluate('Ichimoku Cloud', ichi, [c('Close'),c('Ichi_SpanA'),c('Ichi_SpanB'),c('Ichi_Conv'),c('Ichi_Base')], merged, required_close))
    if c('VI_Plus') and c('VI_Minus'):
        s.append(evaluate('Vortex Cross', (series['VI_Plus']>series['VI_Minus']).astype(int), [c('VI_Plus'),c('VI_Minus')], merged, required_close))
    if c('Supertrend') and c('Close'):
        try:
            st_raw = (series['Close']>series['Supertrend']).astype(int)
        except Exception:
            st_raw = (series['Supertrend']>0).astype(int)
        s.append(evaluate('Supertrend', st_raw, [c('Close'),c('Supertrend')], merged, required_close))
    if c('Aroon_Up') and c('Aroon_Down'):
        s.append(evaluate('Aroon Cross', (series['Aroon_Up']>series['Aroon_Down']).astype(int), [c('Aroon_Up'),c('Aroon_Down')], merged, required_close))
    if c('EMA12') and c('EMA25'):
        s.append(evaluate('EMA 12>25', (series['EMA12']>series['EMA25']).astype(int), [c('EMA12'),c('EMA25')], merged, required_close))
    if c('EMA8') and c('EMA9') and c('EMA12'):
        s.append(evaluate('MA Ribbon (8>9>12)', ((series['EMA8']>series['EMA9'])&(series['EMA9']>series['EMA12'])).astype(int), [c('EMA8'),c('EMA9'),c('EMA12')], merged, required_close))
    if c('Coppock'):
        s.append(evaluate('Coppock>0', (series['Coppock']>0).astype(int), [c('Coppock')], merged, required_close))
    if c('ROC12'):
        s.append(evaluate('ROC>0', (series['ROC12']>0).astype(int), [c('ROC12')], merged, required_close))
    if c('RSI14'):
        s.append(evaluate('RSI MR (<30 → >50)', run_state((series['RSI14']<30),(series['RSI14']>50)), [c('RSI14')], merged, required_close))
    if c('Stoch_%K') and c('Stoch_%D'):
        s.append(evaluate('Stoch MR (K>D<20; exit K>50)', run_state(cross_up(series['Stoch_%K'],series['Stoch_%D'])&(series['Stoch_%K']<20)&(series['Stoch_%D']<20),(series['Stoch_%K']>50)), [c('Stoch_%K'),c('Stoch_%D')], merged, required_close))
    if c('WilliamsR'):
        s.append(evaluate('Williams %R MR (>-80; exit>-50)', run_state((series['WilliamsR']>-80)&(series['WilliamsR'].shift(1)<=-80),(series['WilliamsR']>-50)), [c('WilliamsR')], merged, required_close))
    if c('MFI14'):
        s.append(evaluate('MFI MR (<20; exit≥50)', run_state((series['MFI14']<20),(series['MFI14']>=50)), [c('MFI14')], merged, required_close))
    if c('BB_Top') and c('BB_Mid') and c('Close'):
        s.append(evaluate('Bollinger Breakout', run_state(cross_up(series['Close'],series['BB_Top']),(series['Close']<series['BB_Mid'])), [c('Close'),c('BB_Top'),c('BB_Mid')], merged, required_close))
    if c('Donch_High') and c('Donch_Low') and c('Close'):
        s.append(evaluate('Donchian Breakout', run_state(cross_up(series['Close'],series['Donch_High']),(series['Close']<series['Donch_Low'])), [c('Close'),c('Donch_High'),c('Donch_Low')], merged, required_close))
    if c('CMF20'):
        s.append(evaluate('CMF Zero (CMF>0)', (series['CMF20']>0).astype(int), [c('CMF20')], merged, required_close))
    return s, cols

def build_alerts_df(merged: pd.DataFrame, cols: Dict[str,str]):
    def c(k): return cols.get(k)
    alerts = pd.DataFrame({'Date': merged['Date']})
    if c('Close') and c('BB_Top'): alerts['Arm_BB_Upper'] = cross_up(merged[c('Close')], merged[c('BB_Top')]).fillna(False)
    else: alerts['Arm_BB_Upper']=False
    if c('Close') and c('BB_Mid'): alerts['Exit_BB_Mid'] = cross_up(merged[c('BB_Mid')], merged[c('Close')]).fillna(False)
    else: alerts['Exit_BB_Mid']=False
    if c('VI_Plus') and c('VI_Minus'): alerts['Vortex_CrossUp'] = cross_up(merged[c('VI_Plus')], merged[c('VI_Minus')]).fillna(False)
    else: alerts['Vortex_CrossUp']=False
    if c('Aroon_Up') and c('Aroon_Down'): alerts['Aroon_FlipUp'] = cross_up(merged[c('Aroon_Up')], merged[c('Aroon_Down')]).fillna(False)
    else: alerts['Aroon_FlipUp']=False
    alerts_history = alerts.loc[alerts[['Arm_BB_Upper','Exit_BB_Mid','Vortex_CrossUp','Aroon_FlipUp']].any(axis=1)].copy()
    last = merged.iloc[-1]
    def safe(col):
        return (last[col] if col in merged.columns and pd.notna(last[col]) else None)
    last_status = pd.DataFrame([{
        'Date': last['Date'], 'Close': safe(c('Close')),
        'BB_Top': safe(c('BB_Top')), 'BB_Mid': safe(c('BB_Mid')),
        'VI+': safe(c('VI_Plus')), 'VI-': safe(c('VI_Minus')),
        'Aroon_Up': safe(c('Aroon_Up')), 'Aroon_Down': safe(c('Aroon_Down')),
        'Fired_Arm_BB_Upper_Today': bool(alerts['Arm_BB_Upper'].iloc[-1]) if len(alerts) else False,
        'Fired_Exit_BB_Mid_Today': bool(alerts['Exit_BB_Mid'].iloc[-1]) if len(alerts) else False,
        'Fired_Vortex_CrossUp_Today': bool(alerts['Vortex_CrossUp'].iloc[-1]) if len(alerts) else False,
        'Fired_Aroon_FlipUp_Today': bool(alerts['Aroon_FlipUp'].iloc[-1]) if len(alerts) else False,
    }])
    recent = {}
    for col in ['Arm_BB_Upper','Exit_BB_Mid','Vortex_CrossUp','Aroon_FlipUp']:
        idx = alerts.index[alerts[col]].to_list()
        recent[col] = alerts.loc[idx[-1], 'Date'] if idx else None
    recent_fires = pd.DataFrame([recent])
    return alerts, alerts_history, last_status, recent_fires

# ------------ Routes ------------
@app.get("/")
async def root():
    return {"ok": True, "service": "CSV Quant Preprocessor — Action Server"}

from fastapi import Depends, Security
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

@app.post("/process")
async def process_csv(
    payload: dict,
    api_key: str = Depends(api_key_header)
):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")

    # your existing processing logic here...


@app.post("/process")
async def process(req: ProcessRequest):
    # optional shared secret
    if API_KEY and req.api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # download CSVs
    tmp_dir = os.path.join(FILES_DIR, str(uuid.uuid4()))
    os.makedirs(tmp_dir, exist_ok=True)
    paths = await download_csvs(req.files, tmp_dir)

    # merge + map
    merged = outer_join(paths)
    cmap = map_columns(merged)

    # audits
    dup_merged = int(merged.duplicated(subset=['Date']).sum())
    gaps_over_7 = int((merged['Date'].diff().dropna() > pd.Timedelta(days=7)).sum())
    if 'Close_f1' in merged.columns and 'Close_f2' in merged.columns:
        close_mismatch = int(((merged['Close_f1'].round(6) != merged['Close_f2'].round(6)) & merged['Close_f1'].notna() & merged['Close_f2'].notna()).sum())
    else:
        close_mismatch = -1

    # strategies
    sres, cols = build_strategies(merged, cmap)
    rows = [{
        'Strategy': s['name'], 'Total Return': s['total_return'], 'Win Rate': s['win_rate'],
        'Sharpe': s['sharpe'], 'Max Drawdown': s['max_dd'], 'Trades Used': s['trades_used'],
        'Current Signal': s['current_signal']
    } for s in sres]
    summary = pd.DataFrame(rows).sort_values(['Sharpe','Total Return'], ascending=[False, False]).reset_index(drop=True)

    # last snapshot
    last = merged.iloc[-1]
    def c(k): return cols.get(k)
    def safe(k):
        col = c(k)
        return (last[col] if col in merged.columns else np.nan)
    last_snapshot = pd.DataFrame([{
        'Date': last['Date'], 'Close': safe('Close'), 'RSI14': safe('RSI14'), 'CMF20': safe('CMF20'),
        'KC_Mid': safe('KC_Mid'), 'KC_Top': safe('KC_Top'), 'HMA55': safe('HMA55'),
        'Aroon_Up': safe('Aroon_Up'), 'Aroon_Down': safe('Aroon_Down'), 'VI+': safe('VI_Plus'), 'VI-': safe('VI_Minus'),
        'ROC12': safe('ROC12'), '%K': safe('Stoch_%K'), '%D': safe('Stoch_%D'), 'Donch_High': safe('Donch_High'), 'Donch_Low': safe('Donch_Low'),
        'BB_Top': safe('BB_Top'), 'BB_Mid': safe('BB_Mid'), 'ATR14': safe('ATR14')
    }])

    positions = pd.DataFrame({'Date': merged['Date']})
    raw = pd.DataFrame({'Date': merged['Date']})
    equity = pd.DataFrame({'Date': merged['Date']})
    for s in sres:
        positions[s['name']] = s['pos'].values
        raw[s['name']] = s['pos_raw'].values
        equity[s['name']] = s['eq'].values

    # winner trades
    winner = summary.iloc[0]['Strategy'] if not summary.empty else None
    winner_trades = pd.DataFrame()
    if winner:
        w = next(sr for sr in sres if sr['name']==winner)
        close = merged[c('Close')].astype(float)
        recs = []
        for i,o in w['trades']:
            recs.append({'EntryDate': merged['Date'].iloc[i], 'ExitDate': merged['Date'].iloc[o], 'EntryPx': float(close.iloc[i]), 'ExitPx': float(close.iloc[o]), 'Return': float((close.iloc[o]/close.iloc[i])-1)})
        winner_trades = pd.DataFrame(recs)

    # alerts (optional)
    alerts = alerts_history = last_status = recent_fires = None
    if req.with_alerts:
        alerts, alerts_history, last_status, recent_fires = build_alerts_df(merged, cols)

    # audit sheet
    audit_df = pd.DataFrame([
        {'Metric':'duplicates_merged','Value':dup_merged},
        {'Metric':'gaps_over_7_days','Value':gaps_over_7},
        {'Metric':'close_mismatches_f1_vs_f2','Value':close_mismatch},
        {'Metric':'rows_total','Value': int(merged.shape[0])},
        {'Metric':'last_date','Value': str(last['Date'].date())},
        {'Metric':'last_close','Value': float(merged[c('Close')].iloc[-1]) if c('Close') in merged.columns else None},
    ])

    # column map table
    column_map_tbl = pd.DataFrame([{ 'Logical':k, 'Found':v['found'], 'Candidates(top)': ', '.join(v['candidates']) } for k,v in cmap.items()]).sort_values('Logical')

    # write Excel
    base = f"{req.name.lower()}_{uuid.uuid4().hex}"
    xlsx_path = os.path.join(FILES_DIR, base + ".xlsx")
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as w:
        sd = summary.copy()
        sd['Total Return'] = sd['Total Return'].apply(lambda x: '' if x is None or np.isnan(x) else f"{x*100:.2f}%")
        sd['Win Rate'] = sd['Win Rate'].apply(lambda x: '' if x is None or np.isnan(x) else f"{x*100:.2f}%")
        sd['Sharpe'] = sd['Sharpe'].apply(lambda x: '' if x is None else round(x,2))
        sd['Max Drawdown'] = sd['Max Drawdown'].apply(lambda x: '' if x is None or np.isnan(x) else f"{x*100:.2f}%")
        sd['Trades Used'] = sd['Trades Used'].astype(int)
        sd.to_excel(w, sheet_name='Summary', index=False)
        audit_df.to_excel(w, sheet_name='Audit', index=False)
        column_map_tbl.to_excel(w, sheet_name='Column_Map', index=False)
        last_snapshot.to_excel(w, sheet_name='Last_Snapshot', index=False)
        positions.to_excel(w, sheet_name='Positions', index=False)
        raw.to_excel(w, sheet_name='Raw_Signals', index=False)
        equity.to_excel(w, sheet_name='Equity_Curves', index=False)
        if not winner_trades.empty:
            winner_trades.to_excel(w, sheet_name='Winner_Trades', index=False)
        if req.with_alerts and alerts is not None:
            alerts.to_excel(w, sheet_name='Alerts', index=False)
            alerts_history.to_excel(w, sheet_name='Alerts_History', index=False)
            last_status.to_excel(w, sheet_name='Last_Status', index=False)
            recent_fires.to_excel(w, sheet_name='Recent_Fires', index=False)

    # meta json
    meta = {
        'name': req.name,
        'winner': winner,
        'audits': {m['Metric']: m['Value'] for _, m in audit_df.iterrows()},
        'column_map': {k: v['found'] for k,v in cmap.items()},
        'sheets': ['Summary','Audit','Column_Map','Last_Snapshot','Positions','Raw_Signals','Equity_Curves'] + ((['Winner_Trades'] if not winner_trades.empty else []) + (['Alerts','Alerts_History','Last_Status','Recent_Fires'] if req.with_alerts else []))
    }
    meta_path = os.path.join(FILES_DIR, base + ".meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    # return URLs
    base_url = os.environ.get('BASE_URL')  # optional manual override
    if not base_url:
        # Try to infer from typical reverse proxy; otherwise, instruct user to set BASE_URL
        base_url = ''
    xlsx_url = (base_url + '/files/' + os.path.basename(xlsx_path)) if base_url else ('/files/' + os.path.basename(xlsx_path))
    meta_url = (base_url + '/files/' + os.path.basename(meta_path)) if base_url else ('/files/' + os.path.basename(meta_path))

    return {'ok': True, 'workbook_url': xlsx_url, 'meta_url': meta_url, 'winner': winner}

# ------------- OpenAPI snippet helper -------------
# The hosted OpenAPI for GPT Actions can simply be this app's live /openapi.json.
# In the GPT editor, paste the URL:  https://YOUR_DOMAIN/openapi.json
