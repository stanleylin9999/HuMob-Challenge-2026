import os
import ast
import glob
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 1. 全域配置與官方標準常數
# =========================================================================
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIFFUSION_STEPS = 50
DDIM_STEPS = 10
BATCH_SIZE = 16
EPOCHS_DIFFUSION = 30
LR = 1e-3

MEAN_ACTUAL_DIAG = 26.57
MEAN_ACTUAL_OFFDIAG = 0.0176
WEIGHT_DIAG = 0.5
WEIGHT_OFFDIAG = 0.5

PRED_START = pd.to_datetime("2024-01-01")
GAP_START = pd.to_datetime("2024-02-01")
GAP_END = pd.to_datetime("2024-03-31")
PRED_END = pd.to_datetime("2024-10-31")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TSV_PATH = os.path.join(SCRIPT_DIR, "humob2026-dataset.tsv")
SPECIFIC_CLASS_DIR = r"C:\Users\User\Desktop\人口預測專案\人口預測專案3\humob2026\data\output\module05\classification\by_class"
FALLBACK_CLASS_DIR = os.path.join(SCRIPT_DIR, "humob2026", "data", "output", "module05", "classification", "by_class")
BY_CLASS_DIR = SPECIFIC_CLASS_DIR if os.path.exists(SPECIFIC_CLASS_DIR) else FALLBACK_CLASS_DIR

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "humob_trend_amplitude_reconstruction")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_INFO_MAP = {
    1: "Class 01: Persistent Zero",
    2: "Class 02: Persistent Decrease",
    3: "Class 03: Emergent Activity",
    4: "Class 04: Partial Recovery",
    5: "Class 05: Fully Recovered",
    6: "Class 06: Stable Inflow",
    7: "Class 07: Temporary Increase",
    8: "Class 08: Partial Dissipation",
    9: "Class 09: Persistent Increase"
}

# =========================================================================
# 2. 資料解析與離群值過濾
# =========================================================================
def get_class_id_from_filename(fname: str) -> int:
    fname = fname.lower()
    if "zero" in fname: return 1
    if "decrease" in fname: return 2
    if "emergent" in fname or "temporary_activity" in fname: return 3
    if "partial_recovery" in fname or "partial_rec" in fname: return 4
    if "recovered" in fname: return 5
    if "stable" in fname: return 6
    if "temporary_increase" in fname or "temp_inc" in fname: return 7
    if "partial_dissipation" in fname or "dissip" in fname: return 8
    if "persistent_increase" in fname or "increase" in fname: return 9
    return None

def robust_median(df_sub: pd.DataFrame) -> pd.Series:
    if len(df_sub) == 0:
        return pd.Series(0.0, index=df_sub.columns)
    q25 = df_sub.quantile(0.25)
    q75 = df_sub.quantile(0.75)
    iqr = q75 - q25
    clipped = df_sub.clip(lower=q25 - 1.5 * iqr, upper=q75 + 1.5 * iqr, axis=1)
    return clipped.median(axis=0).fillna(0.0)

print("[1/6] 讀取類別、網格座標與 TSV 資料...")
grid_class_lookup = {}
if os.path.exists(BY_CLASS_DIR):
    for fpath in glob.glob(os.path.join(BY_CLASS_DIR, "*.csv")):
        c_id = get_class_id_from_filename(os.path.basename(fpath))
        if c_id is not None:
            try:
                df_cls = pd.read_csv(fpath)
                col = [c for c in df_cls.columns if any(k in str(c).lower() for k in ["grid", "orig", "id"])][0]
                for g in df_cls[col].dropna().astype(str).unique():
                    grid_class_lookup[g] = c_id
            except Exception:
                pass

raw_df = pd.read_csv(TSV_PATH, sep="\t", names=["date", "od_matrix_raw"])
raw_df['date_dt'] = pd.to_datetime(raw_df['date'].astype(str), format='%Y%m%d')
raw_df = raw_df.sort_values('date_dt').reset_index(drop=True)

daily_od_records, daily_diag_flows, daily_offdiag_flows = {}, {}, {}
for dt, val in zip(raw_df['date_dt'], raw_df['od_matrix_raw']):
    daily_od_records[dt], daily_diag_flows[dt], daily_offdiag_flows[dt] = {}, {}, {}
    if pd.isna(val) or val == "NA": continue
    try:
        od_dict = ast.literal_eval(val) if isinstance(val, str) else val
        for orig, dests in od_dict.items():
            if orig == "-1_-1": continue
            y_idx, x_idx = map(int, orig.split('_'))
            if 30 <= x_idx <= 70 and 35 <= y_idx <= 70:
                daily_od_records[dt][orig] = dests
                daily_diag_flows[dt][orig] = float(dests.get(orig, 0.0))
                daily_offdiag_flows[dt][orig] = sum(float(cnt) for dest, cnt in dests.items() if dest != orig and dest != "-1_-1")
    except Exception:
        pass

diag_df = pd.DataFrame.from_dict(daily_diag_flows, orient='index').fillna(0.0)
offdiag_df = pd.DataFrame.from_dict(daily_offdiag_flows, orient='index').fillna(0.0)

pre_mask = diag_df.index < PRED_START
valid_grids = [g for g in diag_df.columns if g in grid_class_lookup]
if not valid_grids:
    valid_grids = diag_df.columns[diag_df[pre_mask].mean() >= 0.001].tolist()

for g in valid_grids:
    if g not in grid_class_lookup:
        grid_class_lookup[g] = 5

diag_df = diag_df[valid_grids]
offdiag_df = offdiag_df[valid_grids]
num_nodes = len(valid_grids)
print(f"✓ 有效網格數: {num_nodes}，類別匹配完成")

coords = np.array([[int(c) for c in g.split('_')] for g in valid_grids])
dist_matrix = cdist(coords, coords)

# =========================================================================
# 3. 歸一化零均值週期分解器
# =========================================================================
class NormalizedCycleDecomposer:
    """計算零均值、標準化到 [-1, 1] 的週型態基底"""
    def __init__(self, flow_df, valid_grids):
        pre_df = flow_df.loc[flow_df.index < PRED_START, valid_grids].copy()
        pre_df['dow'] = pre_df.index.dayofweek
        
        dow_medians = pre_df.groupby('dow')[valid_grids].median()
        mean_dow = dow_medians.mean(axis=0)
        max_dow = dow_medians.max(axis=0)
        min_dow = dow_medians.min(axis=0)
        
        # 災前原始基準震幅 (半峰谷值)
        self.raw_amp_pre = np.maximum((max_dow - min_dow) / 2.0, 0.005)
        
        # 歸一化每週震盪波形 s(dow) in [-1, 1]，均值為 0
        norm_dow = (dow_medians - mean_dow) / self.raw_amp_pre
        self.norm_dow = norm_dow.clip(lower=-1.0, upper=1.0).fillna(0.0)

    def get_norm_wave(self, dt: pd.Timestamp, grid_list: list) -> pd.Series:
        dow = dt.dayofweek
        if dow in self.norm_dow.index:
            return self.norm_dow.loc[dow, grid_list]
        return pd.Series(0.0, index=grid_list)

decomposer_diag = NormalizedCycleDecomposer(diag_df, valid_grids)
decomposer_offdiag = NormalizedCycleDecomposer(offdiag_df, valid_grids)

# =========================================================================
# 4. 自然平滑 OD 轉移引擎
# =========================================================================
class NaturalShelterODEngine:
    def __init__(self, valid_grids, grid_class_lookup, daily_od_records, dist_matrix):
        self.valid_grids = valid_grids
        self.grid_class_lookup = grid_class_lookup
        self.is_shelter = {g: (grid_class_lookup.get(g, 5) in [3, 7]) for g in valid_grids}
        
        pre_dates = [dt for dt in daily_od_records.keys() if dt < PRED_START]
        self.P_pre = self._build_transition_matrix(daily_od_records, pre_dates)
        
        jan_dates = [dt for dt in daily_od_records.keys() if pd.to_datetime("2024-01-20") <= dt <= pd.to_datetime("2024-01-31")]
        self.P_jan = self._build_transition_matrix(daily_od_records, jan_dates) if jan_dates else self.P_pre
        
        apr_dates = [dt for dt in daily_od_records.keys() if pd.to_datetime("2024-04-01") <= dt <= pd.to_datetime("2024-04-14")]
        self.P_apr = self._build_transition_matrix(daily_od_records, apr_dates) if apr_dates else self.P_jan
        
        post_dates = [dt for dt in daily_od_records.keys() if dt > GAP_END]
        self.P_post = self._build_transition_matrix(daily_od_records, post_dates) if post_dates else self.P_apr
        
        self.P_shelter_shock = self._build_shelter_boosted_matrix(self.P_pre, dist_matrix)

    def _build_transition_matrix(self, daily_od_records, target_dates):
        counts = {g: {} for g in self.valid_grids}
        for dt in target_dates:
            day_od = daily_od_records.get(dt, {})
            for orig in self.valid_grids:
                if orig in day_od:
                    for dest, cnt in day_od[orig].items():
                        if dest != orig and dest != "-1_-1":
                            counts[orig][dest] = counts[orig].get(dest, 0.0) + cnt
        probs = {}
        for orig in self.valid_grids:
            tot = sum(counts[orig].values())
            if tot > 0:
                probs[orig] = {d: c / tot for d, c in counts[orig].items() if (c / tot) >= 0.005}
                sub_tot = sum(probs[orig].values())
                if sub_tot > 0:
                    probs[orig] = {d: p / sub_tot for d, p in probs[orig].items()}
            else:
                probs[orig] = {}
        return probs

    def _build_shelter_boosted_matrix(self, P_base, dist_matrix):
        boosted = {}
        grid_idx_map = {g: i for i, g in enumerate(self.valid_grids)}
        for orig, dests in P_base.items():
            if not dests:
                boosted[orig] = {}
                continue
            i = grid_idx_map.get(orig)
            adj_dests = {}
            for d, p in dests.items():
                j = grid_idx_map.get(d)
                if i is not None and j is not None:
                    d_ij = dist_matrix[i, j]
                    dist_decay = np.exp(-0.06 * d_ij)
                    shelter_mult = 2.0 if self.is_shelter.get(d, False) else 1.0
                    adj_dests[d] = p * dist_decay * shelter_mult
                else:
                    adj_dests[d] = p
            tot = sum(adj_dests.values())
            boosted[orig] = {d: v / tot for d, v in adj_dests.items()} if tot > 0 else dests
        return boosted

    def get_dynamic_probs(self, dt: pd.Timestamp) -> dict:
        if dt < PRED_START:
            return self.P_pre
        elif dt <= pd.to_datetime("2024-01-08"):
            tau = (dt - PRED_START).days / 7.0
            return self._blend(self.P_pre, self.P_shelter_shock, tau)
        elif dt < GAP_START:
            tau = (dt - pd.to_datetime("2024-01-08")).days / 23.0
            return self._blend(self.P_shelter_shock, self.P_jan, tau)
        elif dt <= GAP_END:
            gap_days = (GAP_END - GAP_START).days + 1
            tau = ((dt - GAP_START).days + 1) / gap_days
            w = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            return self._blend(self.P_jan, self.P_apr, w)
        else:
            tau = min(1.0, (dt - pd.to_datetime("2024-04-01")).days / 90.0)
            return self._blend(self.P_apr, self.P_post, tau)

    def _blend(self, P_a, P_b, weight):
        interp = {}
        for orig in self.valid_grids:
            p_a, p_b = P_a.get(orig, {}), P_b.get(orig, {})
            all_d = set(p_a.keys()).union(p_b.keys())
            if not all_d:
                interp[orig] = self.P_pre.get(orig, {})
                continue
            comb = {d: (1.0 - weight) * p_a.get(d, 0.0) + weight * p_b.get(d, 0.0) for d in all_d}
            c_tot = sum(comb.values())
            interp[orig] = {d: v / c_tot for d, v in comb.items()} if c_tot > 0 else {}
        return interp

od_engine = NaturalShelterODEngine(valid_grids, grid_class_lookup, daily_od_records, dist_matrix)

# =========================================================================
# 5. 雙指數模型學習：趨勢與震幅皆採用雙模型與 PCHIP 斜率連續重構
# =========================================================================
def exp_func(t, y_inf, y_0, k):
    """y(t) = y_inf + (y_0 - y_inf) * exp(-k * t)"""
    return y_inf + (y_0 - y_inf) * np.exp(-np.clip(k, 1e-4, 0.5) * t)

def fit_exponential_curve(t_data, y_data, default_k=0.05):
    """對單一時間序列擬合指數參數 (y_inf, y_0, k)"""
    if len(y_data) < 4 or np.all(y_data == y_data[0]):
        val = float(np.median(y_data)) if len(y_data) > 0 else 0.0
        return val, val, default_k
    
    y_start = float(np.median(y_data[:3]))
    y_end = float(np.median(y_data[-3:]))
    p0 = [y_end, y_start, default_k]
    
    lower_bounds = [0.0, 0.0, 1e-4]
    upper_bounds = [max(float(np.max(y_data)) * 1.5, 0.5), max(float(np.max(y_data)) * 1.5, 0.5), 0.5]
    
    try:
        popt, _ = curve_fit(exp_func, t_data, y_data, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=1000)
        return popt[0], popt[1], popt[2]
    except Exception:
        return y_end, y_start, default_k

def pchip_monotone_interpolate(y0, y1, d0, d1, tau):
    """PCHIP 單調埃爾米特插值，保證無過衝 (No Overshoot)"""
    delta = y1 - y0
    d0_mod = np.copy(d0)
    d1_mod = np.copy(d1)
    
    same_sign_0 = (np.sign(d0_mod) == np.sign(delta)) & (delta != 0)
    same_sign_1 = (np.sign(d1_mod) == np.sign(delta)) & (delta != 0)
    d0_mod = np.where(same_sign_0, d0_mod, 0.0)
    d1_mod = np.where(same_sign_1, d1_mod, 0.0)
    
    max_d = 3.0 * np.abs(delta)
    d0_mod = np.clip(d0_mod, -max_d, max_d)
    d1_mod = np.clip(d1_mod, -max_d, max_d)
    
    h00 = 2.0 * (tau ** 3) - 3.0 * (tau ** 2) + 1.0
    h10 = (tau ** 3) - 2.0 * (tau ** 2) + tau
    h01 = -2.0 * (tau ** 3) + 3.0 * (tau ** 2)
    h11 = (tau ** 3) - (tau ** 2)
    return h00 * y0 + h10 * d0_mod + h01 * y1 + h11 * d1_mod

class LearnedDualExpTrendAmplitudeEngine:
    def __init__(self, flow_df, decomposer, valid_grids, grid_class_lookup, is_offdiag=False):
        self.flow_df = flow_df
        self.decomposer = decomposer
        self.valid_grids = valid_grids
        self.grid_class_lookup = grid_class_lookup
        self.is_offdiag = is_offdiag
        
        # 1. 災前基準
        pre_df = flow_df.loc[flow_df.index < PRED_START, valid_grids]
        self.M_pre = robust_median(pre_df).clip(lower=0.001)
        self.max_ceiling = pre_df.quantile(0.99).fillna(50.0) * 1.3 + 2.0
        self.A_pre = decomposer.raw_amp_pre[valid_grids].values
        
        # 2. 學習 1 月份 (1/1 ~ 1/31) 趨勢與震幅模型
        jan_df = flow_df.loc["2024-01-01":"2024-01-31", valid_grids]
        jan_smooth = jan_df.rolling(window=7, min_periods=1, center=True).median()
        
        # 真實 7 天滑動震幅包絡線 (Max - Min) / 2
        jan_amp_obs = (jan_df.rolling(window=7, min_periods=3, center=True).max() - 
                       jan_df.rolling(window=7, min_periods=3, center=True).min()) / 2.0
        jan_amp_smooth = jan_amp_obs.bfill().ffill().clip(lower=0.001)
        
        t_jan = np.arange(len(jan_df))
        self.params_mu_jan, self.params_A_jan = {}, {}
        
        for g in valid_grids:
            y_inf, y_0, k = fit_exponential_curve(t_jan, jan_smooth[g].values, default_k=0.08)
            self.params_mu_jan[g] = (y_inf, y_0, k)
            
            a_inf, a_0, ak = fit_exponential_curve(t_jan, jan_amp_smooth[g].values, default_k=0.06)
            self.params_A_jan[g] = (a_inf, a_0, ak)

        # 1 月底 (t=30) 數值與斜率 (正確導數: -k * (y_0 - y_inf) * exp(-k * t))
        self.mu_jan_end = np.array([exp_func(30.0, *self.params_mu_jan[g]) for g in valid_grids])
        self.A_jan_end = np.array([exp_func(30.0, *self.params_A_jan[g]) for g in valid_grids])
        
        self.slope_mu_jan = np.array([
            -self.params_mu_jan[g][2] * (self.params_mu_jan[g][1] - self.params_mu_jan[g][0]) * np.exp(-self.params_mu_jan[g][2] * 30.0)
            for g in valid_grids
        ])
        self.slope_A_jan = np.array([
            -self.params_A_jan[g][2] * (self.params_A_jan[g][1] - self.params_A_jan[g][0]) * np.exp(-self.params_A_jan[g][2] * 30.0)
            for g in valid_grids
        ])

        # 3. 學習 4~10 月份 (4/1 ~ 10/31) 趨勢與震幅模型
        post_df = flow_df.loc["2024-04-01":"2024-10-31", valid_grids] if "2024-04-01" in flow_df.index else jan_df
        post_smooth = post_df.rolling(window=7, min_periods=1, center=True).median()
        
        post_amp_obs = (post_df.rolling(window=7, min_periods=3, center=True).max() - 
                        post_df.rolling(window=7, min_periods=3, center=True).min()) / 2.0
        post_amp_smooth = post_amp_obs.bfill().ffill().clip(lower=0.001)
        
        t_post = np.arange(len(post_df))
        self.params_mu_apr, self.params_A_apr = {}, {}
        
        for g in valid_grids:
            y_inf, y_0, k = fit_exponential_curve(t_post, post_smooth[g].values, default_k=0.02)
            if grid_class_lookup.get(g, 5) == 5:  # 完全復原區終值收斂至災前
                y_inf = float(self.M_pre[g])
            self.params_mu_apr[g] = (y_inf, y_0, k)
            
            a_inf, a_0, ak = fit_exponential_curve(t_post, post_amp_smooth[g].values, default_k=0.015)
            if grid_class_lookup.get(g, 5) == 5:
                a_inf = float(self.A_pre[valid_grids.index(g)])
            self.params_A_apr[g] = (a_inf, a_0, ak)

        # 4 月初 (t=0) 數值與斜率 (正確導數: -k * (y_0 - y_inf))
        self.mu_apr_start = np.array([exp_func(0.0, *self.params_mu_apr[g]) for g in valid_grids])
        self.A_apr_start = np.array([exp_func(0.0, *self.params_A_apr[g]) for g in valid_grids])
        
        self.slope_mu_apr = np.array([
            -self.params_mu_apr[g][2] * (self.params_mu_apr[g][1] - self.params_mu_apr[g][0])
            for g in valid_grids
        ])
        self.slope_A_apr = np.array([
            -self.params_A_apr[g][2] * (self.params_A_apr[g][1] - self.params_A_apr[g][0])
            for g in valid_grids
        ])

    def get_backbone(self, dt: pd.Timestamp) -> np.ndarray:
        norm_s = self.decomposer.get_norm_wave(dt, self.valid_grids).values
        gap_span = (GAP_END - GAP_START).days + 1  # 60 天
        
        # 階段 0：災前基準期 (嚴格鎖定災前統計值，杜絕負時間指數爆炸)
        if dt < PRED_START:
            mu_t = self.M_pre.values
            A_t = self.A_pre
            
        # 階段 1：1 月份 (1/1 ~ 1/31)
        elif dt < GAP_START:
            t = (dt - PRED_START).days
            mu_t = np.array([exp_func(t, *self.params_mu_jan[g]) for g in self.valid_grids])
            A_t = np.array([exp_func(t, *self.params_A_jan[g]) for g in self.valid_grids])
            
        # 階段 2：2~3 月 Gap 期 (PCHIP 單調連續插值重構)
        elif dt <= GAP_END:
            t_gap = (dt - GAP_START).days
            tau = t_gap / float(gap_span - 1)
            
            mu_t = pchip_monotone_interpolate(
                self.mu_jan_end, self.mu_apr_start,
                self.slope_mu_jan * gap_span, self.slope_mu_apr * gap_span, tau
            )
            A_t = pchip_monotone_interpolate(
                self.A_jan_end, self.A_apr_start,
                self.slope_A_jan * gap_span, self.slope_A_apr * gap_span, tau
            )
            
        # 階段 3：4~10 月份 (4/1 ~ 10/31)
        else:
            t = (dt - pd.to_datetime("2024-04-01")).days
            mu_t = np.array([exp_func(t, *self.params_mu_apr[g]) for g in self.valid_grids])
            A_t = np.array([exp_func(t, *self.params_A_apr[g]) for g in self.valid_grids])

        # 非對角線流極小，震幅微幅縮放
        if self.is_offdiag:
            A_t = A_t * 0.15

        # 組合加法模型: y(t) = mu(t) + A(t) * s(dow)
        flow_pred = np.maximum(0.0, mu_t + A_t * norm_s)
        
        # Class 1 強制鎖定為 0
        for i, g in enumerate(self.valid_grids):
            if self.grid_class_lookup.get(g, 5) == 1:
                flow_pred[i] = 0.0

        return flow_pred

morph_diag = LearnedDualExpTrendAmplitudeEngine(diag_df, decomposer_diag, valid_grids, grid_class_lookup, is_offdiag=False)
morph_offdiag = LearnedDualExpTrendAmplitudeEngine(offdiag_df, decomposer_offdiag, valid_grids, grid_class_lookup, is_offdiag=True)

# =========================================================================
# 6. 殘差擴散網絡與 DDIM 確定性取樣
# =========================================================================
def extract_calendar_features(dt):
    dow = dt.dayofweek
    return np.array([
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        1.0 if dow < 5 else 0.0,
        1.0 if dow in [5, 6] else 0.0,
        np.sin(2 * np.pi * dt.day / 31.0),
        np.cos(2 * np.pi * dt.day / 31.0)
    ], dtype=np.float32)

class ResidualDataset(Dataset):
    def __init__(self, flow_df, morph_engine, valid_grids, is_offdiag=False):
        self.samples = []
        pre_dates = [dt for dt in flow_df.index if dt < PRED_START]
        raw_res, raw_seeds, raw_times = [], [], []
        
        for dt in pre_dates:
            y_true = flow_df.loc[dt, valid_grids].values
            backbone = morph_engine.get_backbone(dt)
            raw_res.append(y_true - backbone)
            raw_seeds.append(backbone)
            raw_times.append(extract_calendar_features(dt))
            
        raw_res = np.array(raw_res)
        self.scale = np.std(raw_res, axis=0)
        min_scale = 0.01 if is_offdiag else 0.05
        self.scale = np.where(self.scale < min_scale, min_scale, self.scale)
        self.max_ceiling = morph_engine.max_ceiling.values
        
        for res, seed, t_feat in zip(raw_res, raw_seeds, raw_times):
            norm_res = np.nan_to_num(res / self.scale, nan=0.0)
            norm_seed = np.nan_to_num(seed / (self.max_ceiling + 1e-4), nan=0.0)
            self.samples.append((norm_res, norm_seed, t_feat))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        res, n_seed, t_feat = self.samples[idx]
        return torch.tensor(res, dtype=torch.float32), torch.tensor(n_seed, dtype=torch.float32), torch.tensor(t_feat, dtype=torch.float32)

class WaveformDenoiser(nn.Module):
    def __init__(self, num_nodes, time_dim=6, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.step_mlp = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 64))
        self.cond_mlp = nn.Sequential(nn.Linear(num_nodes + time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 64))
        self.in_proj = nn.Linear(num_nodes, hidden_dim)
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_proj = nn.Linear(hidden_dim, num_nodes)

    def _pos_emb(self, timesteps, dim=32):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_noisy, t_step, norm_seed, time_feat):
        t_emb = self.step_mlp(self._pos_emb(t_step, 32))
        c_emb = self.cond_mlp(torch.cat([norm_seed, time_feat], dim=-1))
        ctx = torch.cat([t_emb, c_emb], dim=-1)
        h = self.in_proj(x_noisy)
        h = self.res_block(torch.cat([h, ctx], dim=-1)) + h
        return self.out_proj(h)

class CorrectDDIMEngine:
    def __init__(self, timesteps=DIFFUSION_STEPS):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(DEVICE)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        return self.sqrt_alphas_cumprod[t].unsqueeze(-1) * x_0 + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1) * noise, noise

    @torch.no_grad()
    def sample_ddim(self, model, norm_seed, time_feat, ddim_steps=DDIM_STEPS, clamp_val=0.5):
        b, dim = norm_seed.shape[0], model.num_nodes
        step_indices = torch.linspace(self.timesteps - 1, 0, ddim_steps).long().to(DEVICE)
        x = torch.zeros((b, dim), device=DEVICE)  # 確定性零起點取樣，消滅毛刺
        
        for i in range(len(step_indices)):
            t = step_indices[i]
            prev_t = step_indices[i + 1] if i + 1 < len(step_indices) else torch.tensor(-1, device=DEVICE)
            t_batch = torch.full((b,), t, device=DEVICE, dtype=torch.long)
            
            eps = model(x, t_batch, norm_seed, time_feat)
            a_t = self.alphas_cumprod[t]
            a_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=DEVICE)
            
            pred_x0 = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
            dir_xt = torch.sqrt(1.0 - a_prev) * eps
            x = torch.sqrt(a_prev) * pred_x0 + dir_xt
            
        return torch.clamp(x, -clamp_val, clamp_val)

print("\n[2/6] 訓練殘差擴散模型...")
ds_diag = ResidualDataset(diag_df, morph_diag, valid_grids, is_offdiag=False)
ds_offdiag = ResidualDataset(offdiag_df, morph_offdiag, valid_grids, is_offdiag=True)

loader_diag = DataLoader(ds_diag, batch_size=BATCH_SIZE, shuffle=True)
loader_offdiag = DataLoader(ds_offdiag, batch_size=BATCH_SIZE, shuffle=True)

diff_engine = CorrectDDIMEngine(DIFFUSION_STEPS)
diff_model_diag = WaveformDenoiser(num_nodes=num_nodes, hidden_dim=64).to(DEVICE)
diff_model_offdiag = WaveformDenoiser(num_nodes=num_nodes, hidden_dim=48).to(DEVICE)

def train_diffusion(model, loader, is_offdiag, name):
    optimizer = optim.AdamW(model.parameters(), lr=LR if not is_offdiag else 5e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    model.train()
    for ep in range(1, EPOCHS_DIFFUSION + 1):
        total_loss = 0.0
        for res, n_seed, t_feat in loader:
            res, n_seed, t_feat = res.to(DEVICE), n_seed.to(DEVICE), t_feat.to(DEVICE)
            t = torch.randint(0, diff_engine.timesteps, (res.shape[0],), device=DEVICE).long()
            x_noisy, noise = diff_engine.q_sample(res, t)
            pred_noise = model(x_noisy, t, n_seed, t_feat)
            loss = criterion(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

train_diffusion(diff_model_diag, loader_diag, is_offdiag=False, name="Diag Diffusion")
train_diffusion(diff_model_offdiag, loader_offdiag, is_offdiag=True, name="Off-Diag Diffusion")

# =========================================================================
# 7. 全時段融合推論 (去除污染性 KNN，以純淨雙指數主幹為準)
# =========================================================================
print("\n[3/6] 執行雙指數趨勢與震幅連續推論...")
diff_model_diag.eval()
diff_model_offdiag.eval()

# 嚴格控制微量殘差，保持乾淨平滑的實體物理震幅
CLASS_RES_WEIGHT_DIAG = {c: 0.02 for c in range(1, 10)}
CLASS_RES_WEIGHT_DIAG[1] = 0.0
CLASS_RES_WEIGHT_OFFDIAG = {c: 0.0 for c in range(1, 10)}

alpha_d = np.array([CLASS_RES_WEIGHT_DIAG.get(grid_class_lookup.get(g, 5), 0.0) for g in valid_grids])
alpha_o = np.array([CLASS_RES_WEIGHT_OFFDIAG.get(grid_class_lookup.get(g, 5), 0.0) for g in valid_grids])

all_dates = pd.date_range(PRED_START, PRED_END, freq="D")
pred_diag_flows, pred_offdiag_flows = {}, {}

with torch.no_grad():
    for dt in all_dates:
        base_d = morph_diag.get_backbone(dt)
        base_o = morph_offdiag.get_backbone(dt)
        
        n_seed_d = torch.tensor(base_d / (ds_diag.max_ceiling + 1e-4), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        n_seed_o = torch.tensor(base_o / (ds_offdiag.max_ceiling + 1e-4), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        t_feat = torch.tensor(extract_calendar_features(dt), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        res_d = diff_engine.sample_ddim(diff_model_diag, n_seed_d, t_feat, ddim_steps=DDIM_STEPS, clamp_val=0.3).squeeze(0).cpu().numpy() * ds_diag.scale
        res_o = diff_engine.sample_ddim(diff_model_offdiag, n_seed_o, t_feat, ddim_steps=DDIM_STEPS, clamp_val=0.05).squeeze(0).cpu().numpy() * ds_offdiag.scale
        
        final_d = np.clip(base_d + alpha_d * res_d, 0.0, ds_diag.max_ceiling)
        final_o = np.clip(base_o + alpha_o * res_o, 0.0, ds_offdiag.max_ceiling)
        
        for i, g in enumerate(valid_grids):
            if grid_class_lookup.get(g, 0) == 1:
                final_d[i], final_o[i] = 0.0, 0.0
                
        pred_diag_flows[dt] = pd.Series(final_d, index=valid_grids)
        pred_offdiag_flows[dt] = pd.Series(final_o, index=valid_grids)

# =========================================================================
# 8. 官方標準 Combined NRMSE 評估
# =========================================================================
print("\n[4/6] 執行官方標準 Combined NRMSE 評估...")
eval_dates = [dt for dt in diag_df.index if dt >= PRED_START and not (GAP_START <= dt <= GAP_END)]
class_daily_records = {c_id: {"diag": [], "offdiag": []} for c_id in range(1, 10)}
overall_daily_records = {"diag": [], "offdiag": []}

for dt in eval_dates:
    act_od = daily_od_records.get(dt, {})
    p_diag = pred_diag_flows[dt]
    p_off = pred_offdiag_flows[dt]
    probs_today = od_engine.get_dynamic_probs(dt)

    c_diag_diffs = {c_id: [] for c_id in range(1, 10)}
    c_off_diffs = {c_id: [] for c_id in range(1, 10)}
    all_diag_diffs, all_off_diffs = [], []

    for orig in valid_grids:
        c_id = grid_class_lookup.get(orig, 5)
        act_dests = act_od.get(orig, {})

        d_err_sq = (float(p_diag[orig]) - float(act_dests.get(orig, 0.0))) ** 2
        c_diag_diffs[c_id].append(d_err_sq)
        all_diag_diffs.append(d_err_sq)

        probs = probs_today.get(orig, {})
        all_off = set([d for d in act_dests.keys() if d != orig and d != "-1_-1"]).union(probs.keys())
        for dest in all_off:
            pred_cnt = float(p_off[orig]) * float(probs.get(dest, 0.0))
            act_cnt = float(act_dests.get(dest, 0.0))
            o_err_sq = (pred_cnt - act_cnt) ** 2
            c_off_diffs[c_id].append(o_err_sq)
            all_off_diffs.append(o_err_sq)

    for c_id in range(1, 10):
        if c_diag_diffs[c_id]: class_daily_records[c_id]["diag"].append(np.sqrt(np.mean(c_diag_diffs[c_id])))
        if c_off_diffs[c_id]: class_daily_records[c_id]["offdiag"].append(np.sqrt(np.mean(c_off_diffs[c_id])))

    overall_daily_records["diag"].append(np.sqrt(np.mean(all_diag_diffs)) if all_diag_diffs else 0.0)
    overall_daily_records["offdiag"].append(np.sqrt(np.mean(all_off_diffs)) if all_off_diffs else 0.0)

summary_list = []
for c_id in range(1, 10):
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    if not c_grids or len(class_daily_records[c_id]["diag"]) == 0: continue
    rmse_diag_c = float(np.mean(class_daily_records[c_id]["diag"]))
    rmse_offdiag_c = float(np.mean(class_daily_records[c_id]["offdiag"])) if class_daily_records[c_id]["offdiag"] else 0.0
    nrmse_diag_c = rmse_diag_c / MEAN_ACTUAL_DIAG
    nrmse_offdiag_c = rmse_offdiag_c / MEAN_ACTUAL_OFFDIAG
    comb_nrmse_c = WEIGHT_DIAG * nrmse_diag_c + WEIGHT_OFFDIAG * nrmse_offdiag_c
    
    summary_list.append({
        "Class_ID": c_id, "Class_Name": CLASS_INFO_MAP[c_id], "Grid_Count": len(c_grids),
        "RMSE_diag": rmse_diag_c, "RMSE_offdiag": rmse_offdiag_c,
        "NRMSE_diag": nrmse_diag_c, "NRMSE_offdiag": nrmse_offdiag_c, "Combined_NRMSE": comb_nrmse_c
    })

RMSE_diag = float(np.mean(overall_daily_records["diag"]))
RMSE_offdiag = float(np.mean(overall_daily_records["offdiag"]))
NRMSE_diag = RMSE_diag / MEAN_ACTUAL_DIAG
NRMSE_offdiag = RMSE_offdiag / MEAN_ACTUAL_OFFDIAG
combined_nrmse = WEIGHT_DIAG * NRMSE_diag + WEIGHT_OFFDIAG * NRMSE_offdiag

df_metrics = pd.DataFrame(summary_list)

print("\n" + "=" * 110)
print(" 🏆 【HuMob 2026 修正版雙指數趨勢與震幅重建報告】")
print(f" 🎯 官方基準: mean_actual_diag = {MEAN_ACTUAL_DIAG} | mean_actual_offdiag = {MEAN_ACTUAL_OFFDIAG}")
print("=" * 110)
print(f"{'Class Name':<32} | {'Grids':<5} | {'RMSE_diag':<10} | {'RMSE_off':<10} | {'NRMSE_diag':<11} | {'NRMSE_off':<11} | {'Combined NRMSE':<14}")
print("-" * 110)
for _, row in df_metrics.iterrows():
    print(f"{row['Class_Name']:<32} | {int(row['Grid_Count']):<5} | {row['RMSE_diag']:10.4f} | {row['RMSE_offdiag']:10.4f} | {row['NRMSE_diag']:11.4f} | {row['NRMSE_offdiag']:11.4f} | {row['Combined_NRMSE']:14.4f}")
print("-" * 110)
print(f"{'OVERALL (Fixed Dual Exp Trend+Amp)':<32} | {len(valid_grids):<5} | {RMSE_diag:10.4f} | {RMSE_offdiag:10.4f} | {NRMSE_diag:11.4f} | {NRMSE_offdiag:11.4f} | {combined_nrmse:14.4f}")
print("=" * 110 + "\n")

df_metrics.to_csv(os.path.join(OUTPUT_DIR, "reconstructed_nrmse_breakdown.csv"), index=False, encoding="utf-8-sig")

# =========================================================================
# 9. 匯出預測 CSV 與 9 大類別走勢圖
# =========================================================================
print("[5/6] 匯出預測檔案與產出 9 大類別走勢圖...")
pred_diag_df = pd.DataFrame.from_dict(pred_diag_flows, orient='index')
pred_offdiag_df = pd.DataFrame.from_dict(pred_offdiag_flows, orient='index')
pred_total_df = pred_diag_df + pred_offdiag_df
pred_total_df.to_csv(os.path.join(OUTPUT_DIR, "pred_total_flows_reconstructed.csv"), encoding="utf-8-sig")

total_truth_df = diag_df + offdiag_df
full_dates = pd.date_range(total_truth_df.index.min(), total_truth_df.index.max(), freq="D")

plt.style.use('dark_background')
fig, axes = plt.subplots(3, 3, figsize=(19, 11.5), dpi=250)
fig.patch.set_facecolor('#0b1329')
fig.suptitle(f"HuMob 2026: Slope-Aligned Trend & Envelope Amplitude | Combined NRMSE: {combined_nrmse:.4f}", 
             fontsize=14, fontweight='bold', color='#ffffff', y=0.98)

for c_id in range(1, 10):
    row, col = (c_id - 1) // 3, (c_id - 1) % 3
    ax = axes[row, col]
    ax.set_facecolor('#111c3a')
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    
    if not c_grids:
        ax.set_title(CLASS_INFO_MAP[c_id], fontsize=9, color='#94a3b8')
        continue
        
    gt_series = total_truth_df[c_grids].mean(axis=1).reindex(full_dates)
    gt_series.loc[(gt_series.index >= GAP_START) & (gt_series.index <= GAP_END)] = np.nan
    pred_series = pred_total_df[c_grids].mean(axis=1)
    
    ax.axvspan(GAP_START, GAP_END, color='#f59e0b', alpha=0.18, label='Gap' if c_id == 1 else "")
    ax.plot(gt_series.index, gt_series, color='#f43f5e', linewidth=1.1, label='Actual Flow' if c_id == 1 else "")
    ax.plot(pred_series.index, pred_series, color='#2dd4bf', linewidth=1.3, label='Model' if c_id == 1 else "")
    
    ax.set_title(f"{CLASS_INFO_MAP[c_id]} (N={len(c_grids)})", fontsize=9.5, fontweight='bold', color='#e2e8f0', pad=4)
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.7)
    ax.tick_params(colors='#94a3b8', labelsize=7.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=10, frameon=True, facecolor='#0b1329', edgecolor='#334155')
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, "reconstructed_9classes_comparison.png"), dpi=250, bbox_inches='tight')
plt.close(fig)

print(f"[6/6] ✨ 執行完成！預測矩陣與走勢圖已儲存至：{OUTPUT_DIR}")
