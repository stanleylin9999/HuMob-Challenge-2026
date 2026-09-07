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
from scipy.spatial.distance import cdist
from scipy.signal import hilbert, savgol_filter
from scipy.fft import rfft, rfftfreq
import pywt

# =========================================================================
# 1. 全域配置與官方標準常數
# =========================================================================
def seed_everything(seed=42):
    np.random.seed(seed)

seed_everything(42)

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

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "humob_wavelet_dual_amplitude_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHORT_CLASS_NAMES = {
    1: "Persistent Zero",
    2: "Persistent Decrease",
    3: "Emergent / Temporary Activity",
    4: "Partial Recovery",
    5: "Fully Recovered",
    6: "Stable Inflow",
    7: "Temporary Increase",
    8: "Partial Dissipation",
    9: "Persistent Increase"
}

# =========================================================================
# 2. 資料解析與空間拓撲
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

print("[1/6] 讀取類別標籤與每日 OD 矩陣資料...")
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
print(f"✓ 有效網格數: {num_nodes}，空間關聯建置完成")

coords = np.array([[int(c) for c in g.split('_')] for g in valid_grids])
dist_matrix = cdist(coords, coords)
knn_weights = np.zeros_like(dist_matrix)
for i in range(num_nodes):
    neighbors = np.argsort(dist_matrix[i])[1:5]
    w = 1.0 / np.maximum(dist_matrix[i, neighbors], 0.5)
    knn_weights[i, neighbors] = w / w.sum()
spatial_knn = pd.DataFrame(knn_weights, index=valid_grids, columns=valid_grids)

# =========================================================================
# 3. 自然平滑 OD 轉移引擎
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
# 4. 趨勢引擎（宏觀骨架，保持自主推演）
# =========================================================================
class AdaptiveMacroTrendEngine:
    def __init__(self, flow_df, valid_grids, grid_class_lookup):
        self.flow_df = flow_df
        self.valid_grids = valid_grids
        self.grid_class_lookup = grid_class_lookup
        
        pre_df = flow_df.loc[flow_df.index < PRED_START, valid_grids]
        self.M_pre = robust_median(pre_df).clip(lower=0.1)
        self.max_ceiling = pre_df.quantile(0.99).fillna(50.0) * 1.3 + 2.0
        
        jan_sub = flow_df.loc["2024-01-20":"2024-01-31", valid_grids]
        self.l_jan_end = robust_median(jan_sub).clip(lower=0.0)
        self.jan_peaks = flow_df.loc["2024-01-01":"2024-01-06", valid_grids].max().fillna(self.l_jan_end)
        
        apr_sub = flow_df.loc["2024-04-01":"2024-04-14", valid_grids] if "2024-04-01" in flow_df.index else jan_sub
        self.l_resume_start = robust_median(apr_sub).clip(lower=0.0)
        
        post_sub = flow_df.loc["2024-04-01":"2024-10-31", valid_grids] if "2024-04-01" in flow_df.index else apr_sub
        self.l_long_term = robust_median(post_sub).clip(lower=0.0)

    def get_macro_trend(self, dt: pd.Timestamp) -> pd.Series:
        gap_span = (GAP_END - GAP_START).days + 1
        
        if dt < GAP_START:
            day_idx = (dt - PRED_START).days
            tau = day_idx / 30.0
            jan_init = self.flow_df.loc["2024-01-01":"2024-01-04", self.valid_grids].median().fillna(self.l_jan_end)
            mu_t = jan_init + (tau ** 1.2) * (self.l_jan_end - jan_init)
            
            for g in self.valid_grids:
                c = self.grid_class_lookup.get(g, 5)
                p_val = self.jan_peaks[g]
                if c in [3, 7, 8]:
                    if day_idx <= 3:
                        mu_t[g] = jan_init[g] + (p_val - jan_init[g]) * (day_idx / 3.0)
                    else:
                        decay_tau = (day_idx - 3) / 27.0
                        mu_t[g] = self.l_jan_end[g] + (p_val - self.l_jan_end[g]) * np.exp(-2.8 * decay_tau)
                        
        elif dt <= GAP_END:
            tau = ((dt - GAP_START).days + 1) / gap_span
            s_curve = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            mu_t = self.l_jan_end + s_curve * (self.l_resume_start - self.l_jan_end)
            
            for g in self.valid_grids:
                c = self.grid_class_lookup.get(g, 5)
                if c == 1: mu_t[g] = 0.0
                elif c == 3: mu_t[g] += np.sin(np.pi * tau) * 0.12 * self.l_jan_end[g]
                elif c in [7, 8]: mu_t[g] = self.l_jan_end[g] + (1.0 - np.exp(-2.5 * tau)) * (self.l_resume_start[g] - self.l_jan_end[g])
                elif c == 4: mu_t[g] = self.l_jan_end[g] + (tau ** 2.0) * (self.l_resume_start[g] - self.l_jan_end[g])
                
        else:
            tau_post = min(1.0, (dt - (GAP_END + pd.Timedelta(days=1))).days / 90.0)
            mu_t = self.l_resume_start + (1.0 - np.exp(-2.0 * tau_post)) * (self.l_long_term - self.l_resume_start)
            for g in self.valid_grids:
                c = self.grid_class_lookup.get(g, 5)
                if c == 1: mu_t[g] = 0.0
                elif c == 5:
                    s_post = 3.0 * (tau_post ** 2) - 2.0 * (tau_post ** 3)
                    mu_t[g] = self.l_resume_start[g] + s_post * (self.M_pre[g] - self.l_resume_start[g])

        return np.maximum(0.0, mu_t)

trend_diag_engine = AdaptiveMacroTrendEngine(diag_df, valid_grids, grid_class_lookup)
trend_offdiag_engine = AdaptiveMacroTrendEngine(offdiag_df, valid_grids, grid_class_lookup)

# =========================================================================
# 5. 大小振幅解耦、週期分析與小波多解析度預測引擎
# =========================================================================
class DualAmplitudeWaveletForecaster:
    def __init__(self, flow_df, trend_engine, valid_grids, is_offdiag=False):
        self.flow_df = flow_df
        self.trend_engine = trend_engine
        self.valid_grids = valid_grids
        self.is_offdiag = is_offdiag
        
        self.large_periods = {}
        self.small_periods = {}
        self.A_large_pre = {}
        self.A_large_post = {}
        self.phase_large = {}
        self.wavelet_components_pred = {}
        
        self._analyze_and_fit()

    def _analyze_and_fit(self):
        pre_dates = [dt for dt in self.flow_df.index if dt < PRED_START]
        post_dates = [dt for dt in self.flow_df.index if dt > GAP_END]
        
        pre_df = self.flow_df.loc[pre_dates, self.valid_grids]
        post_df = self.flow_df.loc[post_dates, self.valid_grids] if post_dates else pre_df
        
        t_pre = np.array([(dt - PRED_START).days for dt in pre_dates])
        t_post = np.array([(dt - PRED_START).days for dt in post_dates])
        
        flow_tag = "Off-Diagonal" if self.is_offdiag else "Diagonal"
        print(f" -> 執行大小振幅解耦、週期辨識與小波多解析度建模 [{flow_tag}]...")
        
        for g in self.valid_grids:
            y_pre = pre_df[g].values
            y_post = post_df[g].values if len(post_df) > 0 else y_pre
            
            # 1. 大振幅萃取與週期分析 (主週期 T_large)
            w_win = min(35, len(y_post) if len(y_post) % 2 == 1 else len(y_post) - 1)
            macro_post = savgol_filter(y_post, window_length=max(7, w_win), polyorder=1)
            detrend_post = y_post - macro_post
            
            fft_vals = np.abs(rfft(detrend_post))
            fft_freqs = rfftfreq(len(detrend_post), d=1.0)
            if len(fft_vals) > 1:
                top_idx = np.argmax(fft_vals[1:]) + 1
                dom_freq = fft_freqs[top_idx]
                T_large = (1.0 / dom_freq) if dom_freq > 0.05 else 7.0
            else:
                T_large = 7.0
            
            self.large_periods[g] = round(float(T_large), 2)
            omega_large = 2 * np.pi / 7.0
            
            X_large = np.column_stack([np.cos(omega_large * t_post), np.sin(omega_large * t_post)])
            c_l, _, _, _ = np.linalg.lstsq(X_large, detrend_post, rcond=None)
            amp_large_post = np.sqrt(c_l[0]**2 + c_l[1]**2)
            phi_large = np.arctan2(-c_l[1], c_l[0])
            
            detrend_pre = y_pre - np.median(y_pre)
            X_pre = np.column_stack([np.cos(omega_large * t_pre), np.sin(omega_large * t_pre)])
            c_pre, _, _, _ = np.linalg.lstsq(X_pre, detrend_pre, rcond=None)
            amp_large_pre = np.sqrt(c_pre[0]**2 + c_pre[1]**2)
            
            self.A_large_pre[g] = float(amp_large_pre)
            self.A_large_post[g] = float(amp_large_post)
            self.phase_large[g] = float(phi_large)
            
            # 2. 小振幅獨立：殘差分離與次週期分析 (次週期 T_small)
            w_large_curve = c_l[0] * np.cos(omega_large * t_post) + c_l[1] * np.sin(omega_large * t_post)
            r_small = detrend_post - w_large_curve
            
            fft_small = np.abs(rfft(r_small))
            if len(fft_small) > 2:
                sub_idx = np.argmax(fft_small[2:]) + 2
                sub_freq = fft_freqs[sub_idx]
                T_small = (1.0 / sub_freq) if sub_freq > 0.1 else 3.5
            else:
                T_small = 3.5
            self.small_periods[g] = round(float(T_small), 2)
            
            # 3. 小振幅小波多解析度分析 (PyWavelets MRA) 與預測外推
            eta = r_small / np.maximum(macro_post, 0.1)
            level = min(3, pywt.dwt_max_level(len(eta), 'sym4'))
            if level >= 1:
                coeffs = pywt.wavedec(eta, 'sym4', level=level)
                reconstructed_details = []
                for i in range(1, len(coeffs)):
                    zero_coeffs = [np.zeros_like(c) for c in coeffs]
                    zero_coeffs[i] = coeffs[i]
                    d_signal = pywt.waverec(zero_coeffs, 'sym4')[:len(eta)]
                    reconstructed_details.append(d_signal)
                
                dow_post = np.array([dt.dayofweek for dt in post_dates])
                wavelet_dow_profile = np.zeros(7)
                for d in range(7):
                    mask = (dow_post == d)
                    if np.any(mask):
                        val = sum(np.median(layer[mask]) for layer in reconstructed_details)
                        wavelet_dow_profile[d] = val
                
                self.wavelet_components_pred[g] = wavelet_dow_profile
            else:
                self.wavelet_components_pred[g] = np.zeros(7)

    def forecast_day(self, dt: pd.Timestamp) -> pd.Series:
        mu_series = self.trend_engine.get_macro_trend(dt)
        day_t = (dt - PRED_START).days
        dow = dt.dayofweek
        
        if dt < GAP_START:
            blend_post = 0.0
        elif dt <= GAP_END:
            tau = ((dt - GAP_START).days + 1) / ((GAP_END - GAP_START).days + 1)
            blend_post = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
        else:
            blend_post = 1.0

        pred_vals = np.zeros(len(self.valid_grids))
        omega_large = 2 * np.pi / 7.0
        
        for i, g in enumerate(self.valid_grids):
            c = grid_class_lookup.get(g, 5)
            mu_val = float(mu_series[g])
            
            # Class 1 絕對零值鎖定，且非對角線時 Class 3 強制歸零
            if c == 1 or (self.is_offdiag and c == 3) or mu_val <= 1e-4:
                pred_vals[i] = 0.0
                continue
                
            # 1. 大振幅預測
            A_l = (1.0 - blend_post) * self.A_large_pre[g] + blend_post * self.A_large_post[g]
            phi_l = self.phase_large[g]
            W_large = A_l * np.cos(omega_large * day_t + phi_l)
            
            # 2. 小振幅小波預測
            eta_pred = self.wavelet_components_pred[g][dow]
            scale_damp = 0.4 if self.is_offdiag else 0.85
            W_small_wavelet = mu_val * eta_pred * scale_damp
            
            # 3. 最終預測流合成
            total_pred = mu_val + W_large + W_small_wavelet
            pred_vals[i] = max(0.0, total_pred)
            
        return pd.Series(pred_vals, index=self.valid_grids)

# 實例化雙軌分解器
decomposer_diag = DualAmplitudeWaveletForecaster(diag_df, trend_diag_engine, valid_grids, is_offdiag=False)
decomposer_offdiag = DualAmplitudeWaveletForecaster(offdiag_df, trend_offdiag_engine, valid_grids, is_offdiag=True)

# 印出各類別的週期分析報告
print("\n" + "=" * 80)
print(" 🔍 【HuMob 2026 大小振幅週期分析報告】")
print("=" * 80)
print(f"{'Class Name':<35} | {'大振幅主週期 (天)':<18} | {'小振幅次週期 (天)':<18}")
print("-" * 80)
for c_id in range(1, 10):
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    if not c_grids: continue
    t_l_mean = np.mean([decomposer_diag.large_periods[g] for g in c_grids])
    t_s_mean = np.mean([decomposer_diag.small_periods[g] for g in c_grids])
    print(f"{SHORT_CLASS_NAMES[c_id]:<35} | {t_l_mean:18.2f} | {t_s_mean:18.2f}")
print("=" * 80 + "\n")

# =========================================================================
# 6. 全時段融合推論
# =========================================================================
print("[2/6] 執行全時段 (1~10月) 預測合成與空間 KNN 平滑...")
all_dates = pd.date_range(PRED_START, PRED_END, freq="D")
pred_diag_flows, pred_offdiag_flows = {}, {}

for dt in all_dates:
    p_d = decomposer_diag.forecast_day(dt)
    p_o = decomposer_offdiag.forecast_day(dt)
    
    # 空間 KNN 平滑
    smooth_d = 0.98 * p_d.values + 0.02 * spatial_knn.dot(p_d.values).values
    smooth_o = 0.99 * p_o.values + 0.01 * spatial_knn.dot(p_o.values).values
    
    final_d = np.clip(smooth_d, 0.0, trend_diag_engine.max_ceiling.values)
    final_o = np.clip(smooth_o, 0.0, trend_offdiag_engine.max_ceiling.values)
    
    for i, g in enumerate(valid_grids):
        c_id = grid_class_lookup.get(g, 0)
        if c_id == 1:
            final_d[i] = 0.0
            final_o[i] = 0.0
        elif c_id == 3:
            final_o[i] = 0.0  # 強制將 Class 3 的 offdiag 鎖定為 0
            
    pred_diag_flows[dt] = pd.Series(final_d, index=valid_grids)
    pred_offdiag_flows[dt] = pd.Series(final_o, index=valid_grids)

# =========================================================================
# 7. 官方標準 Combined NRMSE 評估與表格產出
# =========================================================================
print("[3/6] 執行官方標準 Combined NRMSE 評估...")
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

        # 對角線誤差
        d_err_sq = (float(p_diag[orig]) - float(act_dests.get(orig, 0.0))) ** 2
        c_diag_diffs[c_id].append(d_err_sq)
        all_diag_diffs.append(d_err_sq)

        # 非對角線誤差
        probs = probs_today.get(orig, {})
        all_off = set([d for d in act_dests.keys() if d != orig and d != "-1_-1"]).union(probs.keys())
        for dest in all_off:
            pred_cnt = float(p_off[orig]) * float(probs.get(dest, 0.0))
            act_cnt = float(act_dests.get(dest, 0.0))
            o_err_sq = (pred_cnt - act_cnt) ** 2
            c_off_diffs[c_id].append(o_err_sq)
            all_off_diffs.append(o_err_sq)

    for c_id in range(1, 10):
        if c_diag_diffs[c_id]:
            class_daily_records[c_id]["diag"].append(np.sqrt(np.mean(c_diag_diffs[c_id])))
        if c_off_diffs[c_id]:
            class_daily_records[c_id]["offdiag"].append(np.sqrt(np.mean(c_off_diffs[c_id])))

    overall_daily_records["diag"].append(np.sqrt(np.mean(all_diag_diffs)) if all_diag_diffs else 0.0)
    overall_daily_records["offdiag"].append(np.sqrt(np.mean(all_off_diffs)) if all_off_diffs else 0.0)

summary_rows = []
for c_id in range(1, 10):
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    grid_cnt = len(c_grids)
    if grid_cnt == 0 or len(class_daily_records[c_id]["diag"]) == 0:
        continue
    
    rmse_diag_c = float(np.mean(class_daily_records[c_id]["diag"]))
    rmse_offdiag_c = float(np.mean(class_daily_records[c_id]["offdiag"])) if class_daily_records[c_id]["offdiag"] else 0.0
    
    nrmse_diag_c = rmse_diag_c / MEAN_ACTUAL_DIAG
    nrmse_offdiag_c = rmse_offdiag_c / MEAN_ACTUAL_OFFDIAG
    comb_nrmse_c = WEIGHT_DIAG * nrmse_diag_c + WEIGHT_OFFDIAG * nrmse_offdiag_c
    
    summary_rows.append({
        "class id": f"class {c_id}",
        "class name": f"{SHORT_CLASS_NAMES[c_id]} ({grid_cnt}格)",
        "NRMSE_diag": round(nrmse_diag_c, 2),
        "NRMSE_off": round(nrmse_offdiag_c, 2),
        "combined NRMSE": round(comb_nrmse_c, 2)
    })

RMSE_diag = float(np.mean(overall_daily_records["diag"]))
RMSE_offdiag = float(np.mean(overall_daily_records["offdiag"]))
NRMSE_diag = RMSE_diag / MEAN_ACTUAL_DIAG
NRMSE_offdiag = RMSE_offdiag / MEAN_ACTUAL_OFFDIAG
combined_nrmse = WEIGHT_DIAG * NRMSE_diag + WEIGHT_OFFDIAG * NRMSE_offdiag

summary_rows.append({
    "class id": "total",
    "class name": f"({len(valid_grids)}格)",
    "NRMSE_diag": round(NRMSE_diag, 2),
    "NRMSE_off": round(NRMSE_offdiag, 2),
    "combined NRMSE": round(combined_nrmse, 2)
})

df_table = pd.DataFrame(summary_rows)

print("\n" + "=" * 85)
print(f" 🏆 HuMob 2026 小波多解析度模型評估表格 (Combined NRMSE: {combined_nrmse:.4f})")
print(f" 🎯 基準常數: mean_diag = {MEAN_ACTUAL_DIAG} | mean_off = {MEAN_ACTUAL_OFFDIAG}")
print("=" * 85)
print(f"{'class id':<10} | {'class name':<40} | {'NRMSE_diag':<10} | {'NRMSE_off':<10} | {'combined NRMSE':<14}")
print("-" * 85)
for _, r in df_table.iterrows():
    print(f"{r['class id']:<10} | {r['class name']:<40} | {r['NRMSE_diag']:<10.2f} | {r['NRMSE_off']:<10.2f} | {r['combined NRMSE']:<14.2f}")
print("=" * 85 + "\n")

table_path = os.path.join(OUTPUT_DIR, "nrmse_summary_table.csv")
df_table.to_csv(table_path, index=False, encoding="utf-8-sig")
print(f"✓ 表格已匯出至: {table_path}")

# =========================================================================
# 8. 匯出預測 CSV 與 9 大類別走勢圖（對角線、非對角線與總流）
# =========================================================================
print("[4/6] 匯出預測 CSV 檔案...")
pred_diag_df = pd.DataFrame.from_dict(pred_diag_flows, orient='index')
pred_offdiag_df = pd.DataFrame.from_dict(pred_offdiag_flows, orient='index')
pred_total_df = pred_diag_df + pred_offdiag_df

pred_diag_df.to_csv(os.path.join(OUTPUT_DIR, "pred_diag_flows.csv"), encoding="utf-8-sig")
pred_offdiag_df.to_csv(os.path.join(OUTPUT_DIR, "pred_offdiag_flows.csv"), encoding="utf-8-sig")
pred_total_df.to_csv(os.path.join(OUTPUT_DIR, "pred_total_flows_fixed.csv"), encoding="utf-8-sig")

total_truth_df = diag_df + offdiag_df
full_dates = pd.date_range(total_truth_df.index.min(), total_truth_df.index.max(), freq="D")

def plot_9classes_flow(truth_flow_df, pred_flow_df, flow_title, save_filename, metric_val):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 3, figsize=(19, 11.5), dpi=250)
    fig.patch.set_facecolor('#0b1329')
    
    fig.suptitle(f"HuMob 2026: Dual Amplitude + Wavelet MRA ({flow_title}) | NRMSE: {metric_val:.4f}", 
                 fontsize=14, fontweight='bold', color='#ffffff', y=0.98)

    for c_id in range(1, 10):
        row, col = (c_id - 1) // 3, (c_id - 1) % 3
        ax = axes[row, col]
        ax.set_facecolor('#111c3a')
        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
        
        if not c_grids:
            ax.set_title(f"Class {c_id:02d}: Empty", fontsize=9, color='#94a3b8')
            continue
            
        gt_series = truth_flow_df[c_grids].mean(axis=1).reindex(full_dates)
        gt_series.loc[(gt_series.index >= GAP_START) & (gt_series.index <= GAP_END)] = np.nan
        pred_series = pred_flow_df[c_grids].mean(axis=1)
        
        ax.axvspan(GAP_START, GAP_END, color='#f59e0b', alpha=0.18, label='Gap' if c_id == 1 else "")
        ax.plot(gt_series.index, gt_series, color='#f43f5e', linewidth=1.1, label='Actual Flow' if c_id == 1 else "")
        ax.plot(pred_series.index, pred_series, color='#2dd4bf', linewidth=1.3, label='Wavelet Model' if c_id == 1 else "")
        
        ax.set_title(f"Class {c_id:02d}: {SHORT_CLASS_NAMES[c_id]} (N={len(c_grids)})", 
                     fontsize=9.5, fontweight='bold', color='#e2e8f0', pad=4)
        ax.grid(True, color='#1e293b', linestyle='--', alpha=0.7)
        ax.tick_params(colors='#94a3b8', labelsize=7.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=10, 
               frameon=True, facecolor='#0b1329', edgecolor='#334155')
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 走勢圖已儲存: {save_filename}")

print("[5/6] 產出 9 大類別視覺化走勢圖...")
plot_9classes_flow(diag_df, pred_diag_df, "Diagonal Flow", "wavelet_9classes_diag.png", NRMSE_diag)
plot_9classes_flow(offdiag_df, pred_offdiag_df, "Off-Diagonal Flow", "wavelet_9classes_offdiag.png", NRMSE_offdiag)
plot_9classes_flow(total_truth_df, pred_total_df, "Total Flow", "wavelet_9classes_total.png", combined_nrmse)

print(f"\n[6/6] ✨ 執行完成！週期報告、預測數據與小波圖表已儲存至：{OUTPUT_DIR}")
