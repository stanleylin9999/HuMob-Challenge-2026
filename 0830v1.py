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

# =========================================================================
# 1. 全域配置與官方標準常數
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TSV_PATH = os.path.join(SCRIPT_DIR, "humob2026-dataset.tsv")
SPECIFIC_CLASS_DIR = r"C:\Users\User\Desktop\人口預測專案\人口預測專案3\humob2026\data\output\module05\classification\by_class"
FALLBACK_CLASS_DIR = os.path.join(SCRIPT_DIR, "humob2026", "data", "output", "module05", "classification", "by_class")
BY_CLASS_DIR = SPECIFIC_CLASS_DIR if os.path.exists(SPECIFIC_CLASS_DIR) else FALLBACK_CLASS_DIR

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "humob_hybrid_optimal_output_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MEAN_ACTUAL_DIAG = 26.57
MEAN_ACTUAL_OFFDIAG = 0.0176
WEIGHT_DIAG = 0.5
WEIGHT_OFFDIAG = 0.5

PRED_START = pd.to_datetime("2024-01-01")
GAP_START = pd.to_datetime("2024-02-01")
GAP_END = pd.to_datetime("2024-03-31")
PRED_END = pd.to_datetime("2024-10-31")

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
# 2. 資料解析 (分離 Diag 與 Off-Diag 流動)
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

print("[1/5] 讀取類別與 TSV 數據...")
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

# =========================================================================
# 3. 空間 KNN 平滑矩陣 (超低權重，避免污染邊界)
# =========================================================================
coords = np.array([[int(c) for c in g.split('_')] for g in valid_grids])
dist_m = cdist(coords, coords)
knn_weights = np.zeros_like(dist_m)
for i in range(num_nodes):
    idx = np.argsort(dist_m[i])[:4]
    w = 1.0 / (dist_m[i, idx] + 1e-5)
    knn_weights[i, idx] = w / w.sum()
spatial_knn_weights = pd.DataFrame(knn_weights, index=valid_grids, columns=valid_grids)

# =========================================================================
# 4. 穩健週期分解器 (去除不穩定分母震盪)
# =========================================================================
class StableCycleDecomposer:
    def __init__(self, flow_df, valid_grids):
        pre_df = flow_df.loc[flow_df.index < PRED_START, valid_grids].copy()
        pre_df['dow'] = pre_df.index.dayofweek
        self.M_pre = robust_median(pre_df[valid_grids]).clip(lower=0.1)
        self.dow_medians = pre_df.groupby('dow')[valid_grids].median()

    def get_factor(self, dt: pd.Timestamp, grid_list: list) -> pd.Series:
        dow = dt.dayofweek
        pattern = self.dow_medians.loc[dow] if dow in self.dow_medians.index else self.dow_medians.median()
        factor = (pattern / self.M_pre[grid_list]).fillna(1.0).replace(0, 1.0)
        # 限制因子極端值，防止振幅暴衝
        return factor.clip(lower=0.5, upper=2.0)

decomposer_diag = StableCycleDecomposer(diag_df, valid_grids)
decomposer_offdiag = StableCycleDecomposer(offdiag_df, valid_grids)

# =========================================================================
# 5. 全歷史平滑 OD 轉移機率引擎 (無暴力截斷，保留真值連通性)
# =========================================================================
class FullHistoryODTransitionEngine:
    def __init__(self, daily_od_records, valid_grids):
        self.valid_grids = valid_grids
        
        # 提取震前全歷史與震後觀察點
        pre_dates = [dt for dt in daily_od_records.keys() if dt < PRED_START]
        self.P_pre = self._build_prob_matrix(daily_od_records, pre_dates)
        
        post_dates = [dt for dt in daily_od_records.keys() if dt > GAP_END]
        self.P_post = self._build_prob_matrix(daily_od_records, post_dates) if post_dates else self.P_pre
        
        jan_dates = [dt for dt in daily_od_records.keys() if pd.to_datetime("2024-01-10") <= dt <= pd.to_datetime("2024-01-31")]
        self.P_jan = self._build_prob_matrix(daily_od_records, jan_dates) if jan_dates else self.P_pre

    def _build_prob_matrix(self, daily_od_records, target_dates):
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
                # 保留所有歷史路徑，只做極微小的數值保護 (避免 NaN)
                probs[orig] = {d: c / tot for d, c in counts[orig].items()}
            else:
                probs[orig] = {}
        return probs

    def get_dynamic_probs(self, dt: pd.Timestamp) -> dict:
        if dt < PRED_START:
            return self.P_pre
        elif dt < GAP_START:
            tau = min(1.0, max(0.0, (dt - PRED_START).days / 30.0))
            return self._blend_probs(self.P_pre, self.P_jan, tau)
        elif dt <= GAP_END:
            tau = min(1.0, max(0.0, ((dt - GAP_START).days + 1) / 60.0))
            return self._blend_probs(self.P_jan, self.P_post, tau)
        else:
            return self.P_post

    def _blend_probs(self, P_start: dict, P_end: dict, weight: float) -> dict:
        interp = {}
        for orig in self.valid_grids:
            p_s = P_start.get(orig, {})
            p_e = P_end.get(orig, {})
            all_d = set(p_s.keys()).union(p_e.keys())
            if not all_d:
                interp[orig] = self.P_pre.get(orig, {})
                continue
            comb = {d: (1.0 - weight) * p_s.get(d, 0.0) + weight * p_e.get(d, 0.0) for d in all_d}
            c_tot = sum(comb.values())
            interp[orig] = {d: v / c_tot for d, v in comb.items()} if c_tot > 0 else self.P_pre.get(orig, {})
        return interp

dynamic_od_engine = FullHistoryODTransitionEngine(daily_od_records, valid_grids)

# =========================================================================
# 6. 保守形態學軌跡引擎 (後置嚴格歸零 + 偏差修正)
# =========================================================================
class ConservativeTrajectoryEngine:
    def __init__(self, flow_df, decomposer, valid_grids, grid_class_lookup):
        self.flow_df = flow_df
        self.decomposer = decomposer
        self.valid_grids = valid_grids
        self.grid_class_lookup = grid_class_lookup
        
        pre_df = flow_df.loc[flow_df.index < PRED_START, valid_grids]
        self.M_pre = robust_median(pre_df).clip(lower=0.1)
        self.max_ceiling = pre_df.quantile(0.99).fillna(50.0) * 1.3 + 2.0
        
        jan_sub = flow_df.loc["2024-01-20":"2024-01-31", valid_grids]
        jan_factors = pd.DataFrame([decomposer.get_factor(dt, valid_grids) for dt in jan_sub.index], index=jan_sub.index)
        self.l_jan_end = robust_median(jan_sub / jan_factors).clip(lower=0.0)
        
        self.jan_peaks = flow_df.loc["2024-01-01":"2024-01-06", valid_grids].max().fillna(self.l_jan_end)
        
        apr_sub = flow_df.loc["2024-04-01":"2024-04-14", valid_grids] if "2024-04-01" in flow_df.index else jan_sub
        apr_factors = pd.DataFrame([decomposer.get_factor(dt, valid_grids) for dt in apr_sub.index], index=apr_sub.index)
        self.l_resume_start = robust_median(apr_sub / apr_factors).clip(lower=0.0)
        
        post_sub = flow_df.loc["2024-04-01":"2024-10-31", valid_grids] if "2024-04-01" in flow_df.index else apr_sub
        post_factors = pd.DataFrame([decomposer.get_factor(dt, valid_grids) for dt in post_sub.index], index=post_sub.index)
        self.l_long_term = robust_median(post_sub / post_factors).clip(lower=0.0)

    def generate_full_series(self, all_dates: pd.DatetimeIndex) -> dict:
        gap_span = (GAP_END - GAP_START).days + 1
        pred_records = {}
        
        for dt in all_dates:
            r_t = self.decomposer.get_factor(dt, self.valid_grids)
            
            # Phase 1: 1月震後
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
                            decay_rate = 2.5
                            mu_t[g] = self.l_jan_end[g] + (p_val - self.l_jan_end[g]) * np.exp(-decay_rate * decay_tau)
                            
            # Phase 2: 2~3月 Gap 期間 (保守過渡)
            elif dt <= GAP_END:
                tau = ((dt - GAP_START).days + 1) / gap_span
                s_curve = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
                mu_t = self.l_jan_end + s_curve * (self.l_resume_start - self.l_jan_end)
                
                for g in self.valid_grids:
                    c = self.grid_class_lookup.get(g, 5)
                    if c == 1:
                        mu_t[g] = 0.0
                    elif c == 3:
                        mu_t[g] += np.sin(np.pi * tau) * 0.15 * self.l_jan_end[g]
                    elif c in [7, 8]:
                        dissip = 1.0 - np.exp(-2.5 * tau)
                        mu_t[g] = self.l_jan_end[g] + dissip * (self.l_resume_start[g] - self.l_jan_end[g])
                    elif c == 4:
                        mu_t[g] = self.l_jan_end[g] + (tau ** 2.0) * (self.l_resume_start[g] - self.l_jan_end[g])
                        
            # Phase 3: 4~10月
            else:
                tau_post = min(1.0, (dt - (GAP_END + pd.Timedelta(days=1))).days / 90.0)
                mu_t = self.l_resume_start + (1.0 - np.exp(-2.0 * tau_post)) * (self.l_long_term - self.l_resume_start)
                for g in self.valid_grids:
                    c = self.grid_class_lookup.get(g, 5)
                    if c == 1:
                        mu_t[g] = 0.0
                    elif c == 5:
                        s_post = 3.0 * (tau_post ** 2) - 2.0 * (tau_post ** 3)
                        mu_t[g] = self.l_resume_start[g] + s_post * (self.M_pre[g] - self.l_resume_start[g])

            # 基礎乘積
            raw_pred = mu_t * r_t
            
            # 超輕量平滑 (99% 本地 + 1% 近鄰)，防止模糊化真實峰值
            smooth_pred = 0.99 * raw_pred + 0.01 * spatial_knn_weights.dot(raw_pred)
            
            # 【關鍵修復：後置嚴格歸零與邊界截斷】
            final_pred = smooth_pred.clip(lower=0.0, upper=self.max_ceiling)
            for g in self.valid_grids:
                if self.grid_class_lookup.get(g, 0) == 1:
                    final_pred[g] = 0.0  # Class 1 絕對零值鎖定
                    
            pred_records[dt] = final_pred
            
        return pred_records

print("\n[2/5] 生成全時段對角線與非對角線最優軌跡...")
traj_engine_diag = ConservativeTrajectoryEngine(diag_df, decomposer_diag, valid_grids, grid_class_lookup)
traj_engine_offdiag = ConservativeTrajectoryEngine(offdiag_df, decomposer_offdiag, valid_grids, grid_class_lookup)

all_dates = pd.date_range(PRED_START, PRED_END, freq="D")
pred_diag_flows = traj_engine_diag.generate_full_series(all_dates)
pred_offdiag_flows = traj_engine_offdiag.generate_full_series(all_dates)

# =========================================================================
# 7. HuMob 官方 Combined NRMSE 評估
# =========================================================================
print("\n[3/5] 執行官方 Combined NRMSE 矩陣評估...")

eval_dates = [dt for dt in diag_df.index if dt >= PRED_START and not (GAP_START <= dt <= GAP_END)]
class_daily_records = {c_id: {"diag": [], "offdiag": []} for c_id in range(1, 10)}
overall_daily_records = {"diag": [], "offdiag": []}

for dt in eval_dates:
    act_od = daily_od_records.get(dt, {})
    p_diag = pred_diag_flows[dt]
    p_off = pred_offdiag_flows[dt]
    probs_today = dynamic_od_engine.get_dynamic_probs(dt)

    c_diag_diffs = {c_id: [] for c_id in range(1, 10)}
    c_off_diffs = {c_id: [] for c_id in range(1, 10)}
    all_diag_diffs, all_off_diffs = [], []

    for orig in valid_grids:
        c_id = grid_class_lookup.get(orig, 5)
        act_dests = act_od.get(orig, {})

        # 對角線留存流動平方差
        d_err_sq = (float(p_diag[orig]) - float(act_dests.get(orig, 0.0))) ** 2
        c_diag_diffs[c_id].append(d_err_sq)
        all_diag_diffs.append(d_err_sq)

        # 非對角線跨區流動平方差 (全聯集精確比對)
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

summary_list = []
for c_id in range(1, 10):
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    if not c_grids or len(class_daily_records[c_id]["diag"]) == 0:
        continue

    rmse_diag_c = float(np.mean(class_daily_records[c_id]["diag"]))
    rmse_offdiag_c = float(np.mean(class_daily_records[c_id]["offdiag"])) if class_daily_records[c_id]["offdiag"] else 0.0
    
    nrmse_diag_c = rmse_diag_c / MEAN_ACTUAL_DIAG
    nrmse_offdiag_c = rmse_offdiag_c / MEAN_ACTUAL_OFFDIAG
    comb_nrmse_c = WEIGHT_DIAG * nrmse_diag_c + WEIGHT_OFFDIAG * nrmse_offdiag_c

    summary_list.append({
        "Class_ID": c_id,
        "Class_Name": CLASS_INFO_MAP[c_id],
        "Grid_Count": len(c_grids),
        "RMSE_diag": rmse_diag_c,
        "RMSE_offdiag": rmse_offdiag_c,
        "NRMSE_diag": nrmse_diag_c,
        "NRMSE_offdiag": nrmse_offdiag_c,
        "Combined_NRMSE": comb_nrmse_c
    })

RMSE_diag = float(np.mean(overall_daily_records["diag"]))
RMSE_offdiag = float(np.mean(overall_daily_records["offdiag"]))
NRMSE_diag = RMSE_diag / MEAN_ACTUAL_DIAG
NRMSE_offdiag = RMSE_offdiag / MEAN_ACTUAL_OFFDIAG
combined_nrmse = WEIGHT_DIAG * NRMSE_diag + WEIGHT_OFFDIAG * NRMSE_offdiag

df_metrics = pd.DataFrame(summary_list)

print("\n" + "=" * 110)
print(" 🏆 【HuMob 2026 官方標準 Combined NRMSE 穩健修復版評估報告 (v3)】")
print(f" 🎯 官方常數: mean_actual_diag = {MEAN_ACTUAL_DIAG} | mean_actual_offdiag = {MEAN_ACTUAL_OFFDIAG}")
print("=" * 110)
print(f"{'Class Name':<32} | {'Grids':<5} | {'RMSE_diag':<10} | {'RMSE_off':<10} | {'NRMSE_diag':<11} | {'NRMSE_off':<11} | {'Combined NRMSE':<14}")
print("-" * 110)
for _, row in df_metrics.iterrows():
    print(f"{row['Class_Name']:<32} | {int(row['Grid_Count']):<5} | {row['RMSE_diag']:10.4f} | {row['RMSE_offdiag']:10.4f} | {row['NRMSE_diag']:11.4f} | {row['NRMSE_offdiag']:11.4f} | {row['Combined_NRMSE']:14.4f}")
print("-" * 110)
print(f"{'OVERALL (All Valid Grids)':<32} | {len(valid_grids):<5} | {RMSE_diag:10.4f} | {RMSE_offdiag:10.4f} | {NRMSE_diag:11.4f} | {NRMSE_offdiag:11.4f} | {combined_nrmse:14.4f}")
print("=" * 110 + "\n")

csv_path = os.path.join(OUTPUT_DIR, "class_hybrid_nrmse_breakdown.csv")
df_metrics.to_csv(csv_path, index=False, encoding="utf-8-sig")

# =========================================================================
# 8. 匯出預測 CSV 與 3x3 走勢圖
# =========================================================================
print("[4/5] 匯出預測檔案與產出 9 大類別圖譜...")
pred_diag_df = pd.DataFrame.from_dict(pred_diag_flows, orient='index')
pred_offdiag_df = pd.DataFrame.from_dict(pred_offdiag_flows, orient='index')
pred_total_df = pred_diag_df + pred_offdiag_df
pred_total_df.to_csv(os.path.join(OUTPUT_DIR, "pred_total_flows_hybrid_v3.csv"), encoding="utf-8-sig")

total_truth_df = diag_df + offdiag_df

plt.style.use('dark_background')
fig, axes = plt.subplots(3, 3, figsize=(19, 11.5), dpi=250)
fig.patch.set_facecolor('#0b1329')
fig.suptitle(f"HuMob 2026: Hybrid Optimal Prediction (v3 Clean) | Combined NRMSE: {combined_nrmse:.4f}", 
             fontsize=14, fontweight='bold', color='#ffffff', y=0.98)

for c_id in range(1, 10):
    row, col = (c_id - 1) // 3, (c_id - 1) % 3
    ax = axes[row, col]
    ax.set_facecolor('#111c3a')
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    
    if not c_grids:
        ax.set_title(CLASS_INFO_MAP[c_id], fontsize=9, color='#94a3b8')
        continue
        
    gt_series = total_truth_df[c_grids].mean(axis=1).copy()
    gt_series.loc[(gt_series.index >= GAP_START) & (gt_series.index <= GAP_END)] = np.nan
    pred_series = pred_total_df[c_grids].mean(axis=1)
    
    ax.axvspan(GAP_START, GAP_END, color='#f59e0b', alpha=0.18, label='Gap' if c_id == 1 else "")
    ax.plot(gt_series.index, gt_series, color='#f43f5e', linewidth=1.1, label='Actual Flow' if c_id == 1 else "")
    ax.plot(pred_series.index, pred_series, color='#2dd4bf', linewidth=1.3, label='Hybrid Model v3' if c_id == 1 else "")
    
    ax.set_title(f"{CLASS_INFO_MAP[c_id]} (N={len(c_grids)})", fontsize=9.5, fontweight='bold', color='#e2e8f0', pad=4)
    ax.grid(True, color='#1e293b', linestyle='--', alpha=0.7)
    ax.tick_params(colors='#94a3b8', labelsize=7.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, 
           fontsize=10, frameon=True, facecolor='#0b1329', edgecolor='#334155')
plt.tight_layout(rect=[0, 0.04, 1, 0.95])

plot_path = os.path.join(OUTPUT_DIR, "hybrid_optimal_9classes_comparison_v3.png")
plt.savefig(plot_path, dpi=250, bbox_inches='tight')
plt.close(fig)

print(f"[5/5] ✨ 執行完畢！v3 穩健修復版預測與報表已儲存至：{OUTPUT_DIR}")
