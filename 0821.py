import os
import re
import ast
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial.distance import cdist

# =========================================================================
# 1. 環境與全域設定
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TSV_PATH = os.path.join(SCRIPT_DIR, "humob2026-dataset.tsv")
BY_CLASS_DIR = os.path.join(SCRIPT_DIR, "humob2026", "data", "output", "module05", "classification", "by_class")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "humob_pipeline_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_INFO_MAP = {
    1: "Class 01: Persistent Zero",
    2: "Class 02: Persistent Decrease",
    3: "Class 03: Emergent / Temporary Activity",
    4: "Class 04: Partial Recovery",
    5: "Class 05: Fully Recovered",
    6: "Class 06: Stable Inflow",
    7: "Class 07: Temporary Increase",
    8: "Class 08: Partial Dissipation",
    9: "Class 09: Persistent Increase"
}

GAP_START = pd.to_datetime("2024-02-01")
GAP_END = pd.to_datetime("2024-03-31")
PRED_START = pd.to_datetime("2024-01-01")
PRED_END = pd.to_datetime("2024-10-31")

print("=" * 80)
print("🚀 HuMob 動態非線性預測流程：類別形態學優化 (Class 07 消退修復) ➔ NRMSE 評估")
print(f"📁 缺測斷線區間: {GAP_START.strftime('%Y-%m-%d')} ~ {GAP_END.strftime('%Y-%m-%d')}")
print("=" * 80)

# =========================================================================
# 2. 解析 9 大類別網格與 TSV 全時空資料
# =========================================================================
print("\n[1/5] 解析 9 大類別標籤與 TSV 全時空人流資料...")

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

grid_class_lookup = {}
if os.path.exists(BY_CLASS_DIR):
    for fpath in glob.glob(os.path.join(BY_CLASS_DIR, "*.csv")):
        fname = os.path.basename(fpath)
        c_id = get_class_id_from_filename(fname)
        if c_id is not None:
            try:
                df_cls = pd.read_csv(fpath)
                col = [c for c in df_cls.columns if any(k in str(c).lower() for k in ["grid", "orig", "id"])][0]
                grids_in_file = df_cls[col].dropna().astype(str).unique()
                for g in grids_in_file:
                    grid_class_lookup[g] = c_id
            except Exception as e:
                print(f"  ⚠️ 讀取失敗 {fname}: {e}")

raw_df = pd.read_csv(TSV_PATH, sep="\t", names=["date", "od_matrix_raw"])
raw_df['date_dt'] = pd.to_datetime(raw_df['date'].astype(str), format='%Y%m%d')
raw_df = raw_df.sort_values('date_dt').reset_index(drop=True)

daily_od_records, daily_grid_flows = {}, {}
for dt, val in zip(raw_df['date_dt'], raw_df['od_matrix_raw']):
    daily_od_records[dt], daily_grid_flows[dt] = {}, {}
    if pd.isna(val) or val == "NA":
        continue
    try:
        od_dict = ast.literal_eval(val) if isinstance(val, str) else val
        for orig, dests in od_dict.items():
            if orig == "-1_-1": continue
            y_idx, x_idx = map(int, orig.split('_'))
            if 30 <= x_idx <= 70 and 35 <= y_idx <= 70:
                daily_od_records[dt][orig] = dests
                daily_grid_flows[dt][orig] = sum(dests.values())
    except Exception:
        pass

flow_df = pd.DataFrame.from_dict(daily_grid_flows, orient='index').fillna(0.0)
valid_grids = [g for g in flow_df.columns if g in grid_class_lookup]
if not valid_grids:
    pre_mask_temp = flow_df.index < PRED_START
    valid_grids = flow_df.columns[flow_df[pre_mask_temp].mean() >= 0.001].tolist()

flow_df = flow_df[valid_grids]
pre_mask = flow_df.index < PRED_START

# =========================================================================
# 3. 震前特徵分解、空間矩陣與動態波動注入
# =========================================================================
print("\n[2/5] 提取週期特徵、計算殘差標準差並生成非平緩動態預測...")

pre_df = flow_df[pre_mask].copy()
pre_df['dow'] = pre_df.index.dayofweek
pre_df['week_of_month'] = (pre_df.index.day - 1) // 7 + 1

# 1. 強健週期模式
clean_records = []
for (wom, dow), group in pre_df.groupby(['week_of_month', 'dow']):
    grp_grids = group[valid_grids]
    q25, q75 = grp_grids.quantile(0.25), grp_grids.quantile(0.75)
    iqr = q75 - q25
    clipped = grp_grids.clip(lower=q25 - 1.5 * iqr, upper=q75 + 1.5 * iqr, axis=1)
    center = clipped.median(axis=0)
    center['week_of_month'], center['dow'] = wom, dow
    clean_records.append(center)

robust_cycle_patterns = pd.DataFrame(clean_records).set_index(['week_of_month', 'dow'])
M_pre_robust = pre_df[valid_grids].median().replace(0, 1.0)
dow_medians_pre = pre_df.groupby('dow')[valid_grids].median()
max_pre_allowable = pre_df[valid_grids].quantile(0.98) * 1.6 + 5.0  # 防極端值上限天花板

def get_cycle_factor(dt, grid_list):
    wom = min((dt.day - 1) // 7 + 1, 4)
    dow = dt.dayofweek
    if (wom, dow) in robust_cycle_patterns.index:
        pattern = robust_cycle_patterns.loc[(wom, dow)][grid_list]
    else:
        pattern = dow_medians_pre.loc[dow]
    factor = pattern / M_pre_robust[grid_list]
    return factor.replace(0, 1.0).fillna(1.0)

# 2. 空間 KNN 矩陣
coords = np.array([[int(c) for c in g.split('_')] for g in valid_grids])
dist_m = cdist(coords, coords)
knn_weights = np.zeros_like(dist_m)
for i in range(len(valid_grids)):
    idx = np.argsort(dist_m[i])[:5]
    w = 1.0 / (dist_m[i, idx] + 1e-5)
    knn_weights[i, idx] = w / w.sum()
spatial_knn_weights = pd.DataFrame(knn_weights, index=valid_grids, columns=valid_grids)

# 3. 歷史殘差與波動度估計
pre_cycles = pd.DataFrame([get_cycle_factor(dt, valid_grids) for dt in pre_df.index], index=pre_df.index)
pre_expected = pre_cycles.multiply(M_pre_robust, axis=1)
historical_residuals = pre_df[valid_grids] - pre_expected
grid_volatility = historical_residuals.std(axis=0).clip(lower=0.05, upper=25.0)

# 4. 各類別專屬錨點與遷移路徑
jan_sub = flow_df.loc["2024-01-25":"2024-01-31"]
jan_factors = pd.DataFrame([get_cycle_factor(dt, valid_grids) for dt in jan_sub.index], index=jan_sub.index)
l_jan_end = (jan_sub / jan_factors).median().fillna(M_pre_robust)

# 預先提取 1 月初峰值 (用於 Class 7 震後激增包絡線)
jan_peaks = flow_df.loc["2024-01-01":"2024-01-10", valid_grids].max().fillna(l_jan_end)

april_sub = flow_df.loc["2024-04-01":"2024-04-07"] if "2024-04-01" in flow_df.index else flow_df.loc["2024-05-01":"2024-05-07"]
may_factors = pd.DataFrame([get_cycle_factor(dt, valid_grids) for dt in april_sub.index], index=april_sub.index)
l_resume_start = (april_sub / may_factors).median().fillna(l_jan_end)

post_sub = flow_df.loc["2024-04-01":"2024-10-31"] if "2024-04-01" in flow_df.index else flow_df.loc["2024-05-01":"2024-10-31"]
post_factors = pd.DataFrame([get_cycle_factor(dt, valid_grids) for dt in post_sub.index], index=post_sub.index)
l_long_term = (post_sub / post_factors).median().fillna(l_resume_start)

# 5. 執行全期動態生成 (含受控微波動 AR(1) 與優化後的類別動態)
np.random.seed(42)
full_pred_dates = pd.date_range(PRED_START, PRED_END, freq="D")
gap_span = (GAP_END - GAP_START).days + 1

ar_state = pd.Series(0.0, index=valid_grids)
pred_grid_records = {}

# 震前歷史 OD 轉移機率
hist_od_trans_prob = {g: {} for g in valid_grids}
for dt in flow_df[pre_mask].index:
    day_od = daily_od_records.get(dt, {})
    for orig in valid_grids:
        if orig in day_od:
            for dest, cnt in day_od[orig].items():
                hist_od_trans_prob[orig][dest] = hist_od_trans_prob[orig].get(dest, 0.0) + cnt

smoothed_od_prob = {}
for orig in valid_grids:
    total_c = sum(hist_od_trans_prob[orig].values())
    smoothed_od_prob[orig] = {d: c / total_c for d, c in hist_od_trans_prob[orig].items()} if total_c > 0 else {orig: 1.0}

pred_od_records = {}

for dt in full_pred_dates:
    r_t = get_cycle_factor(dt, valid_grids)
    
    # -------------------------------------------------------------
    # 基礎位準趨勢計算 (重點優化 Class 7)
    # -------------------------------------------------------------
    if dt < GAP_START:
        day_idx = (dt - PRED_START).days
        tau = day_idx / 30.0
        jan_init = flow_df.loc["2024-01-01":"2024-01-05"].median().fillna(l_jan_end)
        mu_t = jan_init + (tau ** 1.3) * (l_jan_end - jan_init)
        
        # 【修改點 1】Class 07 在 1 月份的激增與衰退包絡線
        for g in valid_grids:
            if grid_class_lookup.get(g, 0) == 7:
                p_val = jan_peaks[g]
                if day_idx <= 4:
                    # 震後前 4 天快速衝頂
                    mu_t[g] = jan_init[g] + (p_val - jan_init[g]) * (day_idx / 4.0)
                else:
                    # 5~31 天指數衰減至 1 月底
                    decay_tau = (day_idx - 4) / 26.0
                    mu_t[g] = l_jan_end[g] + (p_val - l_jan_end[g]) * np.exp(-2.5 * decay_tau)

    elif dt <= GAP_END:
        tau = ((dt - GAP_START).days + 1) / gap_span
        s_curve = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
        mu_t = l_jan_end + s_curve * (l_resume_start - l_jan_end)
        
        # 類別專屬特徵
        for g in valid_grids:
            c = grid_class_lookup.get(g, 0)
            
            if c == 3:  # 僅保留 Class 03 避難活動波峰
                peak_factor = np.sin(np.pi * tau) * 0.35 * l_jan_end[g]
                mu_t[g] += peak_factor
                
            elif c == 7:
                # 【修改點 2】Class 07 移除正弦波！改採單調平滑消退 (Dissipation)
                # 從 1 月底殘餘高位平滑過渡消散至 5 月初水準，杜絕 3 月假雙峰
                dissip_curve = 1.0 - np.exp(-3.0 * tau)
                mu_t[g] = l_jan_end[g] + dissip_curve * (l_resume_start[g] - l_jan_end[g])
                
            elif c == 4:  # Class 04 部分復原：早期緩慢、中後期抬升
                mu_t[g] = l_jan_end[g] + (tau ** 2.2) * (l_resume_start[g] - l_jan_end[g])
    else:
        tau_post = min(1.0, (dt - (GAP_END + pd.Timedelta(days=1))).days / 90.0)
        mu_t = l_resume_start + (1.0 - np.exp(-3.0 * tau_post)) * (l_long_term - l_resume_start)
    
    # -------------------------------------------------------------
    # AR(1) 微擾動生成與邊界防禦
    # -------------------------------------------------------------
    innovations = pd.Series(np.random.normal(0, 1, len(valid_grids)), index=valid_grids) * grid_volatility * 0.40
    ar_state = 0.68 * ar_state + np.sqrt(1 - 0.68**2) * innovations
    clamped_noise = ar_state.clip(lower=-1.2 * grid_volatility, upper=1.2 * grid_volatility)
    
    # Class 1 (Persistent Zero) 保持為零
    for g in valid_grids:
        if grid_class_lookup.get(g) == 1:
            clamped_noise[g] = 0.0
            mu_t[g] = 0.0
    
    # 結合週期、均值趨勢與微擾動
    raw_pred = mu_t * r_t + clamped_noise
    
    # 空間輕度平滑
    smooth_pred = 0.95 * raw_pred + 0.05 * spatial_knn_weights.dot(raw_pred)
    
    # 嚴格邊界保護
    final_pred = smooth_pred.clip(lower=0.0, upper=max_pre_allowable)
    pred_grid_records[dt] = final_pred

    # 還原 OD 矩陣
    day_od_pred = {}
    for orig in valid_grids:
        orig_vol = final_pred[orig]
        if orig_vol > 0 and orig in smoothed_od_prob:
            day_od_pred[orig] = {d: prob * orig_vol for d, prob in smoothed_od_prob[orig].items()}
        else:
            day_od_pred[orig] = {orig: orig_vol}
    pred_od_records[dt] = day_od_pred

pred_df = pd.DataFrame.from_dict(pred_grid_records, orient='index')
pred_csv_path = os.path.join(OUTPUT_DIR, "full_predictions_dynamic_jan_to_oct.csv")
pred_df.to_csv(pred_csv_path, encoding="utf-8-sig")

# =========================================================================
# 4. HuMob 官方 Combined NRMSE 評估
# =========================================================================
print("\n[3/5] 執行 HuMob Combined NRMSE 評估...")

eval_dates = [dt for dt in flow_df.index if dt >= PRED_START and not (GAP_START <= dt <= GAP_END)]
daily_metrics = []

for dt in eval_dates:
    if dt not in daily_od_records or not daily_od_records[dt]: continue
    act_od, prd_od = daily_od_records[dt], pred_od_records.get(dt, {})
    diag_diffs, offdiag_diffs = [], []
    
    for orig in valid_grids:
        act_dests, prd_dests = act_od.get(orig, {}), prd_od.get(orig, {})
        diag_diffs.append((float(prd_dests.get(orig, 0.0)) - float(act_dests.get(orig, 0.0))) ** 2)
        
        all_dests = set(act_dests.keys()).union(set(prd_dests.keys()))
        for dest in all_dests:
            if dest == orig or dest == "-1_-1": continue
            offdiag_diffs.append((float(prd_dests.get(dest, 0.0)) - float(act_dests.get(dest, 0.0))) ** 2)

    daily_metrics.append({
        "RMSE_diag": np.sqrt(np.mean(diag_diffs)) if diag_diffs else 0.0,
        "RMSE_offdiag": np.sqrt(np.mean(offdiag_diffs)) if offdiag_diffs else 0.0
    })

df_eval = pd.DataFrame(daily_metrics)
m_diag, m_off = df_eval["RMSE_diag"].mean(), df_eval["RMSE_offdiag"].mean()
n_diag, n_off = m_diag / 26.57, m_off / 0.0176
final_score = 0.5 * n_diag + 0.5 * n_off

print("-" * 75)
print(f"🔹 Step 1 | Mean RMSE (Diag):        {m_diag:8.4f}")
print(f"🔹 Step 2 | Mean RMSE (Off-Diag):    {m_off:8.4f}")
print(f"🔸 Step 3 | NRMSE (Diag)   [/26.57]: {n_diag:8.4f}")
print(f"🔸 Step 3 | NRMSE (Off-Diag) [/0.0176]: {n_off:8.4f}")
print(f"🏆 Step 4 | Combined NRMSE:          {final_score:8.4f}")
print("-" * 75)

# =========================================================================
# 5. 繪製 9 大類別圖譜
# =========================================================================
print("\n[4/5] 繪製 9 大類別獨立圖與 3x3 總覽圖...")

plt.style.use('dark_background')
COLOR_ACTUAL = '#f43f5e'
COLOR_PRED = '#10b981'
COLOR_BASE = '#64748b'
COLOR_GAP = '#f59e0b'

actual_plot_df = flow_df.copy()
actual_plot_df.loc[(actual_plot_df.index >= GAP_START) & (actual_plot_df.index <= GAP_END)] = np.nan
pred_plot_df = pred_df[pred_df.index >= PRED_START].copy()

baseline_df = pd.DataFrame(
    dow_medians_pre.loc[flow_df.index.dayofweek, valid_grids].values,
    index=flow_df.index,
    columns=valid_grids
)

class_series = {}
for c_id in range(1, 10):
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    if not c_grids: continue
    class_series[c_id] = {
        "name": CLASS_INFO_MAP[c_id],
        "dates": flow_df.index,
        "actual": actual_plot_df[c_grids].mean(axis=1),
        "baseline": baseline_df[c_grids].mean(axis=1),
        "pred_dates": pred_plot_df.index,
        "pred": pred_plot_df[c_grids].mean(axis=1),
        "count": len(c_grids)
    }

fig_grid, axes = plt.subplots(3, 3, figsize=(19, 11.5), dpi=300)
fig_grid.patch.set_facecolor('#0f172a')
fig_grid.suptitle(f'Dynamic 9-Class Predictions (Optimized Class 7) | Combined NRMSE: {final_score:.4f}', 
                  fontsize=15, fontweight='bold', color='#ffffff', y=0.98)

for c_id in range(1, 10):
    row, col = (c_id - 1) // 3, (c_id - 1) % 3
    ax = axes[row, col]
    ax.set_facecolor('#1e293b')
    if c_id not in class_series: continue

    data = class_series[c_id]
    ax.axvspan(GAP_START, GAP_END, color=COLOR_GAP, alpha=0.15, label='Feb-Apr Gap' if c_id == 1 else "")
    ax.plot(data["dates"], data["baseline"], color=COLOR_BASE, linestyle=':', linewidth=1.0, label='Pre-EQ Baseline' if c_id == 1 else "")
    ax.plot(data["dates"], data["actual"], color=COLOR_ACTUAL, linewidth=1.2, label='Actual Flow' if c_id == 1 else "")
    ax.plot(data["pred_dates"], data["pred"], color=COLOR_PRED, linewidth=1.5, label='Dynamic Prediction' if c_id == 1 else "")

    ax.set_title(f"{data['name']} (N={data['count']})", fontsize=11, fontweight='bold', color='#f8fafc', pad=8)
    ax.grid(True, color='#334155', linestyle=':', alpha=0.5)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

fig_grid.legend(loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=4, 
                fontsize=11, frameon=True, facecolor='#1e293b', edgecolor='#475569')
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

overview_path = os.path.join(OUTPUT_DIR, "all_9classes_comparison_overview.png")
plt.savefig(overview_path, dpi=300, bbox_inches='tight')
plt.close(fig_grid)
print(f"✓ 圖表輸出完成：{overview_path}")
