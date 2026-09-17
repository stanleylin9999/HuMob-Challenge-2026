import os
import re
import ast
import glob
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# =========================================================================
# 1. 全域配置、隨機種子與官方標準評估常數
# =========================================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
CLEAN_SCRIPT_DIR = os.path.abspath(os.path.normpath(str(SCRIPT_DIR).strip().replace('\xa0', ' ')))
os.makedirs(CLEAN_SCRIPT_DIR, exist_ok=True)

# 自動探測 TSV 檔案路徑
candidate_tsvs = glob.glob(os.path.join(CLEAN_SCRIPT_DIR, "**", "*dataset*.tsv"), recursive=True) + \
                 glob.glob(os.path.join(CLEAN_SCRIPT_DIR, "*dataset*.tsv"))
TSV_PATH = candidate_tsvs[0] if candidate_tsvs else os.path.join(CLEAN_SCRIPT_DIR, "humob2026-dataset.tsv")

# 自動探測 by_class 分類目錄
candidate_class_dirs = [
    os.path.join(CLEAN_SCRIPT_DIR, "by_class"),
    os.path.join(CLEAN_SCRIPT_DIR, "humob2026", "data", "output", "module05", "classification", "by_class"),
    r"C:\Users\User\Desktop\人口預測專案\人口預測專案3\humob2026\data\output\module05\classification\by_class"
]
BY_CLASS_DIR = next((c for c in candidate_class_dirs if os.path.exists(c) and len(glob.glob(os.path.join(c, "*.csv"))) > 0), None)

# 官方標準評估常數
MEAN_ACTUAL_DIAG = 26.57
MEAN_ACTUAL_OFFDIAG = 0.0176
WEIGHT_DIAG = 0.5
WEIGHT_OFFDIAG = 0.5

BASELINE1_DIAG_RMSE = 2.8527
BASELINE1_DIAG_NRMSE = 0.1074
BASELINE1_OFFDIAG_RMSE = 0.006070
BASELINE1_OFFDIAG_NRMSE = 0.3449
BASELINE1_COMBINED_NRMSE = 0.2261

PRED_START = pd.to_datetime("2024-01-01")
GAP_START = pd.to_datetime("2024-02-01")
GAP_END = pd.to_datetime("2024-03-31")
PRED_END = pd.to_datetime("2024-10-31")
GAP_LEN = (GAP_END - GAP_START).days + 1

CLASS_METADATA = {
    1: {"name": "Persistent Zero", "desc": "Uninhabited / Zero Flow Baseline"},
    2: {"name": "Persistent Decrease", "desc": "Severely Damaged Northern Epicenter"},
    3: {"name": "Emergent / Temporary Activity", "desc": "Relief & Supply Staging Hub"},
    4: {"name": "Partial Recovery", "desc": "Gradual Infrastructure Repair"},
    5: {"name": "Fully Recovered", "desc": "Rapid Commercial Rebound"},
    6: {"name": "Stable Inflow", "desc": "Southern Life Artery Cross-Flow"},
    7: {"name": "Temporary Increase", "desc": "Post-Quake Evacuation Surge"},
    8: {"name": "Partial Dissipation", "desc": "Secondary Relocation Outflow"},
    9: {"name": "Persistent Increase", "desc": "Post-Disaster Reconstruction Zone"}
}

def safe_save_fig(fig, file_path, dpi=220):
    clean_path = str(file_path).replace('\xa0', ' ').replace('\ufeff', '').replace('\u200b', '')
    clean_path = re.sub(r'[\r\n\t]', '', clean_path).strip()
    clean_path = os.path.abspath(os.path.normpath(clean_path))
    try:
        fig.savefig(clean_path, dpi=dpi, bbox_inches='tight')
    except OSError as e:
        if getattr(e, 'errno', None) == 22 and os.name == 'nt':
            ext_path = '\\\\?\\' + clean_path if not clean_path.startswith('\\\\?\\') else clean_path
            fig.savefig(ext_path, dpi=dpi, bbox_inches='tight')
        else:
            raise e

# =========================================================================
# 2. 空間邊界過濾與資料讀取 (對角線與非對角線)
# =========================================================================
def get_class_id(fname):
    f = fname.lower()
    if "zero" in f: return 1
    if "decrease" in f: return 2
    if "emergent" in f or "temporary_activity" in f: return 3
    if "partial_recovery" in f or "partial_rec" in f: return 4
    if "recovered" in f: return 5
    if "stable" in f: return 6
    if "temporary_increase" in f or "temp_inc" in f: return 7
    if "partial_dissipation" in f or "dissip" in f: return 8
    if "persistent_increase" in f or "increase" in f: return 9
    return None

def is_within_official_boundary(grid_str):
    try:
        p = grid_str.split('_')
        if len(p) != 2: return False
        v1, v2 = int(p[0]), int(p[1])
        return ((30 <= v2 <= 70) and (35 <= v1 <= 70)) or ((30 <= v1 <= 70) and (35 <= v2 <= 70))
    except:
        return False

print("[1/8] 載入類別對應表與真實 OD 資料集...")
grid_class_lookup = {}
if BY_CLASS_DIR and os.path.exists(BY_CLASS_DIR):
    for fpath in glob.glob(os.path.join(BY_CLASS_DIR, "*.csv")):
        cid = get_class_id(os.path.basename(fpath))
        if cid:
            try:
                df_cls = pd.read_csv(fpath)
                col = [c for c in df_cls.columns if any(k in str(c).lower() for k in ["grid", "orig", "id"])][0]
                for g in df_cls[col].dropna().astype(str).unique():
                    if is_within_official_boundary(g):
                        grid_class_lookup[g] = cid
            except:
                pass

daily_od_records = {}
raw_df = pd.read_csv(TSV_PATH, sep="\t", names=["date", "od_matrix_raw"])
raw_df['date_dt'] = pd.to_datetime(raw_df['date'].astype(str), format='%Y%m%d')
raw_df = raw_df.sort_values('date_dt').reset_index(drop=True)

for dt, val in zip(raw_df['date_dt'], raw_df['od_matrix_raw']):
    daily_od_records[dt] = {}
    if pd.isna(val) or val == "NA": continue
    try:
        od_dict = ast.literal_eval(val) if isinstance(val, str) else val
        for orig, dests in od_dict.items():
            if orig in grid_class_lookup and is_within_official_boundary(orig):
                filtered_dests = {
                    d: float(cnt) for d, cnt in dests.items() 
                    if d != "-1_-1" and (d == orig or (d in grid_class_lookup and is_within_official_boundary(d)))
                }
                daily_od_records[dt][orig] = filtered_dests
    except:
        pass

valid_grids = sorted(list(grid_class_lookup.keys()))
num_nodes = len(valid_grids)
print(f"✓ 成功載入 {num_nodes} 個範圍內有效網格 (x:30~70, y:35~70)")

diag_dict, off_dict = {}, {}
for dt, day_od in daily_od_records.items():
    diag_dict[dt], off_dict[dt] = {}, {}
    for g in valid_grids:
        dests = day_od.get(g, {})
        diag_dict[dt][g] = float(dests.get(g, 0.0))
        off_dict[dt][g] = sum(float(v) for k, v in dests.items() if k != g)

diag_df = pd.DataFrame.from_dict(diag_dict, orient='index').fillna(0.0).astype(np.float32)
offdiag_df = pd.DataFrame.from_dict(off_dict, orient='index').fillna(0.0).astype(np.float32)

# =========================================================================
# 3. 4 月經驗稀疏先驗 OD 轉移機率矩陣引擎
# =========================================================================
print("[2/8] 構建 4 月經驗稀疏先驗轉移引擎 (鎖定非對角線拓撲)...")
class EmpiricalAprilTransferEngine:
    def __init__(self, valid_grids, daily_od_records):
        self.valid_grids = valid_grids
        apr_dates = [
            dt for dt in daily_od_records 
            if dt.year == 2024 and dt.month == 4 and len(daily_od_records[dt]) > 0
        ]
        if not apr_dates:
            apr_dates = [dt for dt in daily_od_records if dt > GAP_END]

        self.P_apr = self._build_empirical_matrix(daily_od_records, apr_dates)

    def _build_empirical_matrix(self, records, dates):
        counts = {g: {} for g in self.valid_grids}
        for dt in dates:
            day_od = records.get(dt, {})
            for orig in self.valid_grids:
                if orig in day_od:
                    for dest, cnt in day_od[orig].items():
                        if dest != orig and dest in self.valid_grids:
                            counts[orig][dest] = counts[orig].get(dest, 0.0) + float(cnt)

        probs = {}
        for orig in self.valid_grids:
            tot = sum(counts[orig].values())
            if tot > 0:
                probs[orig] = {d: c / tot for d, c in counts[orig].items()}
            else:
                probs[orig] = {}
        return probs

    def get_matrix(self, dt):
        return self.P_apr

transfer_engine = EmpiricalAprilTransferEngine(valid_grids, daily_od_records)

# =========================================================================
# 4. 可微分自適應 Sigmoid 學習器 (以梯度下降自動學習前後資料求取最佳 k 與 tau_m)
# =========================================================================
class DifferentiablePlateauSigmoid(nn.Module):
    """
    實作核心封閉型端點歸一化自適應方程式：
    S(tau) = 1 / (1 + exp(-k * (tau - tau_m)))
    S_hat(tau) = (S(tau) - S(0)) / (S(1) - S(0))
    y(tau) = y0 + (y1 - y0) * S_hat(tau)
    """
    def __init__(self, num_entities, init_k=6.0, init_taum=0.5):
        super().__init__()
        init_k_raw = math.log(max(1e-4, math.exp(init_k - 1.0) - 1.0))
        init_taum_raw = -math.log(max(1e-4, 1.0 / init_taum - 1.0))

        self.k_raw = nn.Parameter(torch.full((num_entities,), init_k_raw, dtype=torch.float32))
        self.taum_raw = nn.Parameter(torch.full((num_entities,), init_taum_raw, dtype=torch.float32))

    def get_constrained_params(self):
        k = F.softplus(self.k_raw) + 1.0
        tau_m = torch.sigmoid(self.taum_raw)
        return k, tau_m

    def forward(self, tau, entity_ids, y0, y1):
        k_all, taum_all = self.get_constrained_params()
        k = k_all[entity_ids].unsqueeze(-1)
        tau_m = taum_all[entity_ids].unsqueeze(-1)
        y0 = y0.unsqueeze(-1)
        y1 = y1.unsqueeze(-1)

        S = torch.sigmoid(k * (tau - tau_m))
        S_0 = torch.sigmoid(k * (0.0 - tau_m))
        S_1 = torch.sigmoid(k * (1.0 - tau_m))
        S_hat = (S - S_0) / (S_1 - S_0 + 1e-8)
        return y0 + (y1 - y0) * S_hat

def learn_plateau_sigmoid_parameters(
    gt_flow_df: pd.DataFrame,
    grid_class_lookup: dict,
    valid_grids: list,
    epochs: int = 350,
    lr: float = 0.04
):
    print("[3/8] 執行 PyTorch 梯度下降自適應學習 9 大類別 Sigmoid 物理參數 (k, tau_m)...")
    learn_dates = pd.date_range("2024-01-15", "2024-04-30", freq="D")
    T_total = len(learn_dates)

    tau_np = np.linspace(0.0, 1.0, T_total, dtype=np.float32)
    obs_mask_np = ~((learn_dates >= GAP_START) & (learn_dates <= GAP_END))

    tau_t = torch.tensor(tau_np, device=DEVICE).unsqueeze(0)
    obs_mask_t = torch.tensor(obs_mask_np, device=DEVICE).unsqueeze(0)

    B = len(valid_grids)
    cids = np.array([grid_class_lookup.get(g, 5) - 1 for g in valid_grids], dtype=np.int64)
    cids_t = torch.tensor(cids, device=DEVICE)

    jan_clean = gt_flow_df.loc["2024-01-15":"2024-01-31", valid_grids]
    apr_clean = gt_flow_df.loc["2024-04-01":"2024-04-15", valid_grids]
    y0_vals = jan_clean.median().values.astype(np.float32)
    y1_vals = apr_clean.median().values.astype(np.float32)

    y0_t = torch.tensor(y0_vals, device=DEVICE)
    y1_t = torch.tensor(y1_vals, device=DEVICE)

    # 解決 KeyError：對齊完整日期序列，Gap 期間補 0.0 並以遮罩過濾[cite: 1]
    aligned_gt_df = gt_flow_df.reindex(learn_dates, fill_value=0.0)
    gt_target_t = torch.tensor(aligned_gt_df[valid_grids].values.T, dtype=torch.float32, device=DEVICE)

    model = DifferentiablePlateauSigmoid(num_entities=9).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred_curve = model(tau_t.repeat(B, 1), cids_t, y0_t, y1_t)

        diff = (pred_curve - gt_target_t) * obs_mask_t
        loss_data = torch.sum(diff ** 2) / (torch.sum(obs_mask_t) * B + 1e-8)

        d2 = pred_curve[:, 2:] - 2.0 * pred_curve[:, 1:-1] + pred_curve[:, :-2]
        loss_smooth = torch.mean(d2 ** 2)

        total_loss = loss_data + 0.05 * loss_smooth
        total_loss.backward()
        optimizer.step()

    learned_k, learned_taum = model.get_constrained_params()
    k_dict = {cid + 1: float(learned_k[cid].item()) for cid in range(9)}
    taum_dict = {cid + 1: float(learned_taum[cid].item()) for cid in range(9)}

    print("=" * 68)
    print(f"{'類別 ID':<10} | {'類別名稱':<24} | {'學習陡度 (k)':<14} | {'轉折中心 (tau_m)':<14}")
    print("-" * 68)
    for cid in range(1, 10):
        print(f"Class {cid:02d}    | {CLASS_METADATA[cid]['name']:<24} | {k_dict[cid]:<14.4f} | {taum_dict[cid]:<14.4f}")
    print("=" * 68)

    return k_dict, taum_dict

learned_k_dict, learned_taum_dict = learn_plateau_sigmoid_parameters(diag_df, grid_class_lookup, valid_grids)

# =========================================================================
# 5. 宏觀動力學基線展開引擎 (結合 Sigmoid 趨勢與週間波形)
# =========================================================================
class LearnedSigmoidDynamicEngine:
    def __init__(self, flow_df, valid_grids, grid_class_lookup, k_dict, taum_dict, is_offdiag=False):
        self.flow_df = flow_df
        self.valid_grids = valid_grids
        self.grid_class_lookup = grid_class_lookup
        self.k_dict = k_dict
        self.taum_dict = taum_dict
        self.is_offdiag = is_offdiag
        self.canonical_waves = {}
        self._fit()

    def _fit(self):
        post_df = self.flow_df.loc[self.flow_df.index > GAP_END].copy()
        if len(post_df) < 14:
            post_df = self.flow_df.loc[~((self.flow_df.index >= GAP_START) & (self.flow_df.index <= GAP_END))].copy()
        post_df['dow'] = post_df.index.dayofweek

        for g in self.valid_grids:
            cid = self.grid_class_lookup.get(g, 5)
            sparsity = (post_df[g] == 0).mean()

            if cid == 1 or (self.is_offdiag and (cid == 3 or sparsity > 0.95)):
                self.canonical_waves[g] = np.zeros(7, dtype=np.float32)
                continue

            medians = post_df.groupby('dow')[g].median().values
            wave = medians - medians[0]
            denom = np.max(np.abs(wave))
            if denom > 1e-6:
                self.canonical_waves[g] = (wave / denom).astype(np.float32)
            else:
                self.canonical_waves[g] = np.zeros(7, dtype=np.float32)

    def generate(self, date_range):
        mondays = date_range[date_range.dayofweek == 0]
        meta = []
        date_to_idx = {d: i for i, d in enumerate(date_range)}
        pred_mat = np.zeros((len(date_range), len(self.valid_grids)), dtype=np.float32)

        for g_idx, g in enumerate(self.valid_grids):
            cid = self.grid_class_lookup.get(g, 5)
            s_d = self.canonical_waves[g]

            if cid == 1:
                pred_mat[:, g_idx] = 0.0
                continue

            jan_series = self.flow_df.loc["2024-01-15":"2024-01-31", g]
            apr_series = self.flow_df.loc["2024-04-01":"2024-04-30", g]

            M_jan = float(jan_series.median()) if len(jan_series) > 0 else float(self.flow_df[g].mean())
            M_apr = float(apr_series.mean()) if len(apr_series) > 0 else M_jan

            amp_apr = float(apr_series.std()) if len(apr_series) > 1 else 0.0
            amp_jan = float(jan_series.std()) if len(jan_series) > 1 else amp_apr

            if self.is_offdiag:
                A_apr = float(np.clip(amp_apr, 0.0005, max(0.002, M_apr * 0.25)))
                A_jan = float(np.clip(amp_jan, 0.0005, max(0.002, M_jan * 0.25)))
            else:
                A_apr = float(np.clip(amp_apr, 0.05, max(0.1, M_apr * 0.20)))
                A_jan = float(np.clip(amp_jan, 0.05, max(0.1, M_jan * 0.20)))

            gap_mondays = [m for m in mondays if GAP_START <= m <= GAP_END]
            mon_dict, amp_dict = {}, {}

            k_val = self.k_dict.get(cid, 6.0)
            taum_val = self.taum_dict.get(cid, 0.5)

            for idx, m in enumerate(gap_mondays):
                u = (idx + 1) / float(len(gap_mondays) + 1)
                
                S_u = 1.0 / (1.0 + math.exp(-np.clip(k_val * (u - taum_val), -20.0, 20.0)))
                S_0 = 1.0 / (1.0 + math.exp(-np.clip(k_val * (0.0 - taum_val), -20.0, 20.0)))
                S_1 = 1.0 / (1.0 + math.exp(-np.clip(k_val * (1.0 - taum_val), -20.0, 20.0)))
                s = float((S_u - S_0) / (S_1 - S_0 + 1e-8))

                mon_dict[m] = float(M_jan + s * (M_apr - M_jan))
                amp_dict[m] = float(A_jan + s * (A_apr - A_jan))

            for m in mondays:
                if m not in mon_dict:
                    if m in self.flow_df.index:
                        mon_dict[m] = float(self.flow_df.loc[m, g])
                        w_span = self.flow_df.loc[m : m + pd.Timedelta(days=6), g]
                        raw_a = float(w_span.std()) if len(w_span) > 1 else A_apr
                        amp_dict[m] = float(np.clip(raw_a, 0.0005 if self.is_offdiag else 0.05, max(0.005 if self.is_offdiag else 0.2, mon_dict[m] * 0.25)))
                    else:
                        mon_dict[m] = M_apr
                        amp_dict[m] = A_apr

                meta.append({
                    "component": "Off-Diagonal" if self.is_offdiag else "Diagonal",
                    "grid_id": g, "class_id": cid, "monday_date": m.strftime("%Y-%m-%d"),
                    "monday_anchor_level": round(float(mon_dict[m]), 4),
                    "weekly_amplitude": round(float(amp_dict[m]), 4)
                })

            for i in range(len(mondays)):
                m_curr = mondays[i]
                m_next = mondays[i+1] if i + 1 < len(mondays) else m_curr + pd.Timedelta(days=7)
                delta_M = float(mon_dict.get(m_next, mon_dict[m_curr]) - mon_dict[m_curr])
                for offset in range(7):
                    t_day = m_curr + pd.Timedelta(days=offset)
                    if t_day in date_to_idx:
                        val = mon_dict[m_curr] + (offset / 7.0) * delta_M + amp_dict[m_curr] * s_d[offset]
                        pred_mat[date_to_idx[t_day], g_idx] = max(0.0, float(val))

        pred_df = pd.DataFrame(pred_mat, index=date_range, columns=self.valid_grids, dtype=np.float32)
        return pred_df, pd.DataFrame(meta)

all_sim_dates = pd.date_range("2023-11-01", PRED_END, freq="D")
macro_diag_df, meta_diag_df = LearnedSigmoidDynamicEngine(diag_df, valid_grids, grid_class_lookup, learned_k_dict, learned_taum_dict, False).generate(all_sim_dates)
macro_offdiag_df, meta_offdiag_df = LearnedSigmoidDynamicEngine(offdiag_df, valid_grids, grid_class_lookup, learned_k_dict, learned_taum_dict, True).generate(all_sim_dates)
all_meta_df = pd.concat([meta_diag_df, meta_offdiag_df], ignore_index=True)

# =========================================================================
# 6. OT-FM 殘差訓練與 Batched RK4 推論 (零均值高頻純振幅生成)
# =========================================================================
print("[4/8] 訓練 OT-FM 殘差網絡並以 RK4 求解零均值有機高頻震盪...")
class FastOTUNet(nn.Module):
    def __init__(self, hidden=48, num_classes=9):
        super().__init__()
        self.c_emb = nn.Embedding(num_classes, hidden)
        self.t_mlp = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.m_proj = nn.Linear(1, hidden)
        self.in_c = nn.Conv1d(2, hidden, 3, padding=1)
        self.b1 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.b2 = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.gn = nn.GroupNorm(4, hidden)
        self.out_c = nn.Conv1d(hidden, 1, 3, padding=1)

    def forward(self, x, t, b, m, cid):
        cond = self.t_mlp(t) + self.m_proj(m) + self.c_emb(cid)
        h = self.in_c(torch.cat([x, b], dim=1)) + cond.unsqueeze(-1)
        return self.out_c(self.b2(F.silu(self.gn(self.b1(h)))) + h)

def train_otfm_fast(gt_df, base_df, epochs=12):
    res_df = (gt_df - base_df).astype(np.float32)
    valid_ms = [
        d for d in gt_df.index 
        if d.dayofweek == 0 
        and not (GAP_START <= d <= GAP_END) 
        and not (GAP_START <= d + pd.Timedelta(days=6) <= GAP_END)
        and all((d + pd.Timedelta(days=i)) in gt_df.index for i in range(7))
    ]
    
    samples = []
    for _ in range(2500):
        m = random.choice(valid_ms)
        g = random.choice(valid_grids)
        cid = grid_class_lookup.get(g, 5) - 1
        span = pd.date_range(m, periods=7, freq="D")
        samples.append((
            res_df.loc[span, g].values.astype(np.float32),
            base_df.loc[span, g].values.astype(np.float32),
            np.array([base_df.loc[m, g]], dtype=np.float32),
            cid
        ))
        
    loader = DataLoader(samples, batch_size=64, shuffle=True)
    model = FastOTUNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3)
    model.train()
    for _ in range(epochs):
        for res, b, m, cid in loader:
            res = res.unsqueeze(1).to(DEVICE)
            b = b.unsqueeze(1).to(DEVICE)
            m, cid = m.to(DEVICE), cid.to(DEVICE)
            B = res.size(0)
            t = torch.rand(B, 1, device=DEVICE)
            x0 = torch.randn_like(res)
            xt = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * res
            pred_v = model(xt, t, b, m, cid)
            loss = F.mse_loss(pred_v, res - x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

ot_diag = train_otfm_fast(diag_df, macro_diag_df)
ot_off = train_otfm_fast(offdiag_df, macro_offdiag_df)

@torch.no_grad()
def solve_batched_rk4(model, base_df, is_offdiag=False, steps=4, ensemble_size=4):
    model.eval()
    dt = 1.0 / steps
    pred_df = base_df.copy().astype(np.float32)
    gap_mondays = pd.date_range(GAP_START - pd.Timedelta(days=6), GAP_END, freq="W-MON")
    N = len(valid_grids)

    cids_np = np.array([grid_class_lookup.get(g, 5) - 1 for g in valid_grids], dtype=np.int64)
    cids_t = torch.from_numpy(cids_np).to(DEVICE).repeat(ensemble_size)
    zero_mask = (cids_np == 0) | ((cids_np == 2) if is_offdiag else False)

    for m in gap_mondays:
        w_span = pd.date_range(m, periods=7, freq="D")
        base_mat = base_df.loc[w_span, valid_grids].values.T.astype(np.float32)
        mon_mat = base_df.loc[m, valid_grids].values[:, None].astype(np.float32)

        base_t = torch.from_numpy(base_mat).unsqueeze(1).to(DEVICE).repeat(ensemble_size, 1, 1)
        mon_t = torch.from_numpy(mon_mat).to(DEVICE).repeat(ensemble_size, 1)
        x = torch.randn(N * ensemble_size, 1, 7, device=DEVICE)

        for s in range(steps):
            t_curr = s / steps
            t1 = torch.full((N * ensemble_size, 1), t_curr, device=DEVICE)
            k1 = model(x, t1, base_t, mon_t, cids_t)
            t2 = torch.full((N * ensemble_size, 1), t_curr + 0.5 * dt, device=DEVICE)
            k2 = model(x + 0.5 * dt * k1, t2, base_t, mon_t, cids_t)
            k3 = model(x + 0.5 * dt * k2, t2, base_t, mon_t, cids_t)
            t4 = torch.full((N * ensemble_size, 1), t_curr + dt, device=DEVICE)
            k4 = model(x + dt * k3, t4, base_t, mon_t, cids_t)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        gen_res = torch.median(x.squeeze(1).view(ensemble_size, N, 7), dim=0).values.cpu().numpy()

        # =====================================================================
        # 核心：殘差零均值校準 (Zero-Centering Calibration)
        # 解耦趨勢與震盪：模型僅專注生成純上下震盪，強制 7 天均值為 0，鎖定 Sigmoid 趨勢水位[cite: 1]
        # =====================================================================
        gen_res = gen_res - np.mean(gen_res, axis=-1, keepdims=True)

        final_w = np.maximum(0.0, base_mat + gen_res)
        final_w[zero_mask, :] = 0.0

        for offset, d in enumerate(w_span):
            if GAP_START <= d <= GAP_END:
                pred_df.loc[d, valid_grids] = final_w[:, offset]
    return pred_df

pred_diag = solve_batched_rk4(ot_diag, macro_diag_df, is_offdiag=False)
pred_off = solve_batched_rk4(ot_off, macro_offdiag_df, is_offdiag=True)

# -------------------------------------------------------------------------
# Class 6 專屬：實測真值樣板直拷貝 (鎖定南方生命線動脈 ~46 水平)[cite: 1]
# -------------------------------------------------------------------------
def apply_class6_copy_and_offset(
    pred_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    diag_obs_df: pd.DataFrame,
    valid_grids: list,
    grid_class_lookup: dict,
    gap_start: pd.Timestamp = GAP_START,
    gap_end: pd.Timestamp = GAP_END,
) -> pd.DataFrame:
    refined = pred_df.copy()
    c6_grids = [g for g in valid_grids if grid_class_lookup.get(g) == 6]
    if not c6_grids:
        return refined

    jan_clean_dates = [
        d for d in obs_df.index 
        if pd.to_datetime("2024-01-15") <= d <= pd.to_datetime("2024-01-31")
        and diag_obs_df.loc[d, c6_grids].mean() > 15.0
    ]
    apr_clean_dates = [
        d for d in obs_df.index 
        if pd.to_datetime("2024-04-01") <= d <= pd.to_datetime("2024-04-30")
        and diag_obs_df.loc[d, c6_grids].mean() > 15.0
    ]

    if len(jan_clean_dates) < 7:
        jan_clean_dates = [d for d in obs_df.index if d < gap_start and diag_obs_df.loc[d, c6_grids].mean() > 15.0]
    if len(apr_clean_dates) < 7:
        apr_clean_dates = [d for d in obs_df.index if d > gap_end and diag_obs_df.loc[d, c6_grids].mean() > 15.0]

    jan_clean = obs_df.loc[jan_clean_dates, c6_grids]
    apr_clean = obs_df.loc[apr_clean_dates, c6_grids]

    dow_template = jan_clean.groupby(jan_clean.index.dayofweek).mean()
    for dow in range(7):
        if dow not in dow_template.index:
            dow_template.loc[dow] = jan_clean.mean(axis=0)
    dow_template = dow_template.sort_index()

    jan_mean = jan_clean.mean(axis=0)
    apr_mean = apr_clean.mean(axis=0) if len(apr_clean) > 0 else jan_mean
    delta_mean = apr_mean - jan_mean

    gap_dates = refined.index[(refined.index >= gap_start) & (refined.index <= gap_end)]
    T_gap = len(gap_dates)

    for step_idx, dt in enumerate(gap_dates):
        dow = dt.dayofweek
        tau = (step_idx + 1) / float(T_gap + 1)
        smooth_tau = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
        val = dow_template.loc[dow] + smooth_tau * delta_mean
        refined.loc[dt, c6_grids] = np.maximum(0.0, val.values).astype(np.float32)

    return refined

print("[4.5/8] 對 Class 06 執行實測真值樣板直拷貝...")
pred_diag = apply_class6_copy_and_offset(pred_diag, diag_df, diag_df, valid_grids, grid_class_lookup)
pred_off = apply_class6_copy_and_offset(pred_off, offdiag_df, diag_df, valid_grids, grid_class_lookup)

# -------------------------------------------------------------------------
# Class 5 與 Class 9 非對角線：離散量子化脈衝校正[cite: 1]
# -------------------------------------------------------------------------
def inject_sparse_spikes_for_class5_and_9(
    pred_off_df: pd.DataFrame,
    obs_off_df: pd.DataFrame,
    valid_grids: list,
    grid_class_lookup: dict,
    gap_start: pd.Timestamp = GAP_START,
    gap_end: pd.Timestamp = GAP_END,
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)
    refined_pred = pred_off_df.copy()

    obs_dates = obs_off_df.index[~((obs_off_df.index >= gap_start) & (obs_off_df.index <= gap_end))]
    obs_clean = obs_off_df.loc[obs_dates].copy()
    obs_clean['dow'] = obs_clean.index.dayofweek

    gap_dates = refined_pred.index[(refined_pred.index >= gap_start) & (refined_pred.index <= gap_end)]

    for cid in [5, 9]:
        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == cid]
        if not c_grids: continue

        for g in c_grids:
            refined_pred.loc[gap_dates, g] = 0.0

        dow_spike_probs, dow_event_weights = {}, {}
        for dow in range(7):
            dow_df = obs_clean.loc[obs_clean['dow'] == dow, c_grids]
            daily_has_spike = (dow_df >= 0.5).any(axis=1)
            dow_spike_probs[dow] = float(daily_has_spike.mean()) if len(daily_has_spike) > 0 else 0.0
            active_vals = dow_df.values[dow_df.values >= 0.5]
            dow_event_weights[dow] = np.round(active_vals) if len(active_vals) > 0 else np.array([1.0], dtype=np.float32)

        T_gap = len(gap_dates)
        for step_idx, dt in enumerate(gap_dates):
            dow = dt.dayofweek
            base_p = dow_spike_probs.get(dow, 0.0)
            weight = (1.15 - 0.35 * (step_idx / float(T_gap))) if cid == 9 else (0.50 + 0.60 * (step_idx / float(T_gap)))
            p_inject = float(np.clip(base_p * weight, 0.0, 0.60))

            if random.random() < p_inject:
                target_grid = random.choice(c_grids)
                candidates = dow_event_weights.get(dow, [1.0])
                val = float(random.choice(candidates))
                refined_pred.loc[dt, target_grid] = max(1.0, val)

    return refined_pred

print("[4.6/8] 對 Class 05 與 Class 09 非對角線執行離散量子化脈衝校正...")
pred_off = inject_sparse_spikes_for_class5_and_9(
    pred_off_df=pred_off,
    obs_off_df=offdiag_df,
    valid_grids=valid_grids,
    grid_class_lookup=grid_class_lookup,
    gap_start=GAP_START,
    gap_end=GAP_END
)

pure_model_diag = pred_diag.copy()
pure_model_off = pred_off.copy()

obs_dates = [d for d in diag_df.index if not (GAP_START <= d <= GAP_END)]
pred_diag.loc[obs_dates, valid_grids] = diag_df.loc[obs_dates, valid_grids].values.astype(np.float32)
pred_off.loc[obs_dates, valid_grids] = offdiag_df.loc[obs_dates, valid_grids].values.astype(np.float32)

pred_total = pred_diag + pred_off
raw_total = diag_df + offdiag_df
macro_total = macro_diag_df + macro_offdiag_df

# =========================================================================
# 7. 官方評估指標計算與 9 大類別報表匯出
# =========================================================================
print("[5/8] 計算官方標準 Combined NRMSE 指標並匯出資料 CSV...")
eval_dates = [d for d in diag_df.index if d >= PRED_START and not (GAP_START <= d <= GAP_END)]
N_diag = num_nodes
N_offdiag = num_nodes * (num_nodes - 1)

daily_eval = []
for dt in eval_dates:
    act_od = daily_od_records.get(dt, {})
    p_d = pred_diag.loc[dt]
    p_o = pred_off.loc[dt]
    Pt = transfer_engine.get_matrix(dt)

    sse_d = sum((float(p_d[g]) - float(act_od.get(g, {}).get(g, 0.0))) ** 2 for g in valid_grids)
    rmse_diag_d = np.sqrt(sse_d / N_diag)

    sse_o = 0.0
    for o in valid_grids:
        act = act_od.get(o, {})
        prb = Pt.get(o, {})
        t_off = float(p_o[o])
        active = set(act.keys()).union(prb.keys()).intersection(valid_grids) - {o}
        for d in active:
            obs = float(act.get(d, 0.0))
            pr = t_off * float(prb.get(d, 0.0))
            sse_o += (pr - obs) ** 2
    rmse_offdiag_d = np.sqrt(sse_o / N_offdiag)

    daily_eval.append({
        "date": dt.strftime('%Y-%m-%d'),
        "RMSE_diag_d": round(rmse_diag_d, 4),
        "RMSE_offdiag_d": round(rmse_offdiag_d, 6),
        "NRMSE_diag_d": round(rmse_diag_d / MEAN_ACTUAL_DIAG, 4),
        "NRMSE_offdiag_d": round(rmse_offdiag_d / MEAN_ACTUAL_OFFDIAG, 4)
    })

RMSE_diag = float(np.mean([r["RMSE_diag_d"] for r in daily_eval]))
RMSE_offdiag = float(np.mean([r["RMSE_offdiag_d"] for r in daily_eval]))
NRMSE_diag = RMSE_diag / MEAN_ACTUAL_DIAG
NRMSE_offdiag = RMSE_offdiag / MEAN_ACTUAL_OFFDIAG
combined_nrmse = (NRMSE_diag + NRMSE_offdiag) / 2.0

april_dates = [d for d in eval_dates if d.year == 2024 and d.month == 4]
april_eval_records = [r for r in daily_eval if pd.to_datetime(r["date"]).year == 2024 and pd.to_datetime(r["date"]).month == 4]

final_apr_rmse_diag = float(np.mean([r["RMSE_diag_d"] for r in april_eval_records])) if april_eval_records else 0.0
final_apr_rmse_off = float(np.mean([r["RMSE_offdiag_d"] for r in april_eval_records])) if april_eval_records else 0.0
final_apr_nrmse_diag = final_apr_rmse_diag / MEAN_ACTUAL_DIAG
final_apr_nrmse_off = final_apr_rmse_off / MEAN_ACTUAL_OFFDIAG
final_apr_combined = (final_apr_nrmse_diag + final_apr_nrmse_off) / 2.0

pure_apr_sse_d = 0.0
pure_apr_sse_o = 0.0
for dt in april_dates:
    act_od = daily_od_records.get(dt, {})
    p_d_pure = pure_model_diag.loc[dt]
    p_o_pure = pure_model_off.loc[dt]
    Pt = transfer_engine.get_matrix(dt)

    pure_apr_sse_d += sum((float(p_d_pure[g]) - float(act_od.get(g, {}).get(g, 0.0))) ** 2 for g in valid_grids)

    for o in valid_grids:
        act = act_od.get(o, {})
        prb = Pt.get(o, {})
        t_off = float(p_o_pure[o])
        active = set(act.keys()).union(prb.keys()).intersection(valid_grids) - {o}
        for d in active:
            obs = float(act.get(d, 0.0))
            pr = t_off * float(prb.get(d, 0.0))
            pure_apr_sse_o += (pr - obs) ** 2

T_apr = max(len(april_dates), 1)
pure_apr_rmse_diag = np.sqrt(pure_apr_sse_d / (T_apr * N_diag))
pure_apr_rmse_off = np.sqrt(pure_apr_sse_o / (T_apr * N_offdiag))
pure_apr_nrmse_diag = pure_apr_rmse_diag / MEAN_ACTUAL_DIAG
pure_apr_nrmse_off = pure_apr_rmse_off / MEAN_ACTUAL_OFFDIAG
pure_apr_combined = (pure_apr_nrmse_diag + pure_apr_nrmse_off) / 2.0

print("\n" + "=" * 80)
print(f" 📈 2024 年 4 月專用 NRMSE 評估報表 (評估有效天數: {len(april_dates)} 天)")
print("=" * 80)
print(f"{'指標評估維度':<26} | {'本程式純模型':<14} | {'本程式最終管線':<14} | {'Baseline 1 (官方基準)':<16}")
print("-" * 80)
print(f"{'Diagonal RMSE':<26} | {pure_apr_rmse_diag:<14.4f} | {final_apr_rmse_diag:<14.4f} | {BASELINE1_DIAG_RMSE:<16.4f}")
print(f"{'Diagonal NRMSE':<26} | {pure_apr_nrmse_diag:<14.4f} | {final_apr_nrmse_diag:<14.4f} | {BASELINE1_DIAG_NRMSE:<16.4f}")
print(f"{'Off-diagonal RMSE':<26} | {pure_apr_rmse_off:<14.6f} | {final_apr_rmse_off:<14.6f} | {BASELINE1_OFFDIAG_RMSE:<16.6f}")
print(f"{'Off-diagonal NRMSE':<26} | {pure_apr_nrmse_off:<14.4f} | {final_apr_nrmse_off:<14.4f} | {BASELINE1_OFFDIAG_NRMSE:<16.4f}")
print(f"{'Combined NRMSE':<26} | {pure_apr_combined:<14.4f} | {final_apr_combined:<14.4f} | {BASELINE1_COMBINED_NRMSE:<16.4f}")
print("=" * 80 + "\n")

df_april_vs_base = pd.DataFrame([
    {"Metric": "Diagonal RMSE", "Pure_Model": round(pure_apr_rmse_diag, 4), "Final_Pipeline": round(final_apr_rmse_diag, 4), "Baseline_1_Official": BASELINE1_DIAG_RMSE},
    {"Metric": "Diagonal NRMSE", "Pure_Model": round(pure_apr_nrmse_diag, 4), "Final_Pipeline": round(final_apr_nrmse_diag, 4), "Baseline_1_Official": BASELINE1_DIAG_NRMSE},
    {"Metric": "Off-diagonal RMSE", "Pure_Model": round(pure_apr_rmse_off, 6), "Final_Pipeline": round(final_apr_rmse_off, 6), "Baseline_1_Official": BASELINE1_OFFDIAG_RMSE},
    {"Metric": "Off-diagonal NRMSE", "Pure_Model": round(pure_apr_nrmse_off, 4), "Final_Pipeline": round(final_apr_nrmse_off, 4), "Baseline_1_Official": BASELINE1_OFFDIAG_NRMSE},
    {"Metric": "Combined NRMSE", "Pure_Model": round(pure_apr_combined, 4), "Final_Pipeline": round(final_apr_combined, 4), "Baseline_1_Official": BASELINE1_COMBINED_NRMSE}
])
df_april_vs_base.to_csv(os.path.join(CLEAN_SCRIPT_DIR, "humob_april2024_vs_baseline1.csv"), index=False, encoding="utf-8-sig")

def compute_classwise_nrmse_table(
    diag_df, offdiag_df, pred_diag, pred_off,
    daily_od_records, transfer_engine, valid_grids,
    grid_class_lookup, class_metadata, eval_dates,
    mean_actual_diag=MEAN_ACTUAL_DIAG, mean_actual_offdiag=MEAN_ACTUAL_OFFDIAG
):
    num_total_nodes = len(valid_grids)
    N_total_offdiag = num_total_nodes * (num_total_nodes - 1)
    
    class_stats = {cid: {"sse_diag": 0.0, "sse_off": 0.0, "count": 0} for cid in range(1, 10)}
    total_sse_diag = 0.0
    total_sse_off = 0.0

    for cid in range(1, 10):
        class_stats[cid]["count"] = sum(1 for g in valid_grids if grid_class_lookup.get(g) == cid)

    for dt in eval_dates:
        act_od = daily_od_records.get(dt, {})
        p_d = pred_diag.loc[dt]
        p_o = pred_off.loc[dt]
        Pt = transfer_engine.get_matrix(dt)

        for g in valid_grids:
            cid = grid_class_lookup.get(g, 5)
            err_d = float(p_d[g]) - float(act_od.get(g, {}).get(g, 0.0))
            se_d = err_d ** 2
            class_stats[cid]["sse_diag"] += se_d
            total_sse_diag += se_d

        for orig in valid_grids:
            cid = grid_class_lookup.get(orig, 5)
            act = act_od.get(orig, {})
            prb = Pt.get(orig, {})
            t_off = float(p_o[orig])
            active_dests = set(act.keys()).union(prb.keys()).intersection(valid_grids) - {orig}

            se_o_orig = 0.0
            for dest in active_dests:
                obs = float(act.get(dest, 0.0))
                pr = t_off * float(prb.get(dest, 0.0))
                se_o_orig += (pr - obs) ** 2

            class_stats[cid]["sse_off"] += se_o_orig
            total_sse_off += se_o_orig

    T = len(eval_dates)
    table_rows = []

    for cid in range(1, 10):
        c_name = class_metadata[cid]["name"]
        n_c = class_stats[cid]["count"]
        if n_c == 0: continue

        rmse_d = np.sqrt(class_stats[cid]["sse_diag"] / (T * n_c))
        nrmse_d = rmse_d / mean_actual_diag

        rmse_o = np.sqrt(class_stats[cid]["sse_off"] / (T * n_c * (num_total_nodes - 1)))
        nrmse_o = rmse_o / mean_actual_offdiag
        comb = (nrmse_d + nrmse_o) / 2.0

        table_rows.append({
            "class_id": f"class {cid}",
            "class_name": f"{c_name} ({n_c}格)",
            "nrmse_diag": nrmse_d,
            "nrmse_off": nrmse_o,
            "combined": comb
        })

    tot_rmse_d = np.sqrt(total_sse_diag / (T * num_total_nodes))
    tot_nrmse_d = tot_rmse_d / mean_actual_diag
    tot_rmse_o = np.sqrt(total_sse_off / (T * N_total_offdiag))
    tot_nrmse_o = tot_rmse_o / mean_actual_offdiag
    tot_comb = (tot_nrmse_d + tot_nrmse_o) / 2.0

    table_rows.append({
        "class_id": "total",
        "class_name": f"({num_total_nodes}格)",
        "nrmse_diag": tot_nrmse_d,
        "nrmse_off": tot_nrmse_o,
        "combined": tot_comb
    })

    return pd.DataFrame(table_rows)

df_metrics_table = compute_classwise_nrmse_table(
    diag_df, offdiag_df, pred_diag, pred_off,
    daily_od_records, transfer_engine, valid_grids,
    grid_class_lookup, CLASS_METADATA, eval_dates
)

# 匯出各項預測與指標 CSV
pd.DataFrame(daily_eval).to_csv(os.path.join(CLEAN_SCRIPT_DIR, "humob_daily_od_metrics.csv"), index=False, encoding="utf-8-sig")
df_metrics_table.to_csv(os.path.join(CLEAN_SCRIPT_DIR, "humob_official_nrmse_summary.csv"), index=False, encoding="utf-8-sig")
df_metrics_table.to_csv(os.path.join(CLEAN_SCRIPT_DIR, "humob_classwise_nrmse_table.csv"), index=False, encoding="utf-8-sig")
all_meta_df.to_csv(os.path.join(CLEAN_SCRIPT_DIR, "monday_trend_and_amplitude.csv"), index=False, encoding="utf-8-sig")
pred_diag.to_csv(os.path.join(CLEAN_SCRIPT_DIR, "pred_diag_flows.csv"), encoding="utf-8-sig")
pred_off.to_csv(os.path.join(CLEAN_SCRIPT_DIR, "pred_offdiag_flows.csv"), encoding="utf-8-sig")
pred_total.to_csv(os.path.join(CLEAN_SCRIPT_DIR, "pred_total_flows.csv"), encoding="utf-8-sig")

# =========================================================================
# 8. 產出對比圖與表格視覺化
# =========================================================================
print("[6/8] 繪製對角線與非對角線專用 Flow Matching 基準圖...")
def plot_flow_matching_benchmark(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    flow_type: str,
    nrmse_val: float,
    valid_grids: list,
    grid_class_lookup: dict,
    class_metadata: dict,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    output_path: str,
    dpi: int = 220
):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 3, figsize=(22, 11.5), dpi=dpi)
    fig.patch.set_facecolor('#070c18')
    
    fig.suptitle(
        f"HuMob 2026: Flow Matching (OT-FM) ({flow_type} Flow) | NRMSE: {nrmse_val:.4f}",
        fontsize=13, fontweight='bold', color='#f8fafc', y=0.982
    )

    gap_connect_start = gap_start - pd.Timedelta(days=1)
    gap_connect_end = gap_end + pd.Timedelta(days=1)
    full_dates = pd.date_range(gt_df.index.min(), pred_df.index.max(), freq='D')

    for c_id in range(1, 10):
        r, c = (c_id - 1) // 3, (c_id - 1) % 3
        ax = axes[r, c]
        ax.set_facecolor('#0d1527')
        
        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
        if not c_grids:
            ax.set_visible(False)
            continue

        gt_mean = gt_df.loc[:, c_grids].mean(axis=1).reindex(full_dates)
        pred_mean = pred_df.loc[:, c_grids].mean(axis=1).reindex(full_dates)

        gt_plot = gt_mean.copy()
        gt_plot.loc[(gt_plot.index >= gap_start) & (gt_plot.index <= gap_end)] = np.nan

        mask_gap_span = (pred_mean.index >= gap_connect_start) & (pred_mean.index <= gap_connect_end)
        otfm_gap_slice = pred_mean.loc[mask_gap_span]

        ax.axvspan(gap_start, gap_end, color='#45271d', alpha=0.55, zorder=1)
        ax.plot(gt_plot.index, gt_plot, color='#f43f5e', linewidth=1.1, alpha=0.9, zorder=2)
        ax.plot(otfm_gap_slice.index, otfm_gap_slice, color='#2dd4bf', linewidth=1.3, alpha=0.98, zorder=3)

        class_title = f"Class {c_id:02d}: {class_metadata[c_id]['name']} (N={len(c_grids)})"
        ax.set_title(class_title, fontsize=9.5, fontweight='bold', color='#cbd5e1', pad=6)
        ax.grid(True, color='#1e293b', linestyle=':', alpha=0.5)
        
        ax.tick_params(colors='#64748b', labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.set_xlim(pd.to_datetime("2023-11-01"), PRED_END)

    legend_elements = [
        matplotlib.patches.Patch(facecolor='#45271d', edgecolor='none', alpha=0.8, label='Gap'),
        plt.Line2D([0], [0], color='#f43f5e', lw=1.3, label='Actual Flow'),
        plt.Line2D([0], [0], color='#2dd4bf', lw=1.5, label='Flow Matching Model')
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        fontsize=9.5,
        frameon=False
    )

    plt.tight_layout(rect=[0.02, 0.045, 0.98, 0.96])
    safe_save_fig(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"✓ 已產出 {flow_type} 專用對比圖: {os.path.basename(output_path)}")

plot_flow_matching_benchmark(
    gt_df=diag_df, pred_df=pred_diag, flow_type="Diagonal", nrmse_val=NRMSE_diag,
    valid_grids=valid_grids, grid_class_lookup=grid_class_lookup, class_metadata=CLASS_METADATA,
    gap_start=GAP_START, gap_end=GAP_END,
    output_path=os.path.join(CLEAN_SCRIPT_DIR, "humob_benchmark_diagonal_flow.png")
)

plot_flow_matching_benchmark(
    gt_df=offdiag_df, pred_df=pred_off, flow_type="Off-Diagonal", nrmse_val=NRMSE_offdiag,
    valid_grids=valid_grids, grid_class_lookup=grid_class_lookup, class_metadata=CLASS_METADATA,
    gap_start=GAP_START, gap_end=GAP_END,
    output_path=os.path.join(CLEAN_SCRIPT_DIR, "humob_benchmark_offdiag_flow.png")
)

print("[7/8] 渲染 9 大類別評估指標表格圖片...")
def render_nrmse_table_image(df_table, output_path, dpi=300):
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=dpi)
    fig.patch.set_facecolor('#ffffff')
    ax.axis('off')

    headers = ["class\nid", "class name", "NRMSE_di\nag", "NRMSE_o\nff", "combined\nNRMSE"]
    cell_data = []
    for _, row in df_table.iterrows():
        fmt_diag = "0" if np.isclose(row["nrmse_diag"], 0.0, atol=1e-4) else f"{row['nrmse_diag']:.2f}"
        fmt_off = "0" if np.isclose(row["nrmse_off"], 0.0, atol=1e-4) else f"{row['nrmse_off']:.2f}"
        fmt_comb = "0" if np.isclose(row["combined"], 0.0, atol=1e-4) else f"{row['combined']:.2f}"
        cell_data.append([row["class_id"], row["class_name"], fmt_diag, fmt_off, fmt_comb])

    table = ax.table(
        cellText=cell_data, colLabels=headers, cellLoc='left', loc='center',
        colWidths=[0.14, 0.40, 0.15, 0.15, 0.16]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.85)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#a0a0a0')
        cell.set_linewidth(0.8)
        cell.set_facecolor('#ffffff')
        cell.set_text_props(color='#111827')
        cell.get_text().set_horizontalalignment('left')

    plt.tight_layout(pad=0.5)
    safe_save_fig(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"✓ 已產出評估指標表格圖片: {os.path.basename(output_path)}")

render_nrmse_table_image(
    df_metrics_table, 
    os.path.join(CLEAN_SCRIPT_DIR, "humob_classwise_nrmse_table.png"),
    dpi=300
)

print("[8/8] 繪製全域總流量圖...")
plt.style.use('dark_background')
fig1, axes1 = plt.subplots(3, 3, figsize=(22, 12), dpi=220)
fig1.patch.set_facecolor('#070c18')
fig1.suptitle(
    f"HuMob 2026: 9-Class Waveform Benchmark ({num_nodes} Grids) | Combined NRMSE: {combined_nrmse:.4f}\n"
    f"(Diag NRMSE: {NRMSE_diag:.4f} | Off-Diag NRMSE: {NRMSE_offdiag:.4f})",
    fontsize=13, fontweight='bold', color='#f8fafc', y=0.985
)

full_date_range_global = pd.date_range(raw_total.index.min(), pred_total.index.max(), freq='D')

for c_id in range(1, 10):
    ax = axes1[(c_id - 1) // 3, (c_id - 1) % 3]
    ax.set_facecolor('#0d1527')
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    if not c_grids: continue

    gt = raw_total.loc[:, c_grids].mean(axis=1).reindex(full_date_range_global)
    gt.loc[(gt.index >= GAP_START) & (gt.index <= GAP_END)] = np.nan
    base = macro_total.loc[:, c_grids].mean(axis=1).reindex(full_date_range_global)
    pred = pred_total.loc[:, c_grids].mean(axis=1).reindex(full_date_range_global)

    ax.axvspan(GAP_START, GAP_END, color='#45271d', alpha=0.52)
    ax.plot(base.index, base, color='#94a3b8', linestyle='--', linewidth=1.1, alpha=0.8)
    ax.plot(pred.index, pred, color='#2dd4bf', linewidth=1.3, alpha=0.95)
    ax.plot(gt.index, gt, color='#f43f5e', linewidth=1.1, alpha=0.9)
    ax.set_title(f"Class {c_id:02d}: {CLASS_METADATA[c_id]['name']} (N={len(c_grids)})", fontsize=9.5, fontweight='bold', color='#cbd5e1')
    ax.grid(True, color='#1e293b', linestyle=':', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

legend_elements = [
    matplotlib.patches.Patch(facecolor='#45271d', alpha=0.7, label='60-Day Missing Gap'),
    plt.Line2D([0], [0], color='#f43f5e', lw=1.3, label='Ground Truth (Observed)'),
    plt.Line2D([0], [0], color='#94a3b8', lw=1.2, linestyle='--', label='Learned Adaptive Plateau Baseline'),
    plt.Line2D([0], [0], color='#2dd4bf', lw=1.4, label='Zero-Centered OT-FM (RK4)')
]
fig1.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.012), ncol=4, fontsize=9.5,
            frameon=True, facecolor='#0a1020', edgecolor='#1e293b')
plt.tight_layout(rect=[0.02, 0.045, 0.98, 0.96])
safe_save_fig(fig1, os.path.join(CLEAN_SCRIPT_DIR, "humob_9class_waveform_benchmark.png"))
plt.close(fig1)

print("\n" + "=" * 95)
print(" 🏆 HuMob 2026 全流程運行完畢！")
print("   - 宏觀基底：自適應學習 Sigmoid (k, tau_m) 控制彈升速度與平穩高原收斂 (Plateau Effect)[cite: 2]")
print("   - 微觀高頻：OT-FM 執行零均值校準 (Zero-Centering)，完整保留有機波動，水位絕不塌陷[cite: 1]")
print("   - Class 6：維持實測樣板鎖定 (~46 水位)[cite: 1]")
print("=" * 95)
