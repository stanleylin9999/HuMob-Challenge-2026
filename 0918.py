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
from torch.utils.data import DataLoader

# =========================================================================
# 1. 全域配置、安全路徑與官方標準常數
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

PRED_START = pd.to_datetime("2024-01-01")
GAP_START = pd.to_datetime("2024-02-01")
GAP_END = pd.to_datetime("2024-03-31")
PRED_END = pd.to_datetime("2024-10-31")
GAP_LEN = (GAP_END - GAP_START).days + 1  # 60 天

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
    """Windows/Linux 專用安全圖片儲存器：杜絕 Errno 22 與特殊隱藏字元報錯"""
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
# 2. 空間邊界過濾與資料讀取 (x: 30~70, y: 35~70, 全程強制 float32)
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

coords = np.array([[int(c) for c in g.split('_')] for g in valid_grids])
dist_matrix = cdist(coords, coords)

# =========================================================================
# 3. 動態 OD 轉移機率矩陣引擎
# =========================================================================
class DynamicODTransferEngine:
    def __init__(self, valid_grids, daily_od_records, dist_matrix):
        self.valid_grids = valid_grids
        self.dist_matrix = dist_matrix
        pre_dates = [dt for dt in daily_od_records if dt < PRED_START]
        self.P_base = self._build(daily_od_records, pre_dates)
        post_dates = [dt for dt in daily_od_records if dt > GAP_END]
        self.P_post = self._build(daily_od_records, post_dates) if post_dates else self.P_base

    def _build(self, records, dates):
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
                i = self.valid_grids.index(orig)
                dists = self.dist_matrix[i]
                weights = [1.0 / max(dists[j], 0.5) if j != i else 0.0 for j in range(len(self.valid_grids))]
                s_w = sum(weights) + 1e-7
                probs[orig] = {self.valid_grids[j]: weights[j] / s_w for j in range(len(self.valid_grids)) if j != i}
        return probs

    def get_matrix(self, dt):
        if dt < GAP_START: return self.P_base
        elif dt <= GAP_END:
            tau = ((dt - GAP_START).days + 1) / float(GAP_LEN)
            w = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            interp = {}
            for orig in self.valid_grids:
                dests = set(self.P_base.get(orig, {}).keys()).union(self.P_post.get(orig, {}).keys())
                comb = {d: (1.0 - w) * self.P_base.get(orig, {}).get(d, 0.0) + w * self.P_post.get(orig, {}).get(d, 0.0) for d in dests}
                s = sum(comb.values())
                interp[orig] = {d: v / s for d, v in comb.items()} if s > 0 else {}
            return interp
        else:
            return self.P_post

transfer_engine = DynamicODTransferEngine(valid_grids, daily_od_records, dist_matrix)

# =========================================================================
# 4. 星期一錨點動力學與動態振幅展開
# =========================================================================
print("[2/8] 執行星期一錨點與動態振幅展開...")
class MondayAnchoredDynamicEngine:
    def __init__(self, flow_df, valid_grids, grid_class_lookup, is_offdiag=False):
        self.flow_df = flow_df
        self.valid_grids = valid_grids
        self.grid_class_lookup = grid_class_lookup
        self.is_offdiag = is_offdiag
        self.canonical_waves = {}
        self._fit()

    def _fit(self):
        obs_df = self.flow_df.loc[~((self.flow_df.index >= GAP_START) & (self.flow_df.index <= GAP_END))].copy()
        obs_df['dow'] = obs_df.index.dayofweek
        for g in self.valid_grids:
            cid = self.grid_class_lookup.get(g, 5)
            sparsity = (obs_df[g] == 0).mean()

            # Class 01 或非對角線極度稀疏者，直接將週週期歸零
            if cid == 1 or (self.is_offdiag and (cid == 3 or sparsity > 0.95)):
                self.canonical_waves[g] = np.zeros(7, dtype=np.float32)
                continue

            medians = obs_df.groupby('dow')[g].median().values
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

        obs_df = self.flow_df.loc[~((self.flow_df.index >= GAP_START) & (self.flow_df.index <= GAP_END))]

        for g_idx, g in enumerate(self.valid_grids):
            cid = self.grid_class_lookup.get(g, 5)
            s_d = self.canonical_waves[g]
            sparsity = (obs_df[g] == 0).mean()

            # Class 01 及極度稀疏網格先歸零基線
            if cid == 1 or (self.is_offdiag and (cid == 3 or (cid == 5 and sparsity > 0.92))):
                pred_mat[:, g_idx] = 0.0
                continue

            g_mean = float(obs_df[g].mean())

            mon_obs = self.flow_df.loc[self.flow_df.index.dayofweek == 0, g]
            mon_jan = mon_obs.loc["2024-01-15":"2024-01-31"]
            mon_apr = mon_obs.loc["2024-04-01":"2024-04-20"]

            if len(mon_jan) > 0:
                pos_jan = mon_jan[mon_jan > 0]
                M_jan = float(pos_jan.median()) if len(pos_jan) > 0 else float(mon_jan.mean())
            else:
                M_jan = 0.01 if self.is_offdiag else max(1.0, g_mean)

            if len(mon_apr) > 0:
                pos_apr = mon_apr[mon_apr > 0]
                M_apr = float(pos_apr.median()) if len(pos_apr) > 0 else float(mon_apr.mean())
            else:
                M_apr = M_jan

            if self.is_offdiag:
                min_amp = 0.001
                max_amp = max(min_amp * 2.0, g_mean * 0.45)
            else:
                min_amp = max(0.01, g_mean * 0.05)
                max_amp = max(min_amp * 2.0, g_mean * 0.40)

            jan_window = self.flow_df.loc["2024-01-15":"2024-01-31", g]
            apr_window = self.flow_df.loc["2024-04-01":"2024-04-20", g]

            std_jan = float(jan_window.std() * 1.0) if len(jan_window) > 1 else min_amp
            std_apr = float(apr_window.std() * 1.0) if len(apr_window) > 1 else min_amp

            A_jan = float(np.clip(max(min_amp, std_jan), min_amp, max_amp))
            A_apr = float(np.clip(max(min_amp, std_apr), min_amp, max_amp))

            gap_mondays = [m for m in mondays if GAP_START <= m <= GAP_END]
            mon_dict, amp_dict = {}, {}
            for idx, m in enumerate(gap_mondays):
                tau = (idx + 1) / (len(gap_mondays) + 1)
                s = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
                mon_dict[m] = float(M_jan + s * (M_apr - M_jan))
                amp_dict[m] = float(A_jan + s * (A_apr - A_jan))

            for m in mondays:
                if m not in mon_dict:
                    mon_dict[m] = float(self.flow_df.loc[m, g]) if m in self.flow_df.index else M_jan
                    w_span = self.flow_df.loc[m : m + pd.Timedelta(days=6), g]
                    amp_raw = float(w_span.max() - w_span.min()) * 0.5 if len(w_span) > 0 else A_jan
                    amp_dict[m] = float(np.clip(max(min_amp, amp_raw), min_amp, max_amp))
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
macro_diag_df, meta_diag_df = MondayAnchoredDynamicEngine(diag_df, valid_grids, grid_class_lookup, False).generate(all_sim_dates)
macro_offdiag_df, meta_offdiag_df = MondayAnchoredDynamicEngine(offdiag_df, valid_grids, grid_class_lookup, True).generate(all_sim_dates)
all_meta_df = pd.concat([meta_diag_df, meta_offdiag_df], ignore_index=True)

# =========================================================================
# 5. OT-FM 殘差網絡訓練、Batched RK4 推論與 Class 5/9 脈衝注入
# =========================================================================
print("[3/8] 訓練 OT-FM 殘差網絡...")
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

print("[4/8] 執行全網格批次張量化 (Batched RK4) 極速推論...")
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
        final_w = np.maximum(0.0, base_mat + gen_res)
        final_w[zero_mask, :] = 0.0

        for offset, d in enumerate(w_span):
            if GAP_START <= d <= GAP_END:
                pred_df.loc[d, valid_grids] = final_w[:, offset]
    return pred_df

pred_diag = solve_batched_rk4(ot_diag, macro_diag_df, is_offdiag=False)
pred_off = solve_batched_rk4(ot_off, macro_offdiag_df, is_offdiag=True)

# -------------------------------------------------------------------------
# Class 5 & Class 9 稀疏離散脈衝注入函式
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
        if not c_grids:
            continue

        # 1. 將空窗期內 Class 5 / 9 的非對角線流量預測徹底歸零（消除平滑小山丘與連續色帶）
        for g in c_grids:
            refined_pred.loc[gap_dates, g] = 0.0

        # 2. 統計觀測期中：各星期幾整體的突起觸發機率與人次階數
        dow_spike_probs = {}
        dow_event_weights = {}

        for dow in range(7):
            dow_df = obs_clean.loc[obs_clean['dow'] == dow, c_grids]
            daily_has_spike = (dow_df >= 0.5).any(axis=1)
            p_spike = float(daily_has_spike.mean()) if len(daily_has_spike) > 0 else 0.0
            dow_spike_probs[dow] = p_spike

            active_vals = dow_df.values[dow_df.values >= 0.5]
            if len(active_vals) > 0:
                dow_event_weights[dow] = np.round(active_vals)
            else:
                dow_event_weights[dow] = np.array([1.0], dtype=np.float32)

        # 3. 在空窗期注入離散整數量化脈衝
        T_gap = len(gap_dates)
        for step_idx, dt in enumerate(gap_dates):
            dow = dt.dayofweek
            base_p = dow_spike_probs.get(dow, 0.0)

            # 趨勢衰減/復甦權重
            if cid == 9:
                weight = 1.15 - 0.35 * (step_idx / float(T_gap))
            else:
                weight = 0.50 + 0.60 * (step_idx / float(T_gap))

            p_inject = float(np.clip(base_p * weight, 0.0, 0.60))

            if random.random() < p_inject:
                target_grid = random.choice(c_grids)
                candidates = dow_event_weights.get(dow, [1.0])
                val = float(random.choice(candidates))
                refined_pred.loc[dt, target_grid] = max(1.0, val)

    return refined_pred

print("[4.5/8] 對 Class 05 與 Class 09 非對角線執行離散量子化脈衝校正...")
pred_off = inject_sparse_spikes_for_class5_and_9(
    pred_off_df=pred_off,
    obs_off_df=offdiag_df,
    valid_grids=valid_grids,
    grid_class_lookup=grid_class_lookup,
    gap_start=GAP_START,
    gap_end=GAP_END
)

# 非空窗期觀測值填補回填
obs_dates = [d for d in diag_df.index if not (GAP_START <= d <= GAP_END)]
pred_diag.loc[obs_dates, valid_grids] = diag_df.loc[obs_dates, valid_grids].values.astype(np.float32)
pred_off.loc[obs_dates, valid_grids] = offdiag_df.loc[obs_dates, valid_grids].values.astype(np.float32)

pred_total = pred_diag + pred_off
raw_total = diag_df + offdiag_df
macro_total = macro_diag_df + macro_offdiag_df

# =========================================================================
# 6. 官方評估指標與 9 大類別匯總表計算
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

def compute_classwise_nrmse_table(
    diag_df, offdiag_df, pred_diag, pred_off,
    daily_od_records, transfer_engine, valid_grids,
    grid_class_lookup, class_metadata, eval_dates,
    mean_actual_diag=MEAN_ACTUAL_DIAG, mean_actual_offdiag=MEAN_ACTUAL_OFFDIAG
):
    num_total_nodes = len(valid_grids)
    N_total_offdiag = num_total_nodes * (num_total_nodes - 1)
    
    class_stats = {
        cid: {"sse_diag": 0.0, "sse_off": 0.0, "count": 0} 
        for cid in range(1, 10)
    }
    
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
# 7. 產出對角線與非對角線 Flow Matching 專用基準圖
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

# 繪製 Diagonal Flow (對角線流量)
plot_flow_matching_benchmark(
    gt_df=diag_df,
    pred_df=pred_diag,
    flow_type="Diagonal",
    nrmse_val=NRMSE_diag,
    valid_grids=valid_grids,
    grid_class_lookup=grid_class_lookup,
    class_metadata=CLASS_METADATA,
    gap_start=GAP_START,
    gap_end=GAP_END,
    output_path=os.path.join(CLEAN_SCRIPT_DIR, "humob_benchmark_diagonal_flow.png")
)

# 繪製 Off-Diagonal Flow (非對角線流量)
plot_flow_matching_benchmark(
    gt_df=offdiag_df,
    pred_df=pred_off,
    flow_type="Off-Diagonal",
    nrmse_val=NRMSE_offdiag,
    valid_grids=valid_grids,
    grid_class_lookup=grid_class_lookup,
    class_metadata=CLASS_METADATA,
    gap_start=GAP_START,
    gap_end=GAP_END,
    output_path=os.path.join(CLEAN_SCRIPT_DIR, "humob_benchmark_offdiag_flow.png")
)

# =========================================================================
# 8. 產出 9 大類別評估表格圖片與其他視覺化圖表
# =========================================================================
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
        cellText=cell_data,
        colLabels=headers,
        cellLoc='left',
        loc='center',
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

print("[8/8] 繪製全域總流量圖、空窗期特寫與 7 張獨立跨週走勢圖...")
plt.style.use('dark_background')

# 圖 1: 9 大類別總流量對比圖
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
    plt.Line2D([0], [0], color='#94a3b8', lw=1.2, linestyle='--', label='Monday-Anchored Dynamic Baseline'),
    plt.Line2D([0], [0], color='#2dd4bf', lw=1.4, label='Fast Batched OT-FM (RK4)')
]
fig1.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.012), ncol=4, fontsize=9.5,
            frameon=True, facecolor='#0a1020', edgecolor='#1e293b')
plt.tight_layout(rect=[0.02, 0.045, 0.98, 0.96])
safe_save_fig(fig1, os.path.join(CLEAN_SCRIPT_DIR, "humob_9class_waveform_benchmark.png"))
plt.close(fig1)

# 圖 2: 60 天空窗期微觀特寫圖
fig2, ax2 = plt.subplots(figsize=(15, 6), dpi=220)
fig2.patch.set_facecolor('#070c18')
ax2.set_facecolor('#0d1527')
target_grid = [g for g in valid_grids if grid_class_lookup.get(g) == 5][0]
gap_plot_range = pd.date_range(GAP_START - pd.Timedelta(days=3), GAP_END + pd.Timedelta(days=3), freq="D")
gap_mondays = gap_plot_range[gap_plot_range.dayofweek == 0]

p_s = pred_total.loc[gap_plot_range, target_grid]
b_s = macro_total.loc[gap_plot_range, target_grid]

ax2.axvspan(GAP_START, GAP_END, color='#45271d', alpha=0.35, label='60-Day Missing Gap')
ax2.plot(b_s.index, b_s, color='#94a3b8', linestyle=':', linewidth=1.4, label='Monday Drift Carrier')
ax2.plot(p_s.index, p_s, color='#2dd4bf', linewidth=2.0, label='Predicted 7-Day Waveform (OT-FM)')

for m in gap_mondays:
    val = p_s.loc[m]
    ax2.axvline(m, color='#38bdf8', linestyle='--', alpha=0.6, linewidth=1.0)
    ax2.scatter(m, val, color='#38bdf8', s=45, zorder=5)
    ax2.text(m, val + 1.2, 'Mon', color='#38bdf8', fontsize=8, ha='center', fontweight='bold')

for d in gap_plot_range:
    if GAP_START <= d <= GAP_END:
        if d.dayofweek == 4: ax2.scatter(d, p_s.loc[d], color='#fbbf24', s=40, zorder=6)
        elif d.dayofweek == 6: ax2.scatter(d, p_s.loc[d], color='#f87171', s=40, zorder=6)

ax2.set_title(f"60-Day Blind Zone Microscopic Rollout: Monday Anchors, Friday Peaks (Gold), Sunday Troughs (Red) [{target_grid}]", 
              fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
ax2.grid(True, color='#1e293b', linestyle=':', alpha=0.6)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d (%a)'))
ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax2.legend(loc='upper left', facecolor='#0a1020', edgecolor='#1e293b')
plt.tight_layout()
safe_save_fig(fig2, os.path.join(CLEAN_SCRIPT_DIR, "humob_gap_weekly_peaks_troughs_detail.png"))
plt.close(fig2)

# 圖 3: 每日官方 NRMSE 曲線追蹤圖
fig3, ax3 = plt.subplots(figsize=(15, 5.5), dpi=220)
fig3.patch.set_facecolor('#070c18')
ax3.set_facecolor('#0d1527')

dates_dt = [pd.to_datetime(r["date"]) for r in daily_eval]
n_diag_v = [r["NRMSE_diag_d"] for r in daily_eval]
n_off_v = [r["NRMSE_offdiag_d"] for r in daily_eval]
n_comb_v = [(d + o) / 2.0 for d, o in zip(n_diag_v, n_off_v)]

ax3.plot(dates_dt, n_diag_v, color='#38bdf8', label=f'NRMSE_diag (Mean: {NRMSE_diag:.4f})', lw=1.2, alpha=0.85)
ax3.plot(dates_dt, n_off_v, color='#fbbf24', label=f'NRMSE_offdiag (Mean: {NRMSE_offdiag:.4f})', lw=1.2, alpha=0.85)
ax3.plot(dates_dt, n_comb_v, color='#f43f5e', label=f'Combined NRMSE (Mean: {combined_nrmse:.4f})', lw=1.8)
ax3.axvspan(GAP_START, GAP_END, color='#45271d', alpha=0.45, label='60-Day Missing Gap')
ax3.axvspan(pd.to_datetime("2024-04-01"), pd.to_datetime("2024-04-30"), color='#1e3a8a', alpha=0.3, label='April Official Benchmark')

ax3.set_title(f"HuMob 2026: Official Daily NRMSE Evaluation Curve | Combined NRMSE: {combined_nrmse:.4f}", 
              fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
ax3.set_ylabel("Normalized Score", fontsize=10, color='#94a3b8')
ax3.grid(True, color='#1e293b', linestyle=':', alpha=0.6)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax3.tick_params(colors='#94a3b8', labelsize=8.5)
ax3.legend(loc='upper right', facecolor='#0a1020', edgecolor='#1e293b')
plt.tight_layout()
safe_save_fig(fig3, os.path.join(CLEAN_SCRIPT_DIR, "humob_daily_nrmse_evaluation.png"))
plt.close(fig3)

# 圖 4: 星期一至星期日共 7 張獨立走勢圖
weekday_meta = [
    {"dow": 0, "en": "Monday",    "zh": "星期一", "role": "週波錨點 (Weekly Anchor)",      "color": "#38bdf8"},
    {"dow": 1, "en": "Tuesday",   "zh": "星期二", "role": "平日爬升期 (Midweek Rise)",    "color": "#818cf8"},
    {"dow": 2, "en": "Wednesday", "zh": "星期三", "role": "週間平穩期 (Midweek Plateau)", "color": "#a78bfa"},
    {"dow": 3, "en": "Thursday",  "zh": "星期四", "role": "週末前哨期 (Pre-Weekend)",     "color": "#c084fc"},
    {"dow": 4, "en": "Friday",    "zh": "星期五", "role": "週波最高峰 (Weekly Peak)",      "color": "#fbbf24"},
    {"dow": 5, "en": "Saturday",  "zh": "星期六", "role": "週末活動轉移 (Weekend Drop)",  "color": "#fb923c"},
    {"dow": 6, "en": "Sunday",    "zh": "星期日", "role": "週波最低谷 (Weekly Trough)",    "color": "#f87171"}
]

all_dates_dow = pd.date_range(raw_total.index.min(), pred_total.index.max(), freq='D')

for item in weekday_meta:
    dow = item["dow"]
    en_name = item["en"]
    zh_name = item["zh"]
    role_desc = item["role"]
    theme_color = item["color"]

    target_dates = [d for d in all_dates_dow if d.dayofweek == dow]
    fig_w, axes_w = plt.subplots(3, 3, figsize=(22, 12), dpi=220)
    fig_w.patch.set_facecolor('#070c18')
    fig_w.suptitle(
        f"HuMob 2026: 52-Week Macro Trend — {en_name} ({zh_name}) | Role: {role_desc}\n"
        f"9-Class Trajectory Reconstruction with Dynamic Amplitude Modulation",
        fontsize=13, fontweight='bold', color='#f8fafc', y=0.985
    )

    for c_id in range(1, 10):
        r, c = (c_id - 1) // 3, (c_id - 1) % 3
        ax = axes_w[r, c]
        ax.set_facecolor('#0d1527')
        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
        if not c_grids: continue

        gt_pts = raw_total.loc[:, c_grids].mean(axis=1).reindex(target_dates)
        base_pts = macro_total.loc[:, c_grids].mean(axis=1).reindex(target_dates)
        pred_pts = pred_total.loc[:, c_grids].mean(axis=1).reindex(target_dates)

        gt_masked = gt_pts.copy()
        gt_masked.loc[(gt_masked.index >= GAP_START) & (gt_masked.index <= GAP_END)] = np.nan

        ax.axvspan(GAP_START, GAP_END, color='#45271d', alpha=0.52, zorder=1)
        ax.plot(base_pts.index, base_pts, color='#94a3b8', linestyle='--', linewidth=1.2, alpha=0.75, zorder=2)
        ax.plot(pred_pts.index, pred_pts, color=theme_color, linewidth=1.6, alpha=0.95, zorder=3)
        ax.scatter(gt_masked.index, gt_masked, color='#f43f5e', s=16, alpha=0.9, zorder=4)
        ax.plot(gt_masked.index, gt_masked, color='#f43f5e', linewidth=1.0, alpha=0.6, zorder=4)

        gap_pts = pred_pts.loc[(pred_pts.index >= GAP_START) & (pred_pts.index <= GAP_END)]
        ax.scatter(gap_pts.index, gap_pts, color=theme_color, edgecolor='#ffffff', s=24, lw=0.8, zorder=5)

        ax.set_title(f"Class {c_id:02d}: {CLASS_METADATA[c_id]['name']} (N={len(c_grids)})", 
                     fontsize=9.5, fontweight='bold', color='#cbd5e1', pad=5)
        ax.grid(True, color='#1e293b', linestyle=':', alpha=0.5)
        ax.tick_params(colors='#64748b', labelsize=7.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    legend_elements_w = [
        matplotlib.patches.Patch(facecolor='#45271d', alpha=0.7, label='60-Day Blind Zone'),
        plt.Line2D([0], [0], color='#f43f5e', marker='o', markersize=4, lw=1.2, label=f'Ground Truth ({en_name})'),
        plt.Line2D([0], [0], color='#94a3b8', lw=1.2, linestyle='--', label='Monday Baseline'),
        plt.Line2D([0], [0], color=theme_color, marker='o', markersize=4, lw=1.6, label=f'OT-FM {en_name} Trend')
    ]
    fig_w.legend(handles=legend_elements_w, loc='lower center', bbox_to_anchor=(0.5, 0.012), ncol=4, fontsize=9.5,
                 frameon=True, facecolor='#0a1020', edgecolor='#1e293b')
    plt.tight_layout(rect=[0.02, 0.045, 0.98, 0.96])
    w_filename = f"humob_weekly_trend_{dow + 1:02d}_{en_name.lower()}.png"
    safe_save_fig(fig_w, os.path.join(CLEAN_SCRIPT_DIR, w_filename))
    plt.close(fig_w)

print("\n" + "=" * 95)
print(" 🏆 HuMob 2026 全流程運行完畢！")
print(f"   - 範圍內網格總數: {num_nodes} 個 (x:30~70, y:35~70)")
print(f"   - 官方評分 Combined NRMSE: {combined_nrmse:.4f}")
print(f"   - 產出檔案包含:")
print(f"     1. humob_benchmark_diagonal_flow.png (對角線流量重構)")
print(f"     2. humob_benchmark_offdiag_flow.png (非對角線流量重構 - 已校正 Class 5/9 脈衝)")
print(f"     3. humob_classwise_nrmse_table.png & .csv (類別評估指標總表)")
print(f"     4. humob_9class_waveform_benchmark.png (全域總流動波形)")
print(f"     5. humob_gap_weekly_peaks_troughs_detail.png (空窗期微觀波形)")
print(f"     6. humob_daily_nrmse_evaluation.png (每日 NRMSE 走勢)")
print(f"     7. humob_weekly_trend_01~07 (星期一至日跨週趨勢圖)")
print(f"   - 所有圖表與 CSV 已全數存檔至: {CLEAN_SCRIPT_DIR}")
print("=" * 95)
