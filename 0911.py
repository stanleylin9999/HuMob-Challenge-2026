import os
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
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 1. 全域配置與官方標準常數
# =========================================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN_ACTUAL_DIAG = 26.57
MEAN_ACTUAL_OFFDIAG = 0.0176
WEIGHT_DIAG = 0.5
WEIGHT_OFFDIAG = 0.5

PRED_START = pd.to_datetime("2024-01-01")
GAP_START = pd.to_datetime("2024-02-01")
GAP_END = pd.to_datetime("2024-03-31")
PRED_END = pd.to_datetime("2024-10-31")
GAP_LEN = (GAP_END - GAP_START).days + 1  # 60 天

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TSV_PATH = os.path.join(SCRIPT_DIR, "humob2026-dataset.tsv")
SPECIFIC_CLASS_DIR = r"C:\Users\User\Desktop\人口預測專案\人口預測專案3\humob2026\data\output\module05\classification\by_class"
FALLBACK_CLASS_DIR = os.path.join(SCRIPT_DIR, "humob2026", "data", "output", "module05", "classification", "by_class")
BY_CLASS_DIR = SPECIFIC_CLASS_DIR if os.path.exists(SPECIFIC_CLASS_DIR) else FALLBACK_CLASS_DIR

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "humob_flow_matching_sota_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

if os.path.exists(TSV_PATH):
    raw_df = pd.read_csv(TSV_PATH, sep="\t", names=["date", "od_matrix_raw"])
    raw_df['date_dt'] = pd.to_datetime(raw_df['date'].astype(str), format='%Y%m%d')
    raw_df = raw_df.sort_values('date_dt').reset_index(drop=True)
else:
    print(" -> 未偵測到原始資料集，生成合成資料供管線驗證...")
    date_rng = pd.date_range("2023-11-01", "2024-10-31", freq="D")
    synth_grids = [f"{y}_{x}" for y in range(40, 45) for x in range(40, 45)]
    records = []
    for d in date_rng:
        day_dict = {}
        for g in synth_grids:
            if not (GAP_START <= d <= GAP_END):
                day_dict[g] = {g: np.random.poisson(30), synth_grids[0]: np.random.poisson(1)}
        records.append({"date": d.strftime('%Y%m%d'), "od_matrix_raw": str(day_dict), "date_dt": d})
    raw_df = pd.DataFrame(records)

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
    valid_grids = diag_df.columns[diag_df[pre_mask].mean() >= 0.001].tolist() if pre_mask.any() else diag_df.columns.tolist()

for i, g in enumerate(valid_grids):
    if g not in grid_class_lookup:
        grid_class_lookup[g] = (i % 9) + 1

diag_df = diag_df[valid_grids]
offdiag_df = offdiag_df[valid_grids]
num_nodes = len(valid_grids)
print(f"✓ 有效網格節點數: {num_nodes}")

coords = np.array([[int(c) for c in g.split('_')] for g in valid_grids])
dist_matrix = cdist(coords, coords)
knn_weights = np.zeros_like(dist_matrix)
for i in range(num_nodes):
    neighbors = np.argsort(dist_matrix[i])[1:min(5, num_nodes)]
    w = 1.0 / np.maximum(dist_matrix[i, neighbors], 0.5)
    knn_weights[i, neighbors] = w / (w.sum() + 1e-7)
spatial_knn = pd.DataFrame(knn_weights, index=valid_grids, columns=valid_grids)

# =========================================================================
# 3. OD 轉移引擎
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
# 4. 自適應宏觀骨架 (Adaptive Drift Baseline)
# =========================================================================
class AdaptiveMacroTrendEngine:
    def __init__(self, flow_df, valid_grids, grid_class_lookup):
        self.flow_df = flow_df
        self.valid_grids = valid_grids
        self.grid_class_lookup = grid_class_lookup
        
        pre_df = flow_df.loc[flow_df.index < PRED_START, valid_grids]
        self.M_pre = robust_median(pre_df).clip(lower=0.1) if len(pre_df) > 0 else flow_df[valid_grids].median().clip(lower=0.1)
        self.max_ceiling = (pre_df.quantile(0.99).fillna(50.0) * 1.4 + 2.0) if len(pre_df) > 0 else flow_df[valid_grids].max() + 5.0
        
        jan_sub = flow_df.loc["2024-01-20":"2024-01-31", valid_grids]
        self.l_jan_end = robust_median(jan_sub).clip(lower=0.0) if len(jan_sub) > 0 else self.M_pre
        self.jan_peaks = flow_df.loc["2024-01-01":"2024-01-06", valid_grids].max().fillna(self.l_jan_end) if "2024-01-01" in flow_df.index else self.l_jan_end
        
        apr_sub = flow_df.loc["2024-04-01":"2024-04-14", valid_grids] if "2024-04-01" in flow_df.index else jan_sub
        self.l_resume_start = robust_median(apr_sub).clip(lower=0.0) if len(apr_sub) > 0 else self.l_jan_end
        
        post_sub = flow_df.loc["2024-04-01":"2024-10-31", valid_grids] if "2024-04-01" in flow_df.index else apr_sub
        self.l_long_term = robust_median(post_sub).clip(lower=0.0) if len(post_sub) > 0 else self.l_resume_start

    def get_macro_trend(self, dt: pd.Timestamp) -> pd.Series:
        gap_span = GAP_LEN
        if dt < GAP_START:
            day_idx = (dt - PRED_START).days
            tau = max(0.0, day_idx / 30.0)
            jan_init = self.flow_df.loc["2024-01-01":"2024-01-04", self.valid_grids].median().fillna(self.l_jan_end) if "2024-01-01" in self.flow_df.index else self.l_jan_end
            mu_t = jan_init + (tau ** 1.2) * (self.l_jan_end - jan_init)
            for g in self.valid_grids:
                c = self.grid_class_lookup.get(g, 5)
                p_val = self.jan_peaks[g]
                if c in [3, 7, 8]:
                    if day_idx <= 3:
                        mu_t[g] = jan_init[g] + (p_val - jan_init[g]) * (max(0, day_idx) / 3.0)
                    else:
                        decay_tau = (day_idx - 3) / 27.0
                        mu_t[g] = self.l_jan_end[g] + (p_val - self.l_jan_end[g]) * np.exp(-2.8 * max(0.0, decay_tau))
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

all_sim_dates = pd.date_range(diag_df.index.min(), PRED_END, freq="D")
macro_baseline_diag = pd.DataFrame([trend_diag_engine.get_macro_trend(d) for d in all_sim_dates], index=all_sim_dates)
macro_baseline_offdiag = pd.DataFrame([trend_offdiag_engine.get_macro_trend(d) for d in all_sim_dates], index=all_sim_dates)

# =========================================================================
# 5. 最優傳輸流匹配 (OT-FM) 模組與神經網絡訓練
# =========================================================================
class TimeSeriesResidualOTDataset(Dataset):
    def __init__(self, flow_df, baseline_df, valid_grids, grid_class_lookup, gap_len=60, num_samples=3000):
        self.samples = []
        valid_dates = [d for d in flow_df.index if not (GAP_START <= d <= GAP_END)]
        flow_sub = flow_df.loc[valid_dates]
        base_sub = baseline_df.loc[valid_dates]
        residual_df = flow_sub - base_sub
        
        consec_blocks = []
        cur_block = []
        for i in range(len(valid_dates) - 1):
            cur_block.append(valid_dates[i])
            if (valid_dates[i+1] - valid_dates[i]).days > 1:
                if len(cur_block) >= gap_len + 14:
                    consec_blocks.append(cur_block)
                cur_block = []
        if len(cur_block) >= gap_len + 14:
            consec_blocks.append(cur_block)

        if not consec_blocks:
            consec_blocks = [valid_dates]

        for _ in range(num_samples):
            blk = random.choice(consec_blocks)
            if len(blk) < gap_len + 2: continue
            start_i = random.randint(1, len(blk) - gap_len - 1)
            target_dates = blk[start_i : start_i + gap_len]
            pre_date = blk[start_i - 1]
            post_date = blk[start_i + gap_len]
            
            g = random.choice(valid_grids)
            c_id = grid_class_lookup.get(g, 5) - 1
            
            target_res = residual_df.loc[target_dates, g].values.astype(np.float32)
            base_seq = base_sub.loc[target_dates, g].values.astype(np.float32)
            b_left = residual_df.loc[pre_date, g]
            b_right = residual_df.loc[post_date, g]
            
            self.samples.append((
                target_res,
                base_seq,
                np.array([b_left, b_right], dtype=np.float32),
                c_id
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        res, base, bounds, cid = self.samples[idx]
        return torch.from_numpy(res).unsqueeze(0), torch.from_numpy(base).unsqueeze(0), torch.from_numpy(bounds), cid

class OTFMResidualUNetBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, dim * 2))
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.gn = nn.GroupNorm(8, dim)

    def forward(self, x, cond):
        scale, shift = self.mlp(cond).unsqueeze(-1).chunk(2, dim=1)
        h = self.conv1(F.silu(self.gn(x)))
        h = h * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return x + h

class FlowMatchingTimeSeriesBridge(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=9):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, hidden_dim)
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bound_proj = nn.Linear(2, hidden_dim)
        
        self.in_conv = nn.Conv1d(2, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([
            OTFMResidualUNetBlock(hidden_dim, hidden_dim) for _ in range(4)
        ])
        self.out_conv = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)

    def forward(self, x_t, t, base_seq, bounds, c_id):
        cond = self.time_emb(t) + self.class_emb(c_id) + self.bound_proj(bounds)
        inp = torch.cat([x_t, base_seq], dim=1)
        h = self.in_conv(inp)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_conv(h)

def train_flow_matching(flow_df, baseline_df, valid_grids, grid_class_lookup, epochs=25, tag="Diagonal"):
    print(f" -> 啟動最優傳輸流匹配 (OT-FM) 神經網絡訓練 [{tag}]...")
    dataset = TimeSeriesResidualOTDataset(flow_df, baseline_df, valid_grids, grid_class_lookup, gap_len=GAP_LEN, num_samples=2500)
    if len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    
    model = FlowMatchingTimeSeriesBridge(hidden_dim=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x_real, base_seq, bounds, c_id in loader:
            x_real = x_real.to(DEVICE)
            base_seq = base_seq.to(DEVICE)
            bounds = bounds.to(DEVICE)
            c_id = c_id.to(DEVICE)
            B, _, T = x_real.shape
            
            t = torch.rand(B, 1, device=DEVICE)
            x_0 = torch.randn_like(x_real)
            
            t_exp = t.unsqueeze(-1)
            x_t = (1.0 - t_exp) * x_0 + t_exp * x_real
            target_v = x_real - x_0
            
            pred_v = model(x_t, t, base_seq, bounds, c_id)
            loss = F.mse_loss(pred_v, target_v)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * B
        scheduler.step()
    return model

ot_model_diag = train_flow_matching(diag_df, macro_baseline_diag, valid_grids, grid_class_lookup, epochs=25, tag="Diagonal")
ot_model_offdiag = train_flow_matching(offdiag_df, macro_baseline_offdiag, valid_grids, grid_class_lookup, epochs=25, tag="Off-Diagonal")

# =========================================================================
# 6. 空窗期 RK4 ODE 數值求解與推論 (批次化並行 Ensemble)
# =========================================================================
@torch.no_grad()
def solve_ot_gap_inpainting(
    model, flow_df, baseline_df, valid_grids, grid_class_lookup, 
    is_offdiag=False, steps=15, solver="rk4", ensemble_size=8
):
    gap_dates = pd.date_range(GAP_START, GAP_END, freq="D")
    pre_day = GAP_START - pd.Timedelta(days=1)
    post_day = GAP_END + pd.Timedelta(days=1)
    
    inpainted_df = flow_df.copy().reindex(all_sim_dates)
    if model is None:
        inpainted_df.loc[gap_dates] = baseline_df.loc[gap_dates]
        return inpainted_df

    model.eval()
    dt = 1.0 / steps

    for g in valid_grids:
        c_id = grid_class_lookup.get(g, 5)
        
        # 物理邊界規則：Class 1 全面歸零；Class 3 之非對角線強制歸零
        if c_id == 1 or (is_offdiag and c_id == 3):
            inpainted_df.loc[gap_dates, g] = 0.0
            continue
            
        base_series = baseline_df.loc[gap_dates, g].values.astype(np.float32)
        b_l = float(flow_df.loc[pre_day, g] - baseline_df.loc[pre_day, g]) if pre_day in flow_df.index else 0.0
        b_r = float(flow_df.loc[post_day, g] - baseline_df.loc[post_day, g]) if post_day in flow_df.index else 0.0
        
        base_tensor = torch.from_numpy(base_series).unsqueeze(0).unsqueeze(0).repeat(ensemble_size, 1, 1).to(DEVICE)
        bounds_tensor = torch.tensor([[b_l, b_r]], dtype=torch.float32, device=DEVICE).repeat(ensemble_size, 1)
        cid_tensor = torch.tensor([c_id - 1], dtype=torch.long, device=DEVICE).repeat(ensemble_size)
        
        x = torch.randn(ensemble_size, 1, GAP_LEN, device=DEVICE)
        
        if solver.lower() == "rk4":
            for i in range(steps):
                t_curr = i / steps
                t_half = t_curr + 0.5 * dt
                t_next = (i + 1) / steps
                
                t1 = torch.full((ensemble_size, 1), t_curr, dtype=torch.float32, device=DEVICE)
                k1 = model(x, t1, base_tensor, bounds_tensor, cid_tensor)
                
                t2 = torch.full((ensemble_size, 1), t_half, dtype=torch.float32, device=DEVICE)
                k2 = model(x + 0.5 * dt * k1, t2, base_tensor, bounds_tensor, cid_tensor)
                
                t3 = torch.full((ensemble_size, 1), t_half, dtype=torch.float32, device=DEVICE)
                k3 = model(x + 0.5 * dt * k2, t3, base_tensor, bounds_tensor, cid_tensor)
                
                t4 = torch.full((ensemble_size, 1), t_next, dtype=torch.float32, device=DEVICE)
                k4 = model(x + dt * k3, t4, base_tensor, bounds_tensor, cid_tensor)
                
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            for i in range(steps):
                t_val = torch.full((ensemble_size, 1), i / steps, dtype=torch.float32, device=DEVICE)
                v = model(x, t_val, base_tensor, bounds_tensor, cid_tensor)
                x = x + v * dt
            
        gen_residual = np.median(x.squeeze(1).cpu().numpy(), axis=0)
        final_flow = np.clip(base_series + gen_residual, 0.0, None)
        inpainted_df.loc[gap_dates, g] = final_flow

    return inpainted_df

print("[2/6] 執行 60 天空窗期 (2~3月) OT-FM RK4 軌跡重構推論...")
pred_diag_df = solve_ot_gap_inpainting(
    ot_model_diag, diag_df, macro_baseline_diag, valid_grids, grid_class_lookup, 
    is_offdiag=False, steps=15, solver="rk4", ensemble_size=8
)
pred_offdiag_df = solve_ot_gap_inpainting(
    ot_model_offdiag, offdiag_df, macro_baseline_offdiag, valid_grids, grid_class_lookup, 
    is_offdiag=True, steps=15, solver="rk4", ensemble_size=8
)

# 空間 k-NN 平滑與物理邊界約束
for dt in pd.date_range(GAP_START, GAP_END, freq="D"):
    d_v = pred_diag_df.loc[dt, valid_grids].values
    o_v = pred_offdiag_df.loc[dt, valid_grids].values
    
    smooth_d = 0.96 * d_v + 0.04 * spatial_knn.dot(d_v).values
    smooth_o = 0.98 * o_v + 0.02 * spatial_knn.dot(o_v).values
    
    for i, g in enumerate(valid_grids):
        c_id = grid_class_lookup.get(g, 5)
        if c_id == 1:
            smooth_d[i] = 0.0
            smooth_o[i] = 0.0
        elif c_id == 3:
            smooth_o[i] = 0.0
            
    pred_diag_df.loc[dt, valid_grids] = np.maximum(0.0, smooth_d)
    pred_offdiag_df.loc[dt, valid_grids] = np.maximum(0.0, smooth_o)

pred_total_df = pred_diag_df + pred_offdiag_df
total_truth_df = (diag_df + offdiag_df).reindex(all_sim_dates)

# =========================================================================
# 7. 官方標準 Combined NRMSE 評估
# =========================================================================
print("[3/6] 執行官方標準 Combined NRMSE 評估...")
eval_dates = [dt for dt in diag_df.index if dt >= PRED_START and not (GAP_START <= dt <= GAP_END)]
class_daily_records = {c_id: {"diag": [], "offdiag": []} for c_id in range(1, 10)}
overall_daily_records = {"diag": [], "offdiag": []}

for dt in eval_dates:
    act_od = daily_od_records.get(dt, {})
    p_diag = pred_diag_df.loc[dt]
    p_off = pred_offdiag_df.loc[dt]
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
        if c_diag_diffs[c_id]:
            class_daily_records[c_id]["diag"].append(np.sqrt(np.mean(c_diag_diffs[c_id])))
        if c_off_diffs[c_id]:
            class_daily_records[c_id]["offdiag"].append(np.sqrt(np.mean(c_off_diffs[c_id])))

    overall_daily_records["diag"].append(np.sqrt(np.mean(all_diag_diffs)) if all_diag_diffs else 0.0)
    overall_daily_records["offdiag"].append(np.sqrt(np.mean(all_off_diffs)) if all_off_diffs else 0.0)

apr_eval_dates = [dt for dt in diag_df.index if pd.to_datetime("2024-04-01") <= dt <= pd.to_datetime("2024-04-30")]
apr_rmse_per_class = {}
for c_id in range(1, 10):
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    if not c_grids or not apr_eval_dates:
        apr_rmse_per_class[c_id] = 0.0
    else:
        diffs = []
        for dt in apr_eval_dates:
            gt_mean = total_truth_df.loc[dt, c_grids].mean()
            pr_mean = pred_total_df.loc[dt, c_grids].mean()
            diffs.append((pr_mean - gt_mean) ** 2)
        apr_rmse_per_class[c_id] = np.sqrt(np.mean(diffs)) if diffs else 0.0

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
        "class name": f"{CLASS_METADATA[c_id]['name']} ({grid_cnt}格)",
        "NRMSE_diag": round(nrmse_diag_c, 3),
        "NRMSE_off": round(nrmse_offdiag_c, 3),
        "combined NRMSE": round(comb_nrmse_c, 3),
        "Apr RMSE": round(apr_rmse_per_class[c_id], 2)
    })

RMSE_diag = float(np.mean(overall_daily_records["diag"]))
RMSE_offdiag = float(np.mean(overall_daily_records["offdiag"]))
NRMSE_diag = RMSE_diag / MEAN_ACTUAL_DIAG
NRMSE_offdiag = RMSE_offdiag / MEAN_ACTUAL_OFFDIAG
combined_nrmse = WEIGHT_DIAG * NRMSE_diag + WEIGHT_OFFDIAG * NRMSE_offdiag

summary_rows.append({
    "class id": "total",
    "class name": f"({len(valid_grids)}格)",
    "NRMSE_diag": round(NRMSE_diag, 3),
    "NRMSE_off": round(NRMSE_offdiag, 3),
    "combined NRMSE": round(combined_nrmse, 3),
    "Apr RMSE": round(np.mean(list(apr_rmse_per_class.values())), 2)
})

df_table = pd.DataFrame(summary_rows)
print("\n" + "=" * 90)
print(f" 🏆 HuMob 2026 Flow Matching SOTA (OT-FM + RK4) 評估報告 (Combined NRMSE: {combined_nrmse:.4f})")
print("=" * 90)
print(f"{'class id':<10} | {'class name':<38} | {'NRMSE_diag':<10} | {'NRMSE_off':<10} | {'combined NRMSE':<14} | {'Apr RMSE':<8}")
print("-" * 90)
for _, r in df_table.iterrows():
    print(f"{r['class id']:<10} | {r['class name']:<38} | {r['NRMSE_diag']:<10.3f} | {r['NRMSE_off']:<10.3f} | {r['combined NRMSE']:<14.3f} | {r['Apr RMSE']:<8.2f}")
print("=" * 90 + "\n")

table_path = os.path.join(OUTPUT_DIR, "otfm_rk4_nrmse_summary.csv")
df_table.to_csv(table_path, index=False, encoding="utf-8-sig")

# 匯出預測 CSV 檔案
pred_diag_df.to_csv(os.path.join(OUTPUT_DIR, "pred_diag_otfm_rk4.csv"), encoding="utf-8-sig")
pred_offdiag_df.to_csv(os.path.join(OUTPUT_DIR, "pred_offdiag_otfm_rk4.csv"), encoding="utf-8-sig")
pred_total_df.to_csv(os.path.join(OUTPUT_DIR, "pred_total_otfm_rk4.csv"), encoding="utf-8-sig")

# =========================================================================
# 8. 對角線與非對角線 3×3 基準圖繪製 (完全還原樣式規格)
# =========================================================================
def render_flow_component_benchmark(
    gt_df: pd.DataFrame, 
    pred_df: pd.DataFrame, 
    flow_title: str, 
    nrmse_val: float, 
    output_filename: str
):
    """
    依照使用者指定的高對比暗黑風格渲染 3x3 九大類別波形圖：
    包含棕暗紅 Gap 覆蓋區間、粉紅 Actual Flow、青綠 Flow Matching Model，以及底部精確圖例。
    """
    gt_reindexed = gt_df.reindex(all_sim_dates)
    pred_reindexed = pred_df.reindex(all_sim_dates)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 3, figsize=(22, 12), dpi=250)
    fig.patch.set_facecolor('#070c18')

    # 主標題格式：HuMob 2026: Flow Matching (OT-FM) ({flow_title} Flow) | NRMSE: {val:.4f}
    fig.suptitle(
        f"HuMob 2026: Flow Matching (OT-FM) ({flow_title} Flow) | NRMSE: {nrmse_val:.4f}",
        fontsize=14, fontweight='bold', color='#f8fafc', y=0.985
    )

    for c_id in range(1, 10):
        row, col = (c_id - 1) // 3, (c_id - 1) % 3
        ax = axes[row, col]
        ax.set_facecolor('#0d1527')

        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
        c_meta = CLASS_METADATA[c_id]
        grid_n = len(c_grids)

        # 子圖標題格式：Class 01: Persistent Zero (N=28)
        title_str = f"Class {c_id:02d}: {c_meta['name']} (N={grid_n})"
        ax.set_title(title_str, fontsize=9.5, fontweight='bold', color='#cbd5e1', pad=6)

        if not c_grids:
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        gt_series = gt_reindexed[c_grids].mean(axis=1)
        pred_series = pred_reindexed[c_grids].mean(axis=1)

        # Gap 區間 Actual Flow 遮蔽以呈現真實資料缺失狀態
        gt_masked = gt_series.copy()
        gt_masked.loc[(gt_masked.index >= GAP_START) & (gt_masked.index <= GAP_END)] = np.nan

        # 1. 繪製 Gap 區塊 (暗褐紅/深琥珀背景)
        ax.axvspan(GAP_START, GAP_END, color='#45271d', alpha=0.52, zorder=1)

        # 2. 繪製 Flow Matching 重建軌跡 (青綠色線條)
        ax.plot(pred_series.index, pred_series, color='#2dd4bf', linewidth=1.15, alpha=0.95, zorder=2)

        # 3. 繪製 Actual Flow 真實流量 (粉紅玫瑰色線條)
        ax.plot(gt_masked.index, gt_masked, color='#f43f5e', linewidth=1.1, alpha=0.9, zorder=3)

        # 樣式微調：刻度、邊框與格線
        ax.grid(True, color='#1e293b', linestyle=':', alpha=0.45, zorder=0)
        ax.tick_params(colors='#64748b', labelsize=7.5, length=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.set_xlim(pd.to_datetime("2023-11-01"), pd.to_datetime("2024-11-01"))

        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            spine.set_linewidth(0.8)

    # 底部圖例 (Gap, Actual Flow, Flow Matching Model)
    legend_elements = [
        matplotlib.patches.Patch(facecolor='#45271d', alpha=0.7, label='Gap'),
        plt.Line2D([0], [0], color='#f43f5e', lw=1.3, label='Actual Flow'),
        plt.Line2D([0], [0], color='#2dd4bf', lw=1.4, label='Flow Matching Model')
    ]

    fig.legend(
        handles=legend_elements, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.015), 
        ncol=3, 
        fontsize=9.5,
        frameon=True, 
        facecolor='#0a1020', 
        edgecolor='#1e293b'
    )

    plt.tight_layout(rect=[0.02, 0.045, 0.98, 0.96])
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 已產出基準圖片: {out_path}")

print("[4/6] 渲染對角線 (Diagonal Flow) 基準對比圖...")
render_flow_component_benchmark(
    gt_df=diag_df,
    pred_df=pred_diag_df,
    flow_title="Diagonal",
    nrmse_val=NRMSE_diag,
    output_filename="humob_flow_matching_diagonal_benchmark.png"
)

print("[5/6] 渲染非對角線 (Off-Diagonal Flow) 基準對比圖...")
render_flow_component_benchmark(
    gt_df=offdiag_df,
    pred_df=pred_offdiag_df,
    flow_title="Off-Diagonal",
    nrmse_val=NRMSE_offdiag,
    output_filename="humob_flow_matching_offdiagonal_benchmark.png"
)

# =========================================================================
# 9. 渲染總流量 (Total Flow) 綜合波形大圖
# =========================================================================
print("[6/6] 渲染總流量 366 天綜合全域對比大圖...")
macro_total_baseline = macro_baseline_diag + macro_baseline_offdiag

fig, axes = plt.subplots(3, 3, figsize=(22, 13), dpi=250)
fig.patch.set_facecolor('#040914')

fig.suptitle(
    "HuMob 2026: 9-Class Waveform Benchmark — Flow Matching SOTA (OT-FM + RK4)\n"
    f"(Ground Truth vs Adaptive Baseline vs OT-FM (RK4) 366-Day Reconstruction | Combined NRMSE: {combined_nrmse:.4f})", 
    fontsize=14, fontweight='bold', color='#f8fafc', y=0.985
)

for c_id in range(1, 10):
    row, col = (c_id - 1) // 3, (c_id - 1) % 3
    ax = axes[row, col]
    ax.set_facecolor('#081126')
    
    c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
    c_meta = CLASS_METADATA[c_id]
    
    if not c_grids:
        ax.set_title(f"Class {c_id:02d}: {c_meta['name']}\n(No Grids Assigned)", fontsize=9, color='#64748b')
        continue
        
    gt_series = total_truth_df[c_grids].mean(axis=1)
    gt_series_masked = gt_series.copy()
    gt_series_masked.loc[(gt_series_masked.index >= GAP_START) & (gt_series_masked.index <= GAP_END)] = np.nan
    
    base_series = macro_total_baseline[c_grids].mean(axis=1)
    pred_series = pred_total_df[c_grids].mean(axis=1)
    
    ax.axvspan(GAP_START, GAP_END, color='#0f4c5c', alpha=0.35, zorder=1)
    ax.axvspan(pd.to_datetime("2024-04-01"), pd.to_datetime("2024-04-30"), color='#1e3a8a', alpha=0.15, zorder=1)
    
    ax.plot(base_series.index, base_series, color='#94a3b8', linestyle='--', linewidth=1.1, alpha=0.85, zorder=2)
    ax.plot(pred_series.index, pred_series, color='#2dd4bf', linewidth=1.4, zorder=3)
    ax.plot(gt_series_masked.index, gt_series_masked, color='#f43f5e', linewidth=1.1, alpha=0.9, zorder=4)
    
    rep_coord = c_grids[0] if len(c_grids) > 0 else "0_0"
    ax.set_title(f"[{rep_coord}] Class {c_id}: {c_meta['name']}\n{c_meta['desc']}", 
                 fontsize=10, fontweight='bold', color='#e2e8f0', pad=6)
    
    apr_val = apr_rmse_per_class.get(c_id, 0.0)
    badge_text = f"Apr RMSE: {apr_val:.2f}"
    ax.text(0.04, 0.90, badge_text, transform=ax.transAxes, fontsize=8.5, fontweight='bold',
            color='#38bdf8', bbox=dict(boxstyle="round,pad=0.35", facecolor='#0c2146', edgecolor='#0284c7', alpha=0.85, lw=1.2))
    
    ax.set_ylabel("Persons / Day", fontsize=8, color='#94a3b8')
    ax.grid(True, color='#172554', linestyle=':', alpha=0.6)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

legend_elements_total = [
    plt.Line2D([0], [0], color='#f43f5e', lw=1.3, label='Ground Truth (Observed)'),
    plt.Line2D([0], [0], color='#94a3b8', lw=1.2, linestyle='--', label='Adaptive Baseline'),
    plt.Line2D([0], [0], color='#2dd4bf', lw=1.5, label='Flow Matching SOTA (OT-FM + RK4)'),
    matplotlib.patches.Patch(facecolor='#0f4c5c', alpha=0.45, label='60-Day Blind Zone'),
    matplotlib.patches.Patch(facecolor='#1e3a8a', alpha=0.35, label='Official Eval (Apr)')
]

fig.legend(handles=legend_elements_total, loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=5, fontsize=10,
           frameon=True, facecolor='#060d1f', edgecolor='#1e293b')

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
total_plot_path = os.path.join(OUTPUT_DIR, "humob_flow_matching_total_benchmark.png")
plt.savefig(total_plot_path, dpi=250, bbox_inches='tight')
plt.close(fig)

print(f"\n✨ 全部完成！3 張對比圖表已全數匯出至資料夾：\n -> {OUTPUT_DIR}")
print("  1. humob_flow_matching_diagonal_benchmark.png    (對角線 3x3 流量圖)")
print("  2. humob_flow_matching_offdiagonal_benchmark.png (非對角線 3x3 流量圖 - 樣式對齊)")
print("  3. humob_flow_matching_total_benchmark.png       (總流量綜合對比圖)")
