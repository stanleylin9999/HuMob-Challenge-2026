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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 1. 全域配置與官方標準正規化常數 / 權重
# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIFFUSION_STEPS = 100
MC_SAMPLES = 16           # 蒙地卡羅採樣次數 (向量化平行推論加速)
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
COND_DROPOUT_PROB = 0.15  # 條件隨機丟棄率
SPARSITY_THRESH = 0.015   # 跨區轉移微小機率截斷門檻

# 官方競賽標準正規化常數與權重
MEAN_ACTUAL_DIAG = 26.57
MEAN_ACTUAL_OFFDIAG = 0.0176
WEIGHT_DIAG = 0.5
WEIGHT_OFFDIAG = 0.5

PRED_START = pd.to_datetime("2024-01-01")
EVAL_GAP_START = pd.to_datetime("2024-02-01")  # 2~3 月缺測補全區間
EVAL_GAP_END = pd.to_datetime("2024-03-31")
PRED_END = pd.to_datetime("2024-10-31")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
TSV_PATH = os.path.join(SCRIPT_DIR, "humob2026-dataset.tsv")
SPECIFIC_CLASS_DIR = r"C:\Users\User\Desktop\人口預測專案\人口預測專案3\humob2026\data\output\module05\classification\by_class"
FALLBACK_CLASS_DIR = os.path.join(SCRIPT_DIR, "humob2026", "data", "output", "module05", "classification", "by_class")
BY_CLASS_DIR = SPECIFIC_CLASS_DIR if os.path.exists(SPECIFIC_CLASS_DIR) else FALLBACK_CLASS_DIR

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "humob_robust_diffusion_output")
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
# 2. 資料解析與離群值過濾工具
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
    """使用 IQR 截斷法剔除離群值後計算中位數基準"""
    if len(df_sub) == 0:
        return pd.Series(0.0, index=df_sub.columns)
    q25 = df_sub.quantile(0.25)
    q75 = df_sub.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    clipped = df_sub.clip(lower=lower_bound, upper=upper_bound, axis=1)
    return clipped.median(axis=0).fillna(0.0)

def load_and_split_flows():
    print("[1/6] 解析 9 大類別與 TSV (分離對角線與非對角線流量)...")
    grid_class_lookup = {}
    if os.path.exists(BY_CLASS_DIR):
        for fpath in glob.glob(os.path.join(BY_CLASS_DIR, "*.csv")):
            c_id = get_class_id_from_filename(os.path.basename(fpath))
            if c_id is not None:
                try:
                    df_cls = pd.read_csv(fpath)
                    col = [c for c in df_cls.columns if any(k in str(c).lower() for k in ["grid", "orig", "id", "mesh"])][0]
                    for g in df_cls[col].dropna().astype(str).unique():
                        grid_class_lookup[g] = c_id
                except Exception:
                    pass
        print(f"✓ 成功匹配 {len(grid_class_lookup)} 個網格的類別標籤")

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
                    diag_val = float(dests.get(orig, 0.0))
                    offdiag_val = sum(float(cnt) for dest, cnt in dests.items() if dest != orig and dest != "-1_-1")
                    daily_diag_flows[dt][orig] = diag_val
                    daily_offdiag_flows[dt][orig] = offdiag_val
        except Exception:
            pass

    diag_df = pd.DataFrame.from_dict(daily_diag_flows, orient='index').fillna(0.0)
    offdiag_df = pd.DataFrame.from_dict(daily_offdiag_flows, orient='index').fillna(0.0)
    
    pre_mask = diag_df.index < PRED_START
    valid_grids = diag_df.columns[diag_df.loc[pre_mask].mean() > 0.0].tolist() if pre_mask.sum() > 0 else diag_df.columns.tolist()

    for g in valid_grids:
        if g not in grid_class_lookup:
            grid_class_lookup[g] = 5

    return diag_df[valid_grids], offdiag_df[valid_grids], daily_od_records, valid_grids, grid_class_lookup

# =========================================================================
# 3. 動態雙錨點 OD 轉移引擎 (Dynamic Bi-Anchor Transition Engine)
# =========================================================================
class DynamicODTransitionEngine:
    def __init__(self, daily_od_records, valid_grids, sparsity_thresh=SPARSITY_THRESH):
        self.valid_grids = valid_grids
        self.sparsity_thresh = sparsity_thresh
        
        # 1. 震前常態轉移矩陣 (2024-01-01 以前)
        pre_dates = [dt for dt in daily_od_records.keys() if dt < PRED_START]
        self.P_pre = self._build_prob_matrix(daily_od_records, pre_dates)
        
        # 2. 1 月底災後應急轉移矩陣 (2024-01-20 ~ 2024-01-31)
        jan_dates = [dt for dt in daily_od_records.keys() if pd.to_datetime("2024-01-20") <= dt <= pd.to_datetime("2024-01-31")]
        self.P_jan = self._build_prob_matrix(daily_od_records, jan_dates)
        
        # 3. 4 月初復原期轉移矩陣 (2024-04-01 ~ 2024-04-14)
        apr_dates = [dt for dt in daily_od_records.keys() if pd.to_datetime("2024-04-01") <= dt <= pd.to_datetime("2024-04-14")]
        if not apr_dates:
            apr_dates = [dt for dt in daily_od_records.keys() if dt > EVAL_GAP_END][:14]
        self.P_apr = self._build_prob_matrix(daily_od_records, apr_dates)
        
        # 4. 震後長期復原轉移矩陣 (2024-04-01 ~ 2024-10-31)
        post_dates = [dt for dt in daily_od_records.keys() if dt > EVAL_GAP_END]
        self.P_post = self._build_prob_matrix(daily_od_records, post_dates)

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
                raw_p = {d: c / tot for d, c in counts[orig].items()}
                filtered = {d: p for d, p in raw_p.items() if p >= self.sparsity_thresh}
                f_tot = sum(filtered.values())
                probs[orig] = {d: p / f_tot for d, p in filtered.items()} if f_tot > 0 else raw_p
            else:
                probs[orig] = {}
        return probs

    def get_dynamic_probs(self, dt):
        """依時間動態插值跨區轉移機率矩陣"""
        if dt < PRED_START:
            return self.P_pre
        elif dt < EVAL_GAP_START:
            tau = min(1.0, max(0.0, (dt - PRED_START).days / 30.0))
            return self._interpolate_probs(self.P_pre, self.P_jan, tau)
        elif dt <= EVAL_GAP_END:
            gap_days = (EVAL_GAP_END - EVAL_GAP_START).days + 1
            tau = min(1.0, max(0.0, ((dt - EVAL_GAP_START).days + 1) / gap_days))
            w = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            return self._interpolate_probs(self.P_jan, self.P_apr, w)
        else:
            tau = min(1.0, max(0.0, (dt - pd.to_datetime("2024-04-01")).days / 90.0))
            return self._interpolate_probs(self.P_apr, self.P_post, tau)

    def _interpolate_probs(self, P_start, P_end, weight):
        interp_probs = {}
        for orig in self.valid_grids:
            p_s = P_start.get(orig, {})
            p_e = P_end.get(orig, {})
            all_dests = set(p_s.keys()).union(p_e.keys())
            
            if not all_dests:
                interp_probs[orig] = self.P_pre.get(orig, {})
                continue
                
            combined = {}
            for d in all_dests:
                prob_val = (1.0 - weight) * p_s.get(d, 0.0) + weight * p_e.get(d, 0.0)
                if prob_val >= self.sparsity_thresh:
                    combined[d] = prob_val
                    
            c_tot = sum(combined.values())
            interp_probs[orig] = {d: p / c_tot for d, p in combined.items()} if c_tot > 0 else self.P_pre.get(orig, {})
        return interp_probs

# =========================================================================
# 4. 全週期 (1~10月) 類別自適應先驗外推引擎 (Robust Category Bridge & Extrapolation)
# =========================================================================
class RobustCategoryBridgePrior:
    def __init__(self, flow_df, grid_class_lookup, valid_grids):
        self.flow_df = flow_df
        self.grid_class_lookup = grid_class_lookup
        self.valid_grids = valid_grids
        
        # 1. 震前常態中位數 (Pre-EQ Baseline)
        pre_df = flow_df.loc[flow_df.index < PRED_START, valid_grids]
        self.M_pre = robust_median(pre_df).clip(lower=1.0)
        
        # 2. 1 月底災後應急左錨點 (Jan Tail)
        jan_tail = flow_df.loc["2024-01-20":"2024-01-31", valid_grids]
        self.l_left = robust_median(jan_tail).clip(lower=0.0)
        
        # 3. 4 月初復原右錨點 (Apr Head)
        if "2024-04-01" in flow_df.index:
            right_sub = flow_df.loc["2024-04-01":"2024-04-14", valid_grids]
        else:
            right_sub = flow_df.loc["2024-05-01":"2024-05-14", valid_grids] if "2024-05-01" in flow_df.index else jan_tail
        self.l_right = robust_median(right_sub).clip(lower=0.0)
        
        # 4. 震後 1 月震波衝擊參數 (自主生成 1 月，不讀取 1 月真值)
        shock_scale_map = {1: 0.0, 2: 0.35, 3: 2.2, 4: 0.25, 5: 0.30, 6: 1.15, 7: 2.8, 8: 1.8, 9: 1.5}
        self.shock_peak_levels = pd.Series(0.0, index=valid_grids)
        self.shock_init_levels = pd.Series(0.0, index=valid_grids)
        for g in valid_grids:
            c = grid_class_lookup.get(g, 5)
            base = self.M_pre[g]
            self.shock_peak_levels[g] = base * shock_scale_map.get(c, 1.0)
            if c in [1, 2, 4, 5]:
                self.shock_init_levels[g] = base * shock_scale_map.get(c, 1.0)
            elif c in [3, 7, 8]:
                self.shock_init_levels[g] = base * 1.2
            else:
                self.shock_init_levels[g] = base

    def compute_robust_seed(self, dt):
        """依據時間階段 (1月衝擊 / 2~3月缺測 / 4~10月長期外推) 自主計算先驗基線"""
        # 階段 A: 2024-01 (1 月震後衝擊與指數消退期 - 全自主預測)
        if dt < EVAL_GAP_START:
            day_idx = (dt - PRED_START).days
            tau_jan = day_idx / 30.0
            mu_t = self.shock_init_levels.copy()
            
            for g in self.valid_grids:
                c = self.grid_class_lookup.get(g, 5)
                p_val = self.shock_peak_levels[g]
                init_val = self.shock_init_levels[g]
                end_val = self.l_left[g]
                
                if c == 1:
                    mu_t[g] = 0.0
                elif c in [3, 7, 8]:
                    if day_idx <= 4:
                        # 震後前 4 天快速衝頂至避難高峰
                        mu_t[g] = init_val + (p_val - init_val) * (day_idx / 4.0)
                    else:
                        # 5~31 天指數衰減至 1 月底過渡位準
                        decay_tau = (day_idx - 4) / 26.0
                        decay_rate = 3.0 if c == 7 else 2.2
                        mu_t[g] = end_val + (p_val - end_val) * np.exp(-decay_rate * decay_tau)
                elif c in [2, 4, 5]:
                    # 震後驟降並維持在低位
                    mu_t[g] = init_val + (tau_jan ** 1.5) * (end_val - init_val)
                else:
                    mu_t[g] = init_val + tau_jan * (end_val - init_val)
                    
            return np.maximum(0.0, mu_t.values)
            
        # 階段 B: 2024-02-01 ~ 2024-03-31 (2~3 月 Gap 內插補全)
        elif dt <= EVAL_GAP_END:
            gap_days = (EVAL_GAP_END - EVAL_GAP_START).days + 1
            tau = min(1.0, max(0.0, ((dt - EVAL_GAP_START).days + 1) / gap_days))
            s_curve = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
            mu_t = self.l_left + s_curve * (self.l_right - self.l_left)
            
            for g in self.valid_grids:
                c = self.grid_class_lookup.get(g, 5)
                if c == 1:
                    mu_t[g] = 0.0
                elif c == 2:
                    mu_t[g] = self.l_left[g] + s_curve * (self.l_right[g] - self.l_left[g])
                elif c in [3, 7]:
                    dissip = 1.0 - np.exp(-3.0 * tau)
                    mu_t[g] = self.l_left[g] + dissip * (self.l_right[g] - self.l_left[g])
                elif c == 4:
                    mu_t[g] = self.l_left[g] + (tau ** 2.0) * (self.l_right[g] - self.l_left[g])
                elif c == 8:
                    dissip = 1.0 - np.exp(-2.0 * tau)
                    mu_t[g] = self.l_left[g] + dissip * (self.l_right[g] - self.l_left[g])
                elif c == 9:
                    mu_t[g] = self.l_left[g] + np.sqrt(tau) * (self.l_right[g] - self.l_left[g])
                    
            return np.maximum(0.0, mu_t.values)
            
        # 階段 C: 2024-04-01 ~ 2024-10-31 (4~10 月長期無洩漏外推預測)
        else:
            post_days = (PRED_END - EVAL_GAP_END).days
            tau_post = min(1.0, max(0.0, (dt - EVAL_GAP_END).days / post_days))
            mu_t = self.l_right.copy()
            
            for g in self.valid_grids:
                c = self.grid_class_lookup.get(g, 5)
                base = self.M_pre[g]
                r_start = self.l_right[g]
                
                if c == 1:
                    mu_t[g] = 0.0
                elif c == 2:
                    mu_t[g] = r_start
                elif c in [3, 7]:
                    mu_t[g] = r_start + (1.0 - np.exp(-1.5 * tau_post)) * (base * 0.5 - r_start)
                elif c == 4:
                    target = base * 0.75
                    mu_t[g] = r_start + np.sqrt(tau_post) * (target - r_start)
                elif c == 5:
                    s_post = 3.0 * (tau_post ** 2) - 2.0 * (tau_post ** 3)
                    mu_t[g] = r_start + s_post * (base - r_start)
                elif c == 6:
                    mu_t[g] = r_start
                elif c == 8:
                    mu_t[g] = r_start + (1.0 - np.exp(-2.0 * tau_post)) * (base - r_start)
                elif c == 9:
                    target = max(r_start, base * 1.2)
                    mu_t[g] = r_start + tau_post * (target - r_start)
                    
            return np.maximum(0.0, mu_t.values)

# =========================================================================
# 5. 雙條件擴散模型 (波型殘差學習)
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

class PreEQWaveformDataset(Dataset):
    def __init__(self, flow_df, valid_grids, max_ceiling):
        self.samples = []
        self.max_ceiling = max_ceiling
        pre_dates = [dt for dt in flow_df.index if dt < PRED_START]
        self.M_pre = robust_median(flow_df.loc[pre_dates, valid_grids]).values
        
        raw_res, raw_seeds, raw_times = [], [], []
        for dt in pre_dates:
            y_true = np.nan_to_num(flow_df.loc[dt, valid_grids].values, nan=0.0)
            seed = self.M_pre.copy()
            time_feat = extract_calendar_features(dt)
            
            raw_res.append(y_true - seed)
            raw_seeds.append(seed)
            raw_times.append(time_feat)
            
        raw_res = np.array(raw_res)
        self.scale = np.std(raw_res, axis=0)
        self.scale = np.where(self.scale < 1.0, 1.0, self.scale)
        
        for res, seed, t_feat in zip(raw_res, raw_seeds, raw_times):
            norm_res = np.nan_to_num(res / self.scale, nan=0.0)
            norm_seed = np.nan_to_num(seed / (self.max_ceiling + 1e-4), nan=0.0)
            self.samples.append((norm_res, norm_seed, t_feat))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        res, n_seed, t_feat = self.samples[idx]
        return (torch.tensor(res, dtype=torch.float32), 
                torch.tensor(n_seed, dtype=torch.float32), 
                torch.tensor(t_feat, dtype=torch.float32))

class RobustWaveformDenoiser(nn.Module):
    def __init__(self, num_nodes, time_dim=6, hidden_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        
        self.step_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.cond_norm = nn.LayerNorm(num_nodes + time_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(num_nodes + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64)
        )
        self.in_proj = nn.Linear(num_nodes, hidden_dim)
        self.res1 = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.res2 = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_proj = nn.Linear(hidden_dim, num_nodes)

    def _get_timestep_embedding(self, timesteps, dim=64):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_noisy, t_step, norm_seed, time_feat, drop_mask=None):
        t_emb = self.step_mlp(self._get_timestep_embedding(t_step, 64))
        if drop_mask is not None:
            norm_seed = norm_seed * drop_mask
            time_feat = time_feat * drop_mask
            
        c_in = self.cond_norm(torch.cat([norm_seed, time_feat], dim=-1))
        c_emb = self.cond_mlp(c_in)
        ctx = torch.cat([t_emb, c_emb], dim=-1)
        
        h = self.in_proj(x_noisy)
        h = self.res1(torch.cat([h, ctx], dim=-1)) + h
        h = self.res2(torch.cat([h, ctx], dim=-1)) + h
        return self.out_proj(h)

class DualDiffusionEngine:
    def __init__(self, timesteps=DIFFUSION_STEPS):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(DEVICE)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=DEVICE), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_var = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        if noise is None: noise = torch.randn_like(x_0)
        return self.sqrt_alphas_cumprod[t].unsqueeze(-1) * x_0 + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1) * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t, norm_seed, time_feat):
        betas_t = self.betas[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        sqrt_recip = torch.sqrt(1.0 / self.alphas[t]).unsqueeze(-1)
        
        pred_eps = model(x_t, t, norm_seed, time_feat)
        mean = sqrt_recip * (x_t - (betas_t / sqrt_one_minus) * pred_eps)
        if (t == 0).all(): return mean
        return mean + torch.sqrt(self.posterior_var[t].unsqueeze(-1)) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample_monte_carlo(self, model, norm_seed, time_feat, k_samples=MC_SAMPLES):
        b, dim = norm_seed.shape[0], model.num_nodes
        norm_seed_rep = norm_seed.repeat_interleave(k_samples, dim=0)
        time_feat_rep = time_feat.repeat_interleave(k_samples, dim=0)
        
        x_t = torch.randn(b * k_samples, dim, device=DEVICE)
        for step in reversed(range(self.timesteps)):
            t_tensor = torch.full((b * k_samples,), step, device=DEVICE, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t_tensor, norm_seed_rep, time_feat_rep)
            
        accum = x_t.view(b, k_samples, dim).mean(dim=1)
        return torch.clamp(accum, -2.5, 2.5)

# =========================================================================
# 6. 官方標準評估引擎 (從 1 月開始完整評估)
# =========================================================================
def compute_official_and_class_nrmse(eval_dates, valid_grids, grid_class_lookup, daily_od_records, 
                                     pred_diag_flows, pred_offdiag_flows, dynamic_engine, output_dir):
    class_daily_records = {c_id: {"diag": [], "offdiag": []} for c_id in range(1, 10)}
    overall_daily_records = {"diag": [], "offdiag": []}

    for dt in eval_dates:
        act_od = daily_od_records.get(dt, {})
        p_diag = pred_diag_flows[dt]
        p_off = pred_offdiag_flows[dt]
        probs_today = dynamic_engine.get_dynamic_probs(dt)

        c_diag_diffs = {c_id: [] for c_id in range(1, 10)}
        c_off_diffs = {c_id: [] for c_id in range(1, 10)}
        all_diag_diffs, all_off_diffs = [], []

        for orig in valid_grids:
            c_id = grid_class_lookup.get(orig, 5)
            act_dests = act_od.get(orig, {})

            # 1. 對角線 (i == j) 留存流量平方差
            d_err_sq = (float(p_diag[orig]) - float(act_dests.get(orig, 0.0))) ** 2
            c_diag_diffs[c_id].append(d_err_sq)
            all_diag_diffs.append(d_err_sq)

            # 2. 非對角線 (i != j) 跨區動態轉移平方差
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
    print(" 🏆 【HuMob 2026 官方標準 Combined NRMSE 全週期外推評估報告 (1月~10月)】")
    print(f" 🎯 官方常數: mean_actual_diag = {MEAN_ACTUAL_DIAG} | mean_actual_offdiag = {MEAN_ACTUAL_OFFDIAG}")
    print("=" * 110)
    print(f"{'Class Name':<32} | {'Grids':<5} | {'RMSE_diag':<10} | {'RMSE_off':<10} | {'NRMSE_diag':<11} | {'NRMSE_off':<11} | {'Combined NRMSE':<14}")
    print("-" * 110)
    for _, row in df_metrics.iterrows():
        print(f"{row['Class_Name']:<32} | {int(row['Grid_Count']):<5} | {row['RMSE_diag']:10.4f} | {row['RMSE_offdiag']:10.4f} | {row['NRMSE_diag']:11.4f} | {row['NRMSE_offdiag']:11.4f} | {row['Combined_NRMSE']:14.4f}")
    print("-" * 110)
    print(f"{'OVERALL (All Valid Grids)':<32} | {len(valid_grids):<5} | {RMSE_diag:10.4f} | {RMSE_offdiag:10.4f} | {NRMSE_diag:11.4f} | {NRMSE_offdiag:11.4f} | {combined_nrmse:14.4f}")
    print("=" * 110 + "\n")

    csv_path = os.path.join(output_dir, "class_dynamic_nrmse_breakdown.csv")
    df_metrics.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✓ 各類別官方標準 NRMSE 評估表已匯出至：{csv_path}")

    return df_metrics, combined_nrmse

# =========================================================================
# 7. 長期人數回復曲線視覺化模組
# =========================================================================
def plot_fixed_class_diffusion_comparison(total_truth_df, pred_total_df, valid_grids, grid_class_lookup, combined_nrmse, output_dir):
    plt.style.use('dark_background')
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 10), dpi=250)
    fig.patch.set_facecolor('#0b1329')
    
    fig.suptitle(f"HuMob 2026: Robust Category Diffusion (Pure Extrapolation Jan-Oct) | NRMSE: {combined_nrmse:.4f}", 
                 fontsize=13, fontweight='bold', color='#ffffff', y=0.97)

    legend_handles = []
    
    for c_id in range(1, 10):
        row, col = (c_id - 1) // 3, (c_id - 1) % 3
        ax = axes[row, col]
        ax.set_facecolor('#111c3a')
        
        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
        n_count = len(c_grids)
        
        if not c_grids:
            ax.text(0.5, 0.5, f'No Grids in Class {c_id}', ha='center', va='center', 
                    transform=ax.transAxes, color='#64748b')
            ax.set_title(f"{CLASS_INFO_MAP[c_id]} (N=0)", fontsize=9, color='#94a3b8')
            continue
            
        gt_series = total_truth_df[c_grids].mean(axis=1).copy()
        gap_mask = (gt_series.index >= EVAL_GAP_START) & (gt_series.index <= EVAL_GAP_END)
        gt_series.loc[gap_mask] = np.nan
        
        pred_series = pred_total_df[c_grids].mean(axis=1)
        
        span = ax.axvspan(EVAL_GAP_START, EVAL_GAP_END, color='#45321f', alpha=0.6, zorder=1)
        line_gt, = ax.plot(gt_series.index, gt_series, color='#f43f5e', linewidth=1.1, alpha=0.9, zorder=2)
        line_pred, = ax.plot(pred_series.index, pred_series, color='#2dd4bf', linestyle='--', linewidth=1.1, alpha=0.95, zorder=3)
        
        if c_id == 1:
            legend_handles = [span, line_gt, line_pred]
        
        ax.set_title(f"{CLASS_INFO_MAP[c_id]} (N={n_count})", fontsize=8.5, fontweight='bold', color='#e2e8f0', pad=3)
        ax.grid(True, color='#1e293b', linestyle='--', alpha=0.7)
        ax.tick_params(colors='#94a3b8', labelsize=7)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        
        min_dt = total_truth_df.index.min()
        max_dt = pred_total_df.index.max()
        ax.set_xlim([min_dt, max_dt])

    fig.legend(handles=legend_handles, 
               labels=['Feb-Mar Gap', 'Ground Truth (Raw)', 'Diffusion Extrapolation (Pred)'],
               loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, 
               fontsize=9, frameon=True, facecolor='#0b1329', edgecolor='#334155')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    
    out_path = os.path.join(output_dir, "robust_category_diffusion_9classes_fixed.png")
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 5. 9 大類別 Diffusion 外推對比圖已輸出：{out_path}")

def plot_long_term_recovery_curves(total_truth_df, pred_total_df, prior_diag, prior_offdiag, valid_grids, grid_class_lookup, output_dir):
    plt.style.use('dark_background')
    baseline_series = prior_diag.M_pre + prior_offdiag.M_pre
    color_palette = ['#94a3b8', '#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#a855f7', '#ec4899', '#3b82f6']
    
    pre_gap_mask = total_truth_df.index < EVAL_GAP_START
    post_gap_mask = total_truth_df.index > EVAL_GAP_END
    summary_rows = []

    # 1. 9 大類別獨立長週期絕對人數趨勢圖 (3x3 Subplots)
    fig, axes = plt.subplots(3, 3, figsize=(20, 12), dpi=250)
    fig.patch.set_facecolor('#0f172a')
    fig.suptitle('HuMob 2026: 9 Classes Long-Term Population Recovery Trajectories (Jan-Oct Extrapolation)', 
                 fontsize=15, fontweight='bold', color='#ffffff', y=0.98)

    for c_id in range(1, 10):
        row, col = (c_id - 1) // 3, (c_id - 1) % 3
        ax = axes[row, col]
        ax.set_facecolor('#1e293b')
        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
        
        if not c_grids:
            ax.text(0.5, 0.5, f'No Grids in Class {c_id}', ha='center', va='center', transform=ax.transAxes, color='#64748b')
            ax.set_title(CLASS_INFO_MAP[c_id], fontsize=10, color='#94a3b8')
            continue
            
        c_baseline = float(baseline_series[c_grids].mean())
        actual_pre = total_truth_df.loc[pre_gap_mask, c_grids].mean(axis=1)
        actual_post = total_truth_df.loc[post_gap_mask, c_grids].mean(axis=1)
        pred_full = pred_total_df[c_grids].mean(axis=1)
        
        pred_7d_ma = pred_full.rolling(window=7, min_periods=1, center=True).mean()
        
        ax.axvspan(EVAL_GAP_START, EVAL_GAP_END, color='#f59e0b', alpha=0.15, label='Feb-Mar Gap' if c_id == 1 else "")
        ax.axhline(c_baseline, color='#38bdf8', linestyle=':', linewidth=1.2, label=f'Pre-EQ Baseline ({c_baseline:.1f} 人)' if c_id == 1 else "")
        
        ax.plot(actual_pre.index, actual_pre, color='#f43f5e', alpha=0.5, linewidth=1.0, label='Ground Truth (Raw)' if c_id == 1 else "")
        ax.plot(actual_post.index, actual_post, color='#f43f5e', alpha=0.5, linewidth=1.0)
        ax.plot(pred_full.index, pred_full, color='#10b981', alpha=0.35, linewidth=0.8, linestyle='--', label='Model Prediction (Daily)' if c_id == 1 else "")
        ax.plot(pred_7d_ma.index, pred_7d_ma, color='#34d399', linewidth=2.0, label='7-Day MA Trend' if c_id == 1 else "")
        
        end_val = float(pred_7d_ma.iloc[-1])
        rec_pct = (end_val / (c_baseline + 1e-4)) * 100.0
        
        summary_rows.append({
            "Class_ID": c_id,
            "Class_Name": CLASS_INFO_MAP[c_id],
            "Grid_Count": len(c_grids),
            "Pre_EQ_Baseline_Flow": c_baseline,
            "Oct_End_7dMA_Flow": end_val,
            "Recovery_Ratio_Pct": rec_pct
        })
        
        ax.set_title(f"{CLASS_INFO_MAP[c_id]} (N={len(c_grids)})\nOct: {rec_pct:.1f}% ({end_val:.1f}/{c_baseline:.1f} 人)", 
                     fontsize=9, fontweight='bold', color='#f8fafc', pad=4)
        ax.grid(True, color='#334155', linestyle=':', alpha=0.5)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.set_ylabel("人數 / 網格", color='#94a3b8', fontsize=8)

    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=5, 
               fontsize=10, frameon=True, facecolor='#1e293b', edgecolor='#475569')
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    
    out_curve_path = os.path.join(output_dir, "long_term_population_recovery_9classes.png")
    plt.savefig(out_curve_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ 1. 9 大類別獨立人數回復圖譜已輸出：{out_curve_path}")

    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "long_term_recovery_summary.csv"), index=False, encoding="utf-8-sig")

    # 2. 9 大類別「平均人數」走勢多線綜合對比圖 (Mean Trajectory Overlay)
    plt.figure(figsize=(15, 8), dpi=250)
    plt.gcf().patch.set_facecolor('#0f172a')
    ax_mean = plt.gca()
    ax_mean.set_facecolor('#1e293b')

    for c_id in range(1, 10):
        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
        if not c_grids: continue
        
        pred_c_mean = pred_total_df[c_grids].mean(axis=1)
        pred_c_7d = pred_c_mean.rolling(window=7, min_periods=1, center=True).mean()
        
        c_name = CLASS_INFO_MAP[c_id].split(': ')[1]
        ax_mean.plot(pred_c_7d.index, pred_c_7d, label=f"Class {c_id:02d} ({c_name}, N={len(c_grids)})", 
                     linewidth=2.0, color=color_palette[c_id - 1])

    ax_mean.axvspan(EVAL_GAP_START, EVAL_GAP_END, color='#f59e0b', alpha=0.15, label='Feb-Mar Gap')
    ax_mean.set_title("HuMob 2026: Mean Population Flow Comparison Across 9 Classes (7-Day MA)", 
                      fontsize=14, fontweight='bold', color='#ffffff', pad=12)
    ax_mean.set_ylabel("平均人數 / 網格 (人/日)", color='#cbd5e1', fontsize=11)
    ax_mean.set_xlabel("日期", color='#cbd5e1', fontsize=11)
    ax_mean.grid(True, color='#334155', linestyle=':', alpha=0.6)
    ax_mean.tick_params(colors='#94a3b8', labelsize=9)
    ax_mean.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_mean.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    ax_mean.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=True, facecolor='#1e293b', edgecolor='#475569')
    plt.tight_layout()
    
    out_mean_path = os.path.join(output_dir, "nine_classes_mean_overlay_comparison.png")
    plt.savefig(out_mean_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"✓ 2. 9 大類別平均人數綜合對比圖已輸出：{out_mean_path}")

    # 3. 全域 (Overall All Grids) 總平均人數長期回復走勢圖
    plt.figure(figsize=(14, 7), dpi=250)
    plt.gcf().patch.set_facecolor('#0f172a')
    ax_ov = plt.gca()
    ax_ov.set_facecolor('#1e293b')

    overall_baseline = float(baseline_series[valid_grids].mean())
    ov_actual_pre = total_truth_df.loc[pre_gap_mask, valid_grids].mean(axis=1)
    ov_actual_post = total_truth_df.loc[post_gap_mask, valid_grids].mean(axis=1)
    ov_pred_full = pred_total_df[valid_grids].mean(axis=1)
    ov_pred_7d_ma = ov_pred_full.rolling(window=7, min_periods=1, center=True).mean()

    ax_ov.axvspan(EVAL_GAP_START, EVAL_GAP_END, color='#f59e0b', alpha=0.18, label='Feb-Mar Gap')
    ax_ov.axhline(overall_baseline, color='#38bdf8', linestyle='--', linewidth=1.5, label=f'Pre-EQ Mean Baseline ({overall_baseline:.2f} 人)')

    ax_ov.plot(ov_actual_pre.index, ov_actual_pre, color='#f43f5e', alpha=0.6, linewidth=1.2, label='Observed Ground Truth (Raw)')
    ax_ov.plot(ov_actual_post.index, ov_actual_post, color='#f43f5e', alpha=0.6, linewidth=1.2)
    ax_ov.plot(ov_pred_full.index, ov_pred_full, color='#10b981', alpha=0.35, linewidth=1.0, linestyle=':', label='Model Daily Output')
    ax_ov.plot(ov_pred_7d_ma.index, ov_pred_7d_ma, color='#34d399', linewidth=2.5, label='Overall 7-Day MA Recovery Trend')

    ov_end_val = float(ov_pred_7d_ma.iloc[-1])
    ov_rec_pct = (ov_end_val / (overall_baseline + 1e-4)) * 100.0

    ax_ov.set_title(f"HuMob 2026: Overall Area Population Recovery Trajectory (All {len(valid_grids)} Grids)\n"
                    f"Pre-EQ Baseline: {overall_baseline:.2f} 人 | Oct End Level: {ov_end_val:.2f} 人 (Recovery: {ov_rec_pct:.1f}%)", 
                    fontsize=13, fontweight='bold', color='#ffffff', pad=10)
    ax_ov.set_ylabel("全域平均人數 / 網格 (人)", color='#cbd5e1', fontsize=10)
    ax_ov.set_xlabel("日期", color='#cbd5e1', fontsize=10)
    ax_ov.grid(True, color='#334155', linestyle=':', alpha=0.6)
    ax_ov.tick_params(colors='#94a3b8', labelsize=9)
    ax_ov.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_ov.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    ax_ov.legend(loc='lower right', fontsize=9.5, frameon=True, facecolor='#1e293b', edgecolor='#475569')
    plt.tight_layout()
    
    out_ov_path = os.path.join(output_dir, "overall_average_population_recovery.png")
    plt.savefig(out_ov_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"✓ 3. 全域總平均人數長期回復走勢圖已輸出：{out_ov_path}")

    # 4. 全類別長期回復率演變圖 (Recovery Ratio % Evolution)
    plt.figure(figsize=(14, 7), dpi=250)
    plt.gcf().patch.set_facecolor('#0f172a')
    ax2 = plt.gca()
    ax2.set_facecolor('#1e293b')

    for c_id in range(1, 10):
        c_grids = [g for g in valid_grids if grid_class_lookup.get(g) == c_id]
        if not c_grids: continue
        c_baseline = float(baseline_series[c_grids].mean())
        if c_baseline < 0.1: continue
        
        pred_c_mean = pred_total_df[c_grids].mean(axis=1)
        pred_c_7d = pred_c_mean.rolling(window=7, min_periods=1, center=True).mean()
        ratio_series = (pred_c_7d / c_baseline) * 100.0
        
        ax2.plot(ratio_series.index, ratio_series, label=f"Class {c_id:02d}: {CLASS_INFO_MAP[c_id].split(': ')[1]}", 
                 linewidth=1.8, color=color_palette[c_id - 1])
        
    ax2.axvspan(EVAL_GAP_START, EVAL_GAP_END, color='#f59e0b', alpha=0.15, label='Gap (Feb-Mar)')
    ax2.axhline(100.0, color='#ffffff', linestyle='--', linewidth=1.2, alpha=0.7, label='100% Pre-EQ Level')
    ax2.axhline(50.0, color='#64748b', linestyle=':', linewidth=1.0, alpha=0.5)
    
    ax2.set_title("HuMob 2026: Multi-Class Population Recovery Ratio (%) Evolution", fontsize=13, fontweight='bold', color='#ffffff', pad=10)
    ax2.set_ylabel("回復率 (% of Pre-EQ Baseline)", color='#cbd5e1', fontsize=10)
    ax2.grid(True, color='#334155', linestyle=':', alpha=0.5)
    ax2.tick_params(colors='#94a3b8', labelsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=True, facecolor='#1e293b', edgecolor='#475569')
    plt.tight_layout()
    
    out_ratio_path = os.path.join(output_dir, "long_term_recovery_ratio_comparison.png")
    plt.savefig(out_ratio_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"✓ 4. 長期回復率對比圖已輸出：{out_ratio_path}")

# =========================================================================
# 8. 主程式管線
# =========================================================================
def main():
    print("=" * 85)
    print("🚀 HuMob 2026：動態雙錨點 OD 轉移引擎 ＋ 雙擴散生成 (全週期外推評估)")
    print(f"🎯 官方常數: mean_actual_diag = {MEAN_ACTUAL_DIAG} | mean_actual_offdiag = {MEAN_ACTUAL_OFFDIAG}")
    print(f"⚖️ 官方權重: Diag x {WEIGHT_DIAG} + Off-Diag x {WEIGHT_OFFDIAG}")
    print("=" * 85)

    # 1. 載入資料
    diag_df, offdiag_df, daily_od_records, valid_grids, grid_class_lookup = load_and_split_flows()
    num_nodes = len(valid_grids)
    print(f"✓ 有效網格數: {num_nodes}")

    # 2. 建立動態雙錨點 OD 轉移引擎
    print("\n[2/6] 建構動態雙錨點 OD 轉移矩陣 (Pre-EQ, Jan Tail, Apr Head, Post-EQ)...")
    dynamic_od_engine = DynamicODTransitionEngine(daily_od_records, valid_grids, sparsity_thresh=SPARSITY_THRESH)
    print("✓ 動態 OD 轉移引擎初始化完成")

    pre_mask = diag_df.index < PRED_START
    max_ceiling_diag = diag_df.loc[pre_mask, valid_grids].quantile(0.99).fillna(50.0).values * 1.5 + 5.0
    max_ceiling_offdiag = offdiag_df.loc[pre_mask, valid_grids].quantile(0.99).fillna(20.0).values * 1.5 + 5.0

    # 3. 建立去離群值先驗外推引擎
    prior_diag = RobustCategoryBridgePrior(diag_df, grid_class_lookup, valid_grids)
    prior_offdiag = RobustCategoryBridgePrior(offdiag_df, grid_class_lookup, valid_grids)

    # 4. 建立震前常態波型資料載入器
    diag_ds = PreEQWaveformDataset(diag_df, valid_grids, max_ceiling_diag)
    offdiag_ds = PreEQWaveformDataset(offdiag_df, valid_grids, max_ceiling_offdiag)
    
    diag_loader = DataLoader(diag_ds, batch_size=BATCH_SIZE, shuffle=True)
    offdiag_loader = DataLoader(offdiag_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 5. 訓練 Model 1 (對角線波型) 與 Model 2 (非對角線跨區波型)
    diff_engine = DualDiffusionEngine(DIFFUSION_STEPS)
    model_diag = RobustWaveformDenoiser(num_nodes=num_nodes).to(DEVICE)
    model_offdiag = RobustWaveformDenoiser(num_nodes=num_nodes).to(DEVICE)

    opt_diag = optim.AdamW(model_diag.parameters(), lr=LR, weight_decay=1e-4)
    opt_offdiag = optim.AdamW(model_offdiag.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("\n[3/6] 訓練對角線留存波型模型 (Diag Pre-EQ)...")
    model_diag.train()
    for ep in range(1, EPOCHS + 1):
        total_loss = 0.0
        for res, n_seed, t_feat in diag_loader:
            res, n_seed, t_feat = res.to(DEVICE), n_seed.to(DEVICE), t_feat.to(DEVICE)
            t = torch.randint(0, diff_engine.timesteps, (res.shape[0],), device=DEVICE).long()
            x_noisy, noise = diff_engine.q_sample(res, t)
            
            drop_mask = (torch.rand(res.shape[0], 1, device=DEVICE) > COND_DROPOUT_PROB).float()
            pred_noise = model_diag(x_noisy, t, n_seed, t_feat, drop_mask)
            
            loss = criterion(pred_noise, noise)
            opt_diag.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_diag.parameters(), max_norm=1.0)
            opt_diag.step()
            total_loss += loss.item()
        if ep % 30 == 0 or ep == EPOCHS:
            print(f"  Epoch [{ep:03d}/{EPOCHS}] - Diag Loss: {total_loss / len(diag_loader):.6f}")

    print("\n[4/6] 訓練非對角線跨區波型模型 (Off-Diag Pre-EQ)...")
    model_offdiag.train()
    for ep in range(1, EPOCHS + 1):
        total_loss = 0.0
        for res, n_seed, t_feat in offdiag_loader:
            res, n_seed, t_feat = res.to(DEVICE), n_seed.to(DEVICE), t_feat.to(DEVICE)
            t = torch.randint(0, diff_engine.timesteps, (res.shape[0],), device=DEVICE).long()
            x_noisy, noise = diff_engine.q_sample(res, t)
            
            drop_mask = (torch.rand(res.shape[0], 1, device=DEVICE) > COND_DROPOUT_PROB).float()
            pred_noise = model_offdiag(x_noisy, t, n_seed, t_feat, drop_mask)
            
            loss = criterion(pred_noise, noise)
            opt_offdiag.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_offdiag.parameters(), max_norm=1.0)
            opt_offdiag.step()
            total_loss += loss.item()
        if ep % 30 == 0 or ep == EPOCHS:
            print(f"  Epoch [{ep:03d}/{EPOCHS}] - Off-Diag Loss: {total_loss / len(offdiag_loader):.6f}")

    # 6. 生成 1~10 月完整預測 (從 1 月開始全程由自適應基線 + 擴散波型外推生成)
    print("\n[5/6] 結合自適應基線與擴散波型生成 1~10 月全週期預測人流...")
    model_diag.eval()
    model_offdiag.eval()
    
    all_dates = pd.date_range(PRED_START, PRED_END, freq="D")
    pred_diag_flows, pred_offdiag_flows = {}, {}

    with torch.no_grad():
        for dt in all_dates:
            # 2024 全年 (1月~10月) 一律由先驗種子 + 擴散殘差生成，不讀取真實值覆蓋
            seed_d = prior_diag.compute_robust_seed(dt)
            seed_o = prior_offdiag.compute_robust_seed(dt)
            
            n_seed_d = torch.tensor(seed_d / (max_ceiling_diag + 1e-4), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            n_seed_o = torch.tensor(seed_o / (max_ceiling_offdiag + 1e-4), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            t_feat = torch.tensor(extract_calendar_features(dt), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            res_d = diff_engine.sample_monte_carlo(model_diag, n_seed_d, t_feat).squeeze(0).cpu().numpy() * diag_ds.scale
            level_ratio_d = np.clip(seed_d / (diag_ds.M_pre + 1e-4), 0.1, 1.2)
            res_d = res_d * level_ratio_d

            res_o = diff_engine.sample_monte_carlo(model_offdiag, n_seed_o, t_feat).squeeze(0).cpu().numpy() * offdiag_ds.scale
            level_ratio_o = np.clip(seed_o / (offdiag_ds.M_pre + 1e-4), 0.1, 1.2)
            res_o = res_o * level_ratio_o
            
            final_d = np.clip(seed_d + res_d, 0.0, max_ceiling_diag)
            final_o = np.clip(seed_o + res_o, 0.0, max_ceiling_offdiag)
            
            for i, g in enumerate(valid_grids):
                if grid_class_lookup.get(g, 0) == 1:
                    final_d[i] = 0.0
                    final_o[i] = 0.0

            pred_diag_flows[dt] = pd.Series(final_d, index=valid_grids)
            pred_offdiag_flows[dt] = pd.Series(final_o, index=valid_grids)

    # 7. 官方標準評估 (評估 1 月及 4~10 月所有真實觀測日)
    eval_dates = [dt for dt in diag_df.index if dt >= PRED_START and not (EVAL_GAP_START <= dt <= EVAL_GAP_END)]
    df_class_metrics, combined_nrmse = compute_official_and_class_nrmse(
        eval_dates=eval_dates,
        valid_grids=valid_grids,
        grid_class_lookup=grid_class_lookup,
        daily_od_records=daily_od_records,
        pred_diag_flows=pred_diag_flows,
        pred_offdiag_flows=pred_offdiag_flows,
        dynamic_engine=dynamic_od_engine,
        output_dir=OUTPUT_DIR
    )

    # 8. 匯出流量 CSV 與產生所有圖譜
    print("\n[6/6] 匯出預測 CSV 與產生長期人數回復圖譜 (含各類別平均圖與 Diffusion 對比圖)...")
    pred_diag_df = pd.DataFrame.from_dict(pred_diag_flows, orient='index')
    pred_offdiag_df = pd.DataFrame.from_dict(pred_offdiag_flows, orient='index')
    pred_total_df = pred_diag_df + pred_offdiag_df
    pred_total_df.to_csv(os.path.join(OUTPUT_DIR, "pred_total_flows.csv"), encoding="utf-8-sig")

    total_truth_df = diag_df + offdiag_df
    
    # 產出既有的 4 張分析圖
    plot_long_term_recovery_curves(
        total_truth_df=total_truth_df,
        pred_total_df=pred_total_df,
        prior_diag=prior_diag,
        prior_offdiag=prior_offdiag,
        valid_grids=valid_grids,
        grid_class_lookup=grid_class_lookup,
        output_dir=OUTPUT_DIR
    )

    # 產出 9 大類別 Diffusion 修正對比圖
    plot_fixed_class_diffusion_comparison(
        total_truth_df=total_truth_df,
        pred_total_df=pred_total_df,
        valid_grids=valid_grids,
        grid_class_lookup=grid_class_lookup,
        combined_nrmse=combined_nrmse,
        output_dir=OUTPUT_DIR
    )
    
    print(f"✨ 全部管線執行完畢！所有報表與圖表已完整儲存至：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
