# -*- coding: utf-8 -*-
"""
Online Inference (tam revize, raporlama + senkron ilerleme):
- Model yükleme ve path doğrulama
- Senaryo başında hedef açık oran bandı (üst sınır dahil) + kantil fallback
- Undirected edge sayımı/raporlama
- Çevrimiçi depremlerde yumuşak kapama: hedef açık oran ↑, kapama bütçesi ↓
- SAFE set (SP yolları + araç çevresi halo) + bağlanırlık garantisi
- Her deprem sonrası: yeni kapanan yol adedi ve listesi (i<j) + açık oran
- Her segmentte: gidilen kenarları (u->v) yaz
- Araçlar eşzamanlı (senkron) adım adım ilerler
"""

import time
import heapq
import random
import os

import numpy as np
import tensorflow as tf
import pandas as pd

from keras.layers import Dense, Input, Lambda
from keras.models import Model

# ============================
# Kullanıcı parametreleri
# ============================
MODEL_PATH = r"cit_q_network_kcekmece.h5"  # Eğitilmiş ağ .h5 (var olmalı)

DATA_DISTANCE_XLSX = "kcekmece_distance.xlsx"
DATA_RISK_XLSX     = "kcekmece_risk_matris.xlsx"
SHEET_NAME         = "Sheet1"


MAX_ACTIONS = 8
N_SCENARIOS = 1

# --- DEPOT ---
RANDOM_DEPOT_ACTIVE   = False
DEPOT_DEGREE_MIN      = 3
DEPOT_CANDIDATES      = None
DEPOT_WEIGHTED_BY_DEG = True
FIXED_DEPOT_IDX       = 663

# --- DEMAND ---
FIXED_DEMAND_NODES       = [316, 464, 607, 748]
DEMAND_QTY_PER_NODE_TEST = 1

CURRICULUM_TEST_ACTIVE   = True
CURRICULUM_DEMANDS_RANGE = (2, 6)
PRIORITY_NODES           = [316, 464, 607, 748]
PRIORITY_MIN             = 2

# --- Araçlar ---
VEHICLE_COUNT      = 1
VEHICLE_CAPACITIES = None  # None → toplam talebe göre bölüştür

# --- Senaryo başı risk (deprem-1) ---
RISK_TARGET_OPEN_FRAC   = (0.50, 0.95)   # hedef açık oran bandı (min,max)#(0.80, 0.95) 
RISK_TRIES_PRIMARY      = 8
RISK_TRIES_FALLBACK     = 8
# Not: r küçülürse daha ÇOK kapama, r büyürse daha AZ kapama
RISK_R_PRIMARY_RANGE    = (0.20, 0.28)#(0.42, 0.48)
RISK_R_FALLBACK_RANGE   = (0.30, 0.60)
RISK_R_FIXED            = None

FORCE_SYMMETRY          = True
REOPEN_ON_DEMAND        = True
REOPEN_LIMIT_PER_TARGET = 30

PROTECT_CRITICAL        = True
PROTECT_CRITICAL_MODE   = 'min_degree'   # 'all' | 'min_degree'
MIN_DEGREE_PER_CRIT     = 1

# --- Shield/test davranışları ---
SHIELD_ACTIVE         = True
SHIELD_TEST_FRAC      = 1.0
SHIELD_KICKIN_STEPS   = 20
NP_STREAK_THRESHOLD   = 15
TABU_RECENT           = 16
LOOP_WINDOW           = 32
LOOP_UNIQUE_MAX       = 7
PROGRESS_TOL          = 1e-6

# Ödül parametreleri (eğitimle uyumlu)
STALL_LIMIT             = 400
STALL_PENALTY           = -20.0
BACKTRACK_PENALTY       = -0.30
NO_PROGRESS_PENALTY     = -0.40
RECENT_REVISIT_PENALTY  = -0.20
RECENT_WINDOW           = 8
C_STEP                  = 0.02
LAMBDA_PBRS             = 2.5
ALPHA_NEAR              = 1.2
ALPHA_HOME              = 2.5
UNREACHABLE_NEAR_MULT   = 50.0
ALPHA_HOME_ESCAPE       = 6.0
ILLEGAL_PENALTY         = -1.0
DELIVERY_BONUS          =  1.2
FINAL_BONUS             =  7.5

MAX_TIMESTEPS_TEST      = 1000  # senkron döngü için geniş

# ============================================================
# Çevrimiçi (episode içi) deprem/kapanma
# ============================================================
ONLINE_RISK_UPDATES_ACTIVE     = True
ONLINE_RISK_AT_GLOBAL_STEPS    = [50, 120]  # örnek tetik adımları
ONLINE_RISK_EVERY_K_STEPS      = None
ONLINE_RISK_MAX_EVENTS         = 3

# Olay 1→2→3 giderek daha YUMUŞAK (hedef açık oran ↑) + daha küçük bütçe
ONLINE_EVENT_TARGET_OPEN_FRACS = [0.88, 0.92, 0.96]  # 1. olay ≈%88 açık, 3. ≈%96
ONLINE_EVENT_BUDGET            = [60, 35, 20]        # undirected kenar bütçeleri

ONLINE_FORCE_SYMMETRY          = True
ONLINE_PROTECT_MODE            = PROTECT_CRITICAL_MODE
ONLINE_MIN_DEG_PER_CRIT        = MIN_DEGREE_PER_CRIT
ONLINE_REOPEN_ON_DEMAND        = True
ONLINE_REOPEN_LIMIT_PER_TARGET = 30
ONLINE_ENSURE_VEHICLE_TO_DEPOT = True
ONLINE_SAFE_HOPS               = 1                   # araç çevresi halo (hops)
ONLINE_PRESERVE_SP_PAIRS       = True                # (depo->talep) + (araç->depo) SP yolları korunur

# Olay sonrası açık oran üst/alt bant + retry
ONLINE_MAX_OPEN_RATIO             = 0.98
ONLINE_MIN_OPEN_RATIO_AFTER_EVENT = 0.75
ONLINE_RETRY_MAX                  = 3

# Raporlamada kapanan kenar listesini sınırlamak için
PRINT_CLOSED_EDGES_LIMIT       = 200

# Rastgelelik
SEED = None

# ============================
# Yardımcılar
# ============================
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

def load_square_matrix_xlsx(path, sheet_name='Sheet1'):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    while df.shape[0] > 0 and df.iloc[0].isna().all():
        df = df.iloc[1:, :].reset_index(drop=True)
    while df.shape[1] > 0 and df.iloc[:, 0].isna().all():
        df = df.iloc[:, 1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    assert df.shape[0] == df.shape[1], f"Matrix kare olmalı, mevcut {df.shape}"
    arr = df.values.astype(np.float32)
    np.fill_diagonal(arr, 0.0)
    return arr

def action_matrix(dst, max_actions):
    action_list = []
    padded_action_list = []
    n = len(dst)
    for i, row in enumerate(dst):
        row = np.asarray(row, dtype=np.float32)
        valid = (row > 0)
        if 0 <= i < n:
            valid[i] = False
        action_id = np.where(valid)[0]
        action_list.append(action_id)
        padded = np.pad(action_id, (0, max(0, max_actions - len(action_id))),
                        constant_values=-1)[:max_actions]
        padded_action_list.extend(padded)
    return action_list, np.array(padded_action_list, dtype=np.int32)

def build_graph(dst):
    G = {i: [] for i in range(len(dst))}
    for i, row in enumerate(dst):
        for j, w in enumerate(row):
            if w > 0:
                G[i].append((j, float(w)))
    return G

def dijkstra_all_sources(dst):
    G = build_graph(dst)
    n = len(dst)
    INF = np.inf
    sp = np.full((n, n), INF, dtype=np.float64)
    for src in range(n):
        dist = np.full(n, INF, dtype=np.float64)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d,u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v,w in G[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        sp[src,:] = dist
    return sp

def dijkstra_path_with_parents(G, src, tgt):
    n = len(G)
    INF = np.inf
    dist = [INF] * n
    parent = [-1] * n
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d,u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == tgt:
            break
        for v,w in G[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[tgt]):
        return float('inf'), []
    path = []
    v = tgt
    while v != -1:
        path.append(v)
        v = parent[v]
    path.reverse()
    return float(dist[tgt]), path

def degree_info(dst, node):
    arr = np.array(dst)
    out_deg = int(np.sum(arr[node] > 0))
    in_deg  = int(np.sum(arr[:, node] > 0))
    return out_deg, in_deg

def check_connectivity_and_routes(dst, src, demand_nodes):
    G = build_graph(dst)
    unreachable = []
    print("=== CONNECTIVITY CHECK ===")
    for d in demand_nodes:
        dist, path = dijkstra_path_with_parents(G, src, d)
        if not np.isfinite(dist):
            out_d, in_d = degree_info(dst, d)
            print(f"- Target {d}: UNREACHABLE ✗ | out_deg={out_d} in_deg={in_d}")
            unreachable.append(d)
        else:
            print(f"- Target {d}: reachable ✓  distance={dist:.2f} | path: {path}")
    if unreachable:
        print(f"Unreachable targets: {unreachable}")
    print("=== END CONNECTIVITY CHECK ===")
    return unreachable

# ----- Undirected edge sayımları / rapor yardımcıları -----
def count_closed_undirected(dm_before, dm_after, symmetric=True):
    mb = (np.array(dm_before) > 0)
    ma = (np.array(dm_after)  > 0)
    eff_closed = mb & (~ma)
    if symmetric:
        return int(np.sum(np.triu(np.logical_or(eff_closed, eff_closed.T), k=1)))
    else:
        return int(np.sum(eff_closed))

def count_open_undirected(dm, symmetric=True):
    m = (np.array(dm) > 0)
    if symmetric:
        return int(np.sum(np.triu(np.logical_or(m, m.T), k=1)))
    else:
        return int(np.sum(m))

def open_ratio_undirected(dm_initial, dm_current, symmetric=True):
    total = count_open_undirected(dm_initial, symmetric=symmetric)
    now   = count_open_undirected(dm_current,  symmetric=symmetric)
    return (now / max(1, total))

def closed_edges_undirected(dm_before, dm_after, symmetric=True):
    mb = (np.array(dm_before) > 0)
    ma = (np.array(dm_after)  > 0)
    eff_closed = mb & (~ma)
    if symmetric:
        eff_closed = np.triu(np.logical_or(eff_closed, eff_closed.T), k=1)
    idx = np.argwhere(eff_closed)
    return [(int(i), int(j)) for (i, j) in idx]  # undirected (i<j)

# ----- Risk uygulama / bağlanırlık -----
def apply_risk_to_distance_safe(dm, rm, r, critical_nodes=None, symmetric=True,
                                protect_mode='all', min_degree_per_critical=1):
    dm = np.array(dm, dtype=np.float32)
    rm = np.nan_to_num(np.array(rm, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    cond = (dm > 0) & (rm > float(r))
    if symmetric:
        cond = np.logical_or(cond, cond.T)
    new_dm = np.where(cond, 0.0, dm)
    if critical_nodes:
        crit = sorted(set(int(c) for c in critical_nodes if c is not None))
        if protect_mode == 'all':
            new_dm[crit, :] = dm[crit, :]
            new_dm[:, crit] = dm[:, crit]
        elif protect_mode == 'min_degree':
            for c in crit:
                open_idx = np.where(new_dm[c, :] > 0)[0]
                if open_idx.size < min_degree_per_critical:
                    cand = np.where(dm[c, :] > 0)[0]
                    need = min_degree_per_critical - open_idx.size
                    pick = cand[np.argsort(dm[c, cand])[:need]]
                    new_dm[c, pick] = dm[c, pick]
                    new_dm[pick, c] = dm[pick, c]
    closed_count = count_closed_undirected(dm, new_dm, symmetric=symmetric)
    return new_dm, closed_count

def ensure_connectivity(dst_current, dst_initial, src, demand_nodes, reopen_limit_per_target=10):
    fixed = np.array(dst_current, dtype=np.float32)
    G_init = build_graph(dst_initial)
    for tgt in list(demand_nodes):
        dist_cur, _ = dijkstra_path_with_parents(build_graph(fixed), src, tgt)
        if np.isfinite(dist_cur):
            continue
        dist0, path0 = dijkstra_path_with_parents(G_init, src, tgt)
        if not np.isfinite(dist0) or len(path0) < 2:
            continue
        reopened = 0
        for u, v in zip(path0[:-1], path0[1:]):
            if fixed[u, v] <= 0.0 and dst_initial[u, v] > 0.0:
                fixed[u, v] = dst_initial[u, v]
                fixed[v, u] = dst_initial[v, u]
                reopened += 1
                if reopened >= reopen_limit_per_target:
                    break
    return fixed

def ensure_connectivity_pairs(dst_current, dst_initial, pairs, reopen_limit_per_pair=10):
    fixed = np.array(dst_current, dtype=np.float32)
    G_init = build_graph(dst_initial)
    for (src, tgt) in list(pairs):
        dist_cur, _ = dijkstra_path_with_parents(build_graph(fixed), src, tgt)
        if np.isfinite(dist_cur):
            continue
        dist0, path0 = dijkstra_path_with_parents(G_init, src, tgt)
        if not np.isfinite(dist0) or len(path0) < 2:
            continue
        reopened = 0
        for u, v in zip(path0[:-1], path0[1:]):
            if fixed[u, v] <= 0.0 and dst_initial[u, v] > 0.0:
                fixed[u, v] = dst_initial[u, v]
                fixed[v, u] = dst_initial[v, u]
                reopened += 1
                if reopened >= reopen_limit_per_pair:
                    break
    return fixed

# ----- Undirected edge iterator -----
def undirected_edge_iter(mask_bool):
    idx = np.argwhere(np.triu(mask_bool, k=1))
    for i, j in idx:
        yield int(i), int(j)

# ----- SAFE & bütçeli kapama -----
def bfs_halo_nodes(dm, sources, hops=1):
    n = len(dm)
    adj = [np.where(dm[i] > 0)[0].tolist() for i in range(n)]
    from collections import deque
    vis = [-1]*n
    q = deque()
    for s in sources:
        if 0 <= s < n:
            vis[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        if vis[u] >= hops:
            continue
        for v in adj[u]:
            if vis[v] == -1:
                vis[v] = vis[u] + 1
                q.append(v)
    return {i for i,d in enumerate(vis) if d != -1}

def sp_edge_set_undirected(dm, pairs):
    G = build_graph(dm)
    edges = set()
    for (s,t) in pairs:
        dist, path = dijkstra_path_with_parents(G, int(s), int(t))
        if np.isfinite(dist) and len(path) >= 2:
            for u, v in zip(path[:-1], path[1:]):
                a, b = (u, v) if u < v else (v, u)
                edges.add((a, b))
    return edges

def selective_close_with_budget(dm, rm, r, budget=None, safe_edges=None, symmetric=True):
    dm = np.array(dm, dtype=np.float32)
    rm = np.array(rm, dtype=np.float32)
    cand_dir = (dm > 0) & (rm > float(r))
    if symmetric:
        cand_dir = np.logical_or(cand_dir, cand_dir.T)
    cand_undir = np.triu(cand_dir, k=1)
    if safe_edges:
        safe_mask = np.zeros_like(cand_undir, dtype=bool)
        for (i,j) in safe_edges:
            safe_mask[i, j] = True
        cand_undir = np.logical_and(cand_undir, ~safe_mask)
    pairs = list(undirected_edge_iter(cand_undir))
    if not pairs:
        return dm, 0
    scores = [max(rm[i, j], rm[j, i]) for (i, j) in pairs]
    order = np.argsort(-np.array(scores, dtype=np.float32))
    pairs_sorted = [pairs[k] for k in order]
    if (budget is not None) and (budget >= 0):
        pairs_sorted = pairs_sorted[:int(budget)]
    new_dm = dm.copy()
    for (i, j) in pairs_sorted:
        if new_dm[i, j] > 0:
            new_dm[i, j] = 0.0
        if symmetric and new_dm[j, i] > 0:
            new_dm[j, i] = 0.0
    closed_count = count_closed_undirected(dm, new_dm, symmetric=symmetric)
    return new_dm, closed_count

# ----- Eksik yardımcılar (depo seçimi, kapasite bölüştürme, talep seçimi) -----
def pick_random_depot(dst_mat, degree_min=1, candidates=None, weighted=True):
    arr = np.array(dst_mat, dtype=np.float32)
    outdeg = np.sum(arr > 0, axis=1)
    nodes = [i for i, d in enumerate(outdeg) if d >= degree_min]
    if candidates is not None:
        cand = set(int(x) for x in candidates)
        nodes = [n for n in nodes if n in cand]
    if not nodes:
        return 0
    if weighted:
        w = outdeg[nodes].astype(np.float64)
        w = w / np.sum(w)
        return int(np.random.choice(nodes, p=w))
    else:
        return int(random.choice(nodes))

def split_sum_positive(total, parts):
    if parts <= 0:
        return []
    base = [total // parts] * parts
    for i in range(total % parts):
        base[i] += 1
    return base

def choose_demands_with_priority(n_loc, depot, k, priority_nodes, priority_min, qty_per_node):
    pr = [p for p in priority_nodes if 0 <= p < n_loc and p != depot]
    rest = [i for i in range(n_loc) if i != depot and i not in pr]
    k_pr = max(0, min(k, min(priority_min, len(pr))))
    chosen_pr = random.sample(pr, k_pr) if k_pr > 0 else []
    remaining = max(0, k - len(chosen_pr))
    chosen_rest = random.sample(rest, remaining) if remaining > 0 else []
    chosen = chosen_pr + chosen_rest
    return [[int(n), int(qty_per_node)] for n in chosen]

# ============================
# Dueling-DQN mimarisi + model yükleme
# ============================
def dueling_combine(va):
    V, A = va
    return V + (A - tf.reduce_mean(A, axis=1, keepdims=True))

def build_dueling_dqn(state_size, num_actions):
    inputs = Input(shape=(state_size,))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(1024, activation='relu')(x)
    V = Dense(1, activation='linear', name='V')(x)
    A = Dense(num_actions, activation='linear', name='A')(x)
    Q = Lambda(dueling_combine, name="dueling_combine")([V, A])
    return Model(inputs=inputs, outputs=Q)

def load_trained_q_network(model_path, state_size, num_actions):
    model = build_dueling_dqn(state_size, num_actions)
    assert os.path.exists(model_path), f"Model dosyası bulunamadı: {model_path}"
    try:
        model.load_weights(model_path)
        print(f"[MODEL] Weights loaded via load_weights() from: {model_path}")
        return model
    except Exception as e1:
        print(f"[MODEL] load_weights failed: {e1}. Trying load_model() fallback...")
        try:
            full = tf.keras.models.load_model(
                model_path,
                custom_objects={"dueling_combine": dueling_combine},
                compile=False,
                safe_mode=False
            )
            model.set_weights(full.get_weights())
            print(f"[MODEL] Loaded via load_model() and transferred weights.")
            return model
        except Exception as e2:
            raise RuntimeError(
                f"Could not load model from {model_path}. "
                f"First error: {e1} | Fallback error: {e2}"
            )

def r_for_target_open_ratio(dst0, risk, target_open_ratio, symmetric=True):
    m0 = (np.array(dst0) > 0)
    if symmetric:
        sel = np.triu(np.logical_or(m0, m0.T), k=1)
    else:
        sel = m0
    vals = np.array(risk, dtype=np.float32)[sel]
    if vals.size == 0:
        return 1.0
    target = float(np.clip(target_open_ratio, 0.0, 1.0))
    return float(np.quantile(vals, target))

# ============================
# Veri yükle + model hazırla
# ============================
dst  = load_square_matrix_xlsx(DATA_DISTANCE_XLSX, sheet_name=SHEET_NAME)
risk = load_square_matrix_xlsx(DATA_RISK_XLSX,     sheet_name=SHEET_NAME)
assert dst.shape == risk.shape, "Mesafe ve risk matrisleri aynı boyutta olmalı!"
n_locations = len(dst); N_LOC = n_locations

_arr   = np.array(dst, dtype=np.float32)
_pos   = _arr[_arr > 0]
D_SCALE = float(np.percentile(_pos, 95)) if _pos.size > 0 else 1.0

state_size = 2 * n_locations + 1
num_actions = MAX_ACTIONS
q_network = load_trained_q_network(MODEL_PATH, state_size, num_actions)
dummy_out  = q_network(np.zeros((1, state_size), dtype=np.float32))
NUM_ACTIONS = int(dummy_out.shape[-1])
assert NUM_ACTIONS == MAX_ACTIONS, f"Model çıkış sayısı ({NUM_ACTIONS}) MAX_ACTIONS ({MAX_ACTIONS}) ile uyumlu olmalı!"
print(f"[MODEL] Forward OK. Output dim={NUM_ACTIONS}")

# ============================
# Ortam sınıfları
# ============================
init_location = 0  # global

class Environment:
    def __init__(self, dst, demands, n_vehicles, vehicle_capacities, n_locations, sp_dist):
        self.dst = dst
        self.n_locations = n_locations
        self.demands_init = [(int(node), int(amount)) for node, amount in demands]
        self.demands = {node: amount for node, amount in self.demands_init}
        self.n_vehicles = n_vehicles
        self.vehicle_capacities = vehicle_capacities

        self.action_list, self.padded_action_list = action_matrix(dst, MAX_ACTIONS)

        self.location = init_location
        self.one_hot_location = self.one_hot_encode(init_location, self.n_locations)
        self.one_hot_demands = self.one_hot_encode_demands()

        self.d_scale = D_SCALE
        self.initial_total_demand = sum(amount for _, amount in demands)
        self.gamma_rl = 0.995

        self.steps_since_delivery = 0
        self.stall_limit = STALL_LIMIT
        self.stall_penalty = STALL_PENALTY

        self.sp_dist = sp_dist

        self.last_term_reason = None
        self.no_progress_streak = 0
        self.vehicles_ref = None

    def one_hot_encode(self, index, size):
        v = np.zeros(size); v[index] = 1; return v

    def one_hot_encode_demands(self):
        v = np.zeros(self.n_locations)
        for node in self.demands:
            v[node] = 1
        return v

    def get_state_from_external(self, vehicle):
        vehicle_capacity = vehicle.capacity / max(1, vehicle.initial_capacity)
        self.one_hot_location = self.one_hot_encode(vehicle.current_location, self.n_locations)
        self.one_hot_demands  = self.one_hot_encode_demands()
        return np.concatenate((self.one_hot_location, [vehicle_capacity], self.one_hot_demands))

    def attach_vehicles(self, vehicles): self.vehicles_ref = vehicles

    def all_vehicles_at_depot(self):
        if not self.vehicles_ref: return False
        return all(v.current_location == init_location for v in self.vehicles_ref)

    def _can_reach_any_demand(self, node: int) -> bool:
        if len(self.demands) == 0: return True
        dem_idx = list(self.demands.keys())
        row = self.sp_dist[node, dem_idx]
        return np.isfinite(row).any()

    def _nearest_demand_dist(self, from_node: int) -> float:
        if len(self.demands) == 0: return 0.0
        idxs = list(self.demands.keys())
        d = self.sp_dist[from_node, idxs]
        m = float(np.min(d)) if len(d) else np.inf
        if not np.isfinite(m): return self.d_scale * UNREACHABLE_NEAR_MULT
        return m

    def _potential(self, vehicle) -> float:
        delivered = self.initial_total_demand - sum(self.demands.values())
        delivered_frac = delivered / max(1, self.initial_total_demand)
        near = self._nearest_demand_dist(vehicle.current_location) / self.d_scale
        d0 = self.sp_dist[vehicle.current_location, init_location]
        to_depot = UNREACHABLE_NEAR_MULT if ((not np.isfinite(d0)) or (d0 >= 1e17)) else (d0 / self.d_scale)
        phi = (1.0 * delivered_frac) - (ALPHA_NEAR * near)
        if len(self.demands) == 0:
            phi -= (ALPHA_HOME * to_depot)
        else:
            if not self._can_reach_any_demand(vehicle.current_location):
                phi -= (ALPHA_HOME_ESCAPE * to_depot)
        return float(phi)

    def next_hop_to_nearest_demand(self, src: int):
        if len(self.demands) == 0: return None
        dem_idx = list(self.demands.keys())
        drow = self.sp_dist[src, dem_idx]
        if not np.isfinite(drow).any(): return None
        tgt = dem_idx[int(np.argmin(drow))]
        G = build_graph(self.dst)
        dist, path = dijkstra_path_with_parents(G, src, tgt)
        if not np.isfinite(dist) or len(path) < 2: return None
        return int(path[1])

    def next_hop_to_home(self, src: int):
        G = build_graph(self.dst)
        dist, path = dijkstra_path_with_parents(G, src, init_location)
        if not np.isfinite(dist) or len(path) < 2: return None
        return int(path[1])

    def is_looping(self, visited: list) -> bool:
        if len(visited) < LOOP_WINDOW: return False
        uniq = len(set(visited[-LOOP_WINDOW:]))
        return uniq <= LOOP_UNIQUE_MAX

    def topk_action_indices(self, node: int, force_first: int = None,
                            tabu_nodes: set = None, return_neighbors: bool = False):
        neigh = self.action_list[node]
        n_all = len(neigh)
        maxlen = min(n_all, NUM_ACTIONS)
        if maxlen == 0:
            return ([] if not return_neighbors else ([], []))
        slots = np.arange(maxlen, dtype=np.int32)
        neighbor_nodes = neigh[slots].astype(np.int32)
        if len(self.demands) > 0:
            dem_idx = np.array(list(self.demands.keys()), dtype=np.int32)
            d = self.sp_dist[neighbor_nodes[:, None], dem_idx]
            best = np.min(d, axis=1)
            cur_row = self.sp_dist[node, dem_idx]
            cur_reach = np.isfinite(cur_row).any()
            reach_mask = np.isfinite(best)
            if not reach_mask.any():
                best_home = self.sp_dist[neighbor_nodes, init_location]
                order = np.argsort(best_home)
            else:
                if not cur_reach:
                    keep = reach_mask
                    slots = slots[keep]; neighbor_nodes = neighbor_nodes[keep]; best = best[keep]
                cur_near = self._nearest_demand_dist(node)
                progress = cur_near - best
                step_costs = np.array([self.dst[node][int(j)] for j in neighbor_nodes], dtype=np.float32)
                score = (progress / self.d_scale) - C_STEP * (step_costs / self.d_scale)
                order = np.argsort(-score)
        else:
            best = self.sp_dist[neighbor_nodes, init_location]
            step_costs = np.array([self.dst[node][int(j)] for j in neighbor_nodes], dtype=np.float32)
            score = ((self.sp_dist[node, init_location] - best) / self.d_scale) - C_STEP * (step_costs / self.d_scale)
            order = np.argsort(-score)
        slots = slots[order]; neighbor_nodes = neighbor_nodes[order]
        if tabu_nodes:
            mask = np.array([n not in tabu_nodes for n in neighbor_nodes], dtype=bool)
            if force_first is not None: mask = np.logical_or(mask, neighbor_nodes == force_first)
            slots = slots[mask]; neighbor_nodes = neighbor_nodes[mask]
            if len(slots) == 0:
                slots = np.arange(maxlen, dtype=np.int32)[order]
                neighbor_nodes = neigh[slots].astype(np.int32)
        if force_first is not None and len(slots) > 0:
            where = np.where(neighbor_nodes == force_first)[0]
            if len(where) > 0 and where[0] != 0:
                k = int(where[0])
                slots = np.concatenate([[slots[k]], slots[np.arange(len(slots)) != k]])
                neighbor_nodes = np.concatenate([[neighbor_nodes[k]], neighbor_nodes[np.arange(len(neighbor_nodes)) != k]])
        if return_neighbors:
            return list(slots[:NUM_ACTIONS]), list(neighbor_nodes[:NUM_ACTIONS])
        return list(slots[:NUM_ACTIONS])

    def step(self, vehicle, action, action_index):
        current_state = self.get_state_from_external(vehicle)
        n_valid = min(len(self.action_list[vehicle.current_location]), NUM_ACTIONS)
        invalid = (action_index is None or action_index >= n_valid or action is None or
                   action < 0 or action >= self.n_locations or
                   self.dst[vehicle.current_location][action] <= 0 or
                   action == vehicle.current_location)
        prev_node = vehicle.visited_locations[-1] if len(vehicle.visited_locations) >= 1 else None
        phi_prev = self._potential(vehicle)
        self.last_term_reason = None
        if invalid:
            next_state = current_state
            reward = ILLEGAL_PENALTY
            return next_state, reward, False, False
        prev_prev = vehicle.visited_locations[-2] if len(vehicle.visited_locations) >= 2 else None
        will_backtrack = (prev_prev is not None and action == prev_prev)
        recent_revisit = (action in self.visited_slice(vehicle.visited_locations))
        near_prev = self._nearest_demand_dist(vehicle.current_location)
        distance_traveled = float(self.dst[vehicle.current_location][action])
        vehicle.total_distance += distance_traveled
        vehicle.visited_locations.append(action)
        vehicle.current_location = action
        reward = - distance_traveled / self.d_scale
        if will_backtrack:   reward += BACKTRACK_PENALTY
        if recent_revisit:   reward += RECENT_REVISIT_PENALTY
        delivered_now = False
        if vehicle.current_location in self.demands and vehicle.capacity > 0:
            vehicle.capacity -= 1
            self.demands[vehicle.current_location] -= 1
            reward += DELIVERY_BONUS
            delivered_now = True
            if self.demands[vehicle.current_location] == 0:
                del self.demands[vehicle.current_location]
        next_state = self.get_state_from_external(vehicle)
        if len(self.demands) > 0:
            near_next = self._nearest_demand_dist(vehicle.current_location)
            if (not delivered_now) and (near_next >= near_prev - PROGRESS_TOL):
                reward += NO_PROGRESS_PENALTY
                self.no_progress_streak += 1
            else:
                self.no_progress_streak = 0
        else:
            self.no_progress_streak = 0
        if delivered_now:
            self.steps_since_delivery = 0
            self.no_progress_streak = 0
        else:
            self.steps_since_delivery += 1
            reward += -0.002 * min(self.steps_since_delivery / self.stall_limit, 1.0)
        veh_at_home   = (vehicle.current_location == init_location)
        demands_empty = (len(self.demands) == 0)
        terminal_global = demands_empty and self.all_vehicles_at_depot()
        phi_next = self._potential(vehicle)
        gamma_t = 0.0 if terminal_global else self.gamma_rl
        reward += LAMBDA_PBRS * (gamma_t * phi_next - phi_prev)
        if (vehicle.capacity <= 0) and (not demands_empty) and veh_at_home:
            self.last_term_reason = "empty_home"
            return next_state, reward, False, True
        done_vehicle = False
        if demands_empty and veh_at_home:
            self.last_term_reason = "vehicle_home"
            done_vehicle = True
        if terminal_global:
            reward += FINAL_BONUS
            self.last_term_reason = "solved"
            return next_state, reward, True, True
        if self.steps_since_delivery >= self.stall_limit:
            reward += self.stall_penalty
            self.last_term_reason = "stall_vehicle"
            return next_state, reward, False, True
        if not np.isfinite(reward) or abs(reward) > 1e6:
            print(f"[SANITY] reward anomaly prev={prev_node} -> action={action}, "
                  f"new={vehicle.current_location}, reward={reward}")
            reward = float(np.clip(reward, -1e6, 1e6))
        return next_state, reward, False, done_vehicle

    def visited_slice(self, lst):
        return lst[-RECENT_WINDOW:] if len(lst) >= RECENT_WINDOW else lst

    def reset(self):
        self.location = init_location
        self.one_hot_location = self.one_hot_encode(init_location, self.n_locations)
        self.demands = {node: amount for node, amount in self.demands_init}
        self.one_hot_demands = self.one_hot_encode_demands()
        self.steps_since_delivery = 0
        self.last_term_reason = None
        self.no_progress_streak = 0


class Vehicle:
    def __init__(self, vehicle_id, capacity, environment):
        self.vehicle_id = vehicle_id
        self.initial_capacity = capacity
        self.capacity = capacity
        self.environment = environment
        self.current_location = init_location
        self.total_distance = 0.0
        self.total_duration = 0.0
        self.total_fuel = 0.0
        self.visited_location_time_window = [0]
        self.visited_time = [0.0]
        self.current_time_window = 0
        self.visited_locations = [init_location]
        self.visited_time_window = [0]
        self.done_vehicle = False

    def reset(self):
        self.current_location = init_location
        self.total_distance = 0.0
        self.total_duration = 0.0
        self.total_fuel = 0.0
        self.visited_location_time_window = [0]
        self.visited_time = [0.0]
        self.visited_locations = [init_location]
        self.visited_time_window = [0]
        self.current_time_window = 0
        self.done_vehicle = False
        self.capacity = self.initial_capacity

# ============================
# Senkron test koşturucu
# ============================
def run_single_scenario(sid: int):
    global init_location

    # --- Depo ---
    if RANDOM_DEPOT_ACTIVE:
        init_location = pick_random_depot(
            dst, degree_min=DEPOT_DEGREE_MIN,
            candidates=DEPOT_CANDIDATES,
            weighted=DEPOT_WEIGHTED_BY_DEG
        )
    else:
        init_location = int(FIXED_DEPOT_IDX)
    print(f"\n[SCENARIO {sid}] Depot={init_location}")

    # --- Talepler ---
    if FIXED_DEMAND_NODES is not None and len(FIXED_DEMAND_NODES) > 0:
        ep_demands = [[int(n), int(DEMAND_QTY_PER_NODE_TEST)]
                      for n in FIXED_DEMAND_NODES if int(n) != init_location]
    elif CURRICULUM_TEST_ACTIVE:
        k_demands = random.randint(CURRICULUM_DEMANDS_RANGE[0], CURRICULUM_DEMANDS_RANGE[1])
        ep_demands = choose_demands_with_priority(
            n_locations, init_location,
            k_demands, PRIORITY_NODES, PRIORITY_MIN,
            DEMAND_QTY_PER_NODE_TEST
        )
    else:
        ep_demands = []

    total_demand = int(sum(a for _, a in ep_demands))
    # --- Kapasiteler ---
    if VEHICLE_CAPACITIES is not None and len(VEHICLE_CAPACITIES) == VEHICLE_COUNT:
        caps = [int(c) for c in VEHICLE_CAPACITIES]
        if sum(caps) < total_demand:
            deficit = total_demand - sum(caps)
            add = split_sum_positive(deficit, VEHICLE_COUNT)
            caps = [c + add[i] for i, c in enumerate(caps)]
    else:
        caps = split_sum_positive(total_demand, VEHICLE_COUNT)

    n_vehicles = VEHICLE_COUNT
    print(f"[SCENARIO {sid}] demands={sorted([n for n,_ in ep_demands])} "
          f"| total={total_demand} | capacities={caps}")

    # Başlangıç grafik ve SP
    dst_initial = np.array(dst, dtype=np.float32)
    SP_DIST = dijkstra_all_sources(dst_initial)
    action_list, _ = action_matrix(dst_initial, MAX_ACTIONS)
    NONZERO_MASK = (dst_initial > 0).astype(np.uint8)
    total_edges_undir_initial = count_open_undirected(dst_initial, symmetric=FORCE_SYMMETRY)

    # ====================== DEPREM-1 (senaryo başı) ======================
    print("\n=== DEPREM#1 — initial ===")
    attempt = 0
    accepted = False
    dst_risk = np.array(dst_initial, dtype=np.float32)
    r_used_final = None

    while (not accepted) and (attempt < (RISK_TRIES_PRIMARY + RISK_TRIES_FALLBACK)):
        if RISK_R_FIXED is not None:
            r_used = float(RISK_R_FIXED)
        else:
            if attempt < RISK_TRIES_PRIMARY:
                r_used = float(np.random.uniform(*RISK_R_PRIMARY_RANGE))
            else:
                r_used = float(np.random.uniform(*RISK_R_FALLBACK_RANGE))

        critical_nodes = [init_location] + [int(n) for n,_ in ep_demands] if PROTECT_CRITICAL else None
        dm_tmp, closed_before = apply_risk_to_distance_safe(
            dst_initial, risk, r_used,
            critical_nodes=critical_nodes,
            symmetric=FORCE_SYMMETRY,
            protect_mode=PROTECT_CRITICAL_MODE,
            min_degree_per_critical=MIN_DEGREE_PER_CRIT
        )

        if REOPEN_ON_DEMAND and total_demand > 0:
            dm_final = ensure_connectivity(
                dm_tmp, dst_initial, init_location, [n for n,_ in ep_demands],
                reopen_limit_per_target=REOPEN_LIMIT_PER_TARGET
            )
        else:
            dm_final = dm_tmp

        closed_after = count_closed_undirected(dst_initial, dm_final, symmetric=FORCE_SYMMETRY)
        open_ratio  = open_ratio_undirected(dst_initial, dm_final, symmetric=FORCE_SYMMETRY)
        reopened    = closed_before - closed_after

        print(f"[RISK@try{attempt}] r={r_used:.4f} | closed_before={closed_before} "
              f"| closed_after={closed_after} | reopened={reopened} "
              f"| open_ratio={open_ratio*100:.1f}%")

        if (RISK_TARGET_OPEN_FRAC[0] <= open_ratio <= RISK_TARGET_OPEN_FRAC[1]):
            dst_risk = dm_final
            r_used_final = r_used
            accepted = True
            break
        attempt += 1

    # KANTİL fallback
    if not accepted:
        target_mid = 0.5 * (RISK_TARGET_OPEN_FRAC[0] + RISK_TARGET_OPEN_FRAC[1])
        r_q = r_for_target_open_ratio(dst_initial, risk, target_mid, symmetric=FORCE_SYMMETRY)
        critical_nodes = [init_location] + [int(n) for n,_ in ep_demands] if PROTECT_CRITICAL else None
        dm_tmp, _ = apply_risk_to_distance_safe(
            dst_initial, risk, r_q,
            critical_nodes=critical_nodes,
            symmetric=FORCE_SYMMETRY,
            protect_mode=PROTECT_CRITICAL_MODE,
            min_degree_per_critical=MIN_DEGREE_PER_CRIT
        )
        if REOPEN_ON_DEMAND and total_demand > 0:
            dst_risk = ensure_connectivity(
                dm_tmp, dst_initial, init_location, [n for n,_ in ep_demands],
                reopen_limit_per_target=REOPEN_LIMIT_PER_TARGET
            )
        else:
            dst_risk = dm_tmp
        r_used_final = r_q

    # Graf değiştiyse SP/komşuluk güncelle
    arr_now  = np.array(dst_risk)
    mask_now = (arr_now > 0).astype(np.uint8)
    if np.any(mask_now != NONZERO_MASK):
        changed_edges = int(np.sum(mask_now != NONZERO_MASK))
        print(f"[Graph change] nonzero edges changed: {changed_edges} (r≈{r_used_final:.3f}). Recompute SP ...")
        SP_DIST = dijkstra_all_sources(dst_risk)
        action_list, _ = action_matrix(dst_risk, MAX_ACTIONS)
        NONZERO_MASK = mask_now

    # Başlangıç deprem raporu
    closed_initial_list = closed_edges_undirected(dst_initial, dst_risk, symmetric=FORCE_SYMMETRY)
    initial_open_ratio  = open_ratio_undirected(dst_initial, dst_risk, symmetric=FORCE_SYMMETRY)
    print("\n=== DEPREM#1 (EPIZOT BAŞI) RAPOR ===")
    print(f"Kapanan yol adedi (undirected): {len(closed_initial_list)} | Açıklık oranı: {initial_open_ratio*100:.1f}%")
    if len(closed_initial_list) <= PRINT_CLOSED_EDGES_LIMIT:
        print("Kapanan yollar (i<j):", closed_initial_list)
    else:
        head = closed_initial_list[:PRINT_CLOSED_EDGES_LIMIT]
        print(f"Kapanan yollar (ilk {PRINT_CLOSED_EDGES_LIMIT}/{len(closed_initial_list)}):", head)

    _ = check_connectivity_and_routes(dst_risk, init_location, [n for n,_ in ep_demands])

    # Ortam/araçlar
    env = Environment(dst_risk, ep_demands, n_vehicles, caps, n_locations, SP_DIST)
    vehicles = [Vehicle(i, caps[i], env) for i in range(n_vehicles)]
    env.attach_vehicles(vehicles)
    env.reset()
    for v in vehicles: v.reset()

    # ---------------- ROTA SEGMENT ve senkron ilerleme ----------------
    segment_info = {vi: {"start_idx": 0, "start_dist": 0.0, "segments": []} for vi in range(n_vehicles)}

    def log_and_cut_segments(event_label: str, end_global_step: int):
        print(f"\n--- [SEGMENT SNAPSHOT] {event_label} ---")
        for vi, v in enumerate(vehicles):
            si = segment_info[vi]
            s_idx = int(si["start_idx"])
            path_seg = v.visited_locations[s_idx:] or [v.current_location]
            seg_dist = float(v.total_distance - si["start_dist"])
            edges_seg = list(zip(path_seg[:-1], path_seg[1:]))
            print(f"[veh={vi}] start={path_seg[0]} -> end={path_seg[-1]} "
                  f"| nodes={path_seg} | edges={edges_seg} "
                  f"| seg_dist={seg_dist:.2f} | cum_dist={v.total_distance:.2f}")
            si["segments"].append({
                "label": event_label,
                "path": list(path_seg),
                "edges": list(edges_seg),
                "distance": seg_dist,
                "cum_distance": float(v.total_distance),
                "end_step": int(end_global_step)
            })
            si["start_idx"]  = len(v.visited_locations) - 1
            si["start_dist"] = float(v.total_distance)

    # EQ#1 anlık segment
    log_and_cut_segments(event_label="EQ#1 (initial)", end_global_step=0)

    total_points    = 0.0
    deliveries_done = 0
    backtracks_test = 0

    shield_frac = float(np.clip(SHIELD_TEST_FRAC, 0.0, 1.0))

    global_step        = 0
    scheduled_steps    = set(ONLINE_RISK_AT_GLOBAL_STEPS or [])
    online_events_done = 0
    online_event_id    = 0

    def maybe_apply_online_risk():
        """Deprem (online) uygula; bu olayda kapanan kenarları ve açık oranı yaz."""
        nonlocal env, vehicles, SP_DIST, action_list, online_events_done, online_event_id

        if not ONLINE_RISK_UPDATES_ACTIVE:
            return

        trigger = False
        if global_step in scheduled_steps:
            scheduled_steps.discard(global_step); trigger = True
        if (not trigger) and ONLINE_RISK_EVERY_K_STEPS:
            if global_step > 0 and (global_step % int(ONLINE_RISK_EVERY_K_STEPS) == 0):
                trigger = True
        if (not trigger) or (online_events_done >= ONLINE_RISK_MAX_EVENTS):
            return

        online_events_done += 1
        online_event_id    += 1

        # Olay hedefleri
        event_ix   = max(1, online_event_id)
        tgt_idx    = min(event_ix-1, len(ONLINE_EVENT_TARGET_OPEN_FRACS)-1)
        budget_idx = min(event_ix-1, len(ONLINE_EVENT_BUDGET)-1)

        target_open  = float(ONLINE_EVENT_TARGET_OPEN_FRACS[tgt_idx])
        budget_edges = ONLINE_EVENT_BUDGET[budget_idx]

        # SAFE set
        rem_demands = sorted(list(env.demands.keys()))
        safe_edges = set()
        if ONLINE_PRESERVE_SP_PAIRS and len(rem_demands) > 0:
            pairs1 = [(int(init_location), int(d)) for d in rem_demands]  # depo->talep
            pairs2 = [(int(v.current_location), int(init_location)) for v in vehicles]  # araç->depo
            safe_edges |= sp_edge_set_undirected(env.dst, pairs1 + pairs2)
        if ONLINE_SAFE_HOPS and ONLINE_SAFE_HOPS > 0:
            sources = [int(v.current_location) for v in vehicles]
            halo = bfs_halo_nodes(env.dst, sources, hops=int(ONLINE_SAFE_HOPS))
            for u in halo:
                nbrs = np.where(env.dst[u] > 0)[0]
                for v in nbrs:
                    a,b = (u,v) if u < v else (v,u)
                    safe_edges.add((a,b))

        r_on   = r_for_target_open_ratio(dst_initial, risk, target_open, symmetric=ONLINE_FORCE_SYMMETRY)
        before = np.array(env.dst, dtype=np.float32)

        # Bütçeli + SAFE korumalı kapama
        new_dst, _ = selective_close_with_budget(
            env.dst, risk, r_on,
            budget=budget_edges,
            safe_edges=safe_edges,
            symmetric=ONLINE_FORCE_SYMMETRY
        )

        # Reopen + kaçış hatları
        if ONLINE_REOPEN_ON_DEMAND and len(rem_demands) > 0:
            new_dst = ensure_connectivity(
                new_dst, dst_initial, init_location, rem_demands,
                reopen_limit_per_target=ONLINE_REOPEN_LIMIT_PER_TARGET
            )
        if ONLINE_ENSURE_VEHICLE_TO_DEPOT:
            pairs = [(int(v.current_location), int(init_location)) for v in vehicles]
            new_dst = ensure_connectivity_pairs(new_dst, dst_initial, pairs)

        # Yaz & güncelle
        env.dst = new_dst
        SP_DIST = dijkstra_all_sources(new_dst)
        env.sp_dist = SP_DIST
        action_list, _ = action_matrix(new_dst, MAX_ACTIONS)
        env.action_list = action_list

        after = np.array(new_dst, dtype=np.float32)
        ratio = open_ratio_undirected(dst_initial, after, symmetric=ONLINE_FORCE_SYMMETRY)
        closed_now_list = closed_edges_undirected(before, after, symmetric=ONLINE_FORCE_SYMMETRY)
        changes = int(np.sum((before > 0) != (after > 0)))

        print(f"\n=== DEPREM#{online_event_id} (ONLINE) RAPOR ===")
        print(f"step={global_step} | hedef_açıklık≈{target_open*100:.1f}% | r≈{r_on:.3f} "
              f"| açık_oranı={ratio*100:.1f}% | değişen_hücre={changes}")
        print(f"Bu olayda kapanan yol adedi (undirected): {len(closed_now_list)}")
        if len(closed_now_list) <= PRINT_CLOSED_EDGES_LIMIT:
            print("Kapanan yollar (i<j):", closed_now_list)
        else:
            head = closed_now_list[:PRINT_CLOSED_EDGES_LIMIT]
            print(f"Kapanan yollar (ilk {PRINT_CLOSED_EDGES_LIMIT}/{len(closed_now_list)}):", head)

        # Segment raporu
        log_and_cut_segments(event_label=f"EQ#{online_event_id} (r≈{r_on:.3f})", end_global_step=global_step)

        # Çok açık kaldıysa yumuşak retry
        retry = 0
        while ratio > max(ONLINE_MAX_OPEN_RATIO, target_open + 0.02) and retry < ONLINE_RETRY_MAX:
            r_on2  = max(0.0, r_on - 0.05*(retry+1))
            before2 = np.array(env.dst, dtype=np.float32)
            new_dst2, _ = selective_close_with_budget(
                env.dst, risk, r_on2,
                budget=budget_edges,
                safe_edges=safe_edges,
                symmetric=ONLINE_FORCE_SYMMETRY
            )
            if ONLINE_REOPEN_ON_DEMAND and len(rem_demands) > 0:
                new_dst2 = ensure_connectivity(
                    new_dst2, dst_initial, init_location, rem_demands,
                    reopen_limit_per_target=ONLINE_REOPEN_LIMIT_PER_TARGET
                )
            if ONLINE_ENSURE_VEHICLE_TO_DEPOT:
                pairs = [(int(v.current_location), int(init_location)) for v in vehicles]
                new_dst2 = ensure_connectivity_pairs(new_dst2, dst_initial, pairs)

            env.dst = new_dst2
            SP_DIST = dijkstra_all_sources(new_dst2)
            env.sp_dist = SP_DIST
            action_list, _ = action_matrix(new_dst2, MAX_ACTIONS)
            env.action_list = action_list

            after2 = np.array(new_dst2, dtype=np.float32)
            ratio2 = open_ratio_undirected(dst_initial, after2, symmetric=ONLINE_FORCE_SYMMETRY)
            closed_retry_list = closed_edges_undirected(before2, after2, symmetric=ONLINE_FORCE_SYMMETRY)
            print(f"[ONLINE-RISK] retry#{retry+1}: r≈{r_on2:.3f} | açık_oranı={ratio2*100:.1f}% "
                  f"| ek_kapanan_yol={len(closed_retry_list)}")
            r_on  = r_on2
            ratio = ratio2
            retry += 1

        # Çok sert olduysa hafif gevşet (minimum açık oran)
        if ratio < ONLINE_MIN_OPEN_RATIO_AFTER_EVENT:
            r_relax = min(1.0, r_on + 0.05)
            before3 = np.array(env.dst, dtype=np.float32)
            new_dst3, _ = selective_close_with_budget(
                before, risk, r_relax,
                budget=budget_edges,
                safe_edges=safe_edges,
                symmetric=ONLINE_FORCE_SYMMETRY
            )
            if ONLINE_REOPEN_ON_DEMAND and len(rem_demands) > 0:
                new_dst3 = ensure_connectivity(
                    new_dst3, dst_initial, init_location, rem_demands,
                    reopen_limit_per_target=ONLINE_REOPEN_LIMIT_PER_TARGET
                )
            if ONLINE_ENSURE_VEHICLE_TO_DEPOT:
                pairs = [(int(v.current_location), int(init_location)) for v in vehicles]
                new_dst3 = ensure_connectivity_pairs(new_dst3, dst_initial, pairs)
            env.dst = new_dst3
            SP_DIST = dijkstra_all_sources(new_dst3)
            env.sp_dist = SP_DIST
            action_list, _ = action_matrix(new_dst3, MAX_ACTIONS)
            env.action_list = action_list
            ratio3 = open_ratio_undirected(dst_initial, new_dst3, symmetric=ONLINE_FORCE_SYMMETRY)
            closed_relax_list = closed_edges_undirected(before3, new_dst3, symmetric=ONLINE_FORCE_SYMMETRY)
            print(f"[ONLINE-RISK] relax: r≈{r_relax:.3f} | açık_oranı={ratio3*100:.1f}% "
                  f"| bu_adımda_kapanan={len(closed_relax_list)}")

    # --- SENKRON İLERLEME ---
    print("\n[ACTION] Araçlar eşzamanlı olarak rotaya çıkıyor.")
    done_flags = [False]*n_vehicles
    t_loop = 0
    while (t_loop < MAX_TIMESTEPS_TEST) and (not all(done_flags)):
        for vid, vehicle in enumerate(vehicles):
            if done_flags[vid]:
                continue

            # Deprem tetiklemesi (global_step eşiği yakalandıysa)
            maybe_apply_online_risk()

            state = env.get_state_from_external(vehicle)

            if vehicle.capacity <= 0 and vehicle.current_location == init_location:
                env.last_term_reason = "empty_home"
                done_flags[vid] = True
                print(f"[TEST] veh={vid} early terminal: empty_home(idle)")
                continue

            history = vehicle.visited_locations
            prev_prev = history[-2] if len(history) >= 2 else None
            tabu_nodes = set(history[-TABU_RECENT:])

            shield_gate = (random.random() < shield_frac)
            use_shield = SHIELD_ACTIVE and shield_gate and (
                env.steps_since_delivery >= SHIELD_KICKIN_STEPS or
                env.no_progress_streak   >= NP_STREAK_THRESHOLD or
                env.is_looping(history)
            )

            if vehicle.capacity <= 0:
                force_first = env.next_hop_to_home(vehicle.current_location)
            else:
                if len(env.demands) == 0:
                    force_first = env.next_hop_to_home(vehicle.current_location)
                else:
                    force_first = env.next_hop_to_nearest_demand(vehicle.current_location) if use_shield else None

            slots, neigh_nodes = env.topk_action_indices(
                vehicle.current_location,
                force_first=force_first,
                tabu_nodes=tabu_nodes,
                return_neighbors=True
            )
            n_valid = len(slots)

            if n_valid > 1 and prev_prev is not None:
                filtered = [(s, n) for s, n in zip(slots, neigh_nodes) if n != prev_prev]
                if len(filtered) > 0:
                    slots, neigh_nodes = zip(*filtered)
                    slots, neigh_nodes = list(slots), list(neigh_nodes)
                    n_valid = len(slots)

            # Greedy seçim (test)
            if n_valid > 0:
                if (vehicle.capacity <= 0) or (len(env.demands) == 0) or use_shield:
                    action_index = int(slots[0])
                else:
                    q_vals = q_network(state.reshape(1, -1)).numpy()[0]
                    best_slot = slots[int(np.argmax(q_vals[slots]))]
                    action_index = int(best_slot)
                action = int(env.action_list[vehicle.current_location][action_index])
            else:
                action, action_index = None, None

            # İlerleme kapısı
            if (vehicle.capacity > 0) and (len(env.demands) > 0) and (not use_shield) and (action is not None):
                cur_near  = env._nearest_demand_dist(vehicle.current_location)
                cand_near = env._nearest_demand_dist(action)
                if cand_near >= cur_near - PROGRESS_TOL and env.steps_since_delivery >= SHIELD_KICKIN_STEPS//2:
                    action_index = int(slots[0])
                    action = int(env.action_list[vehicle.current_location][action_index])

            cap_before = vehicle.capacity
            next_state, reward, done, done_vehicle = env.step(vehicle, action, action_index)
            total_points += reward

            if action is not None:
                global_step += 1

            if len(vehicle.visited_locations) >= 3:
                if vehicle.visited_locations[-1] == vehicle.visited_locations[-3]:
                    backtracks_test += 1
            if vehicle.capacity < cap_before:
                deliveries_done += 1

            if done_vehicle:
                done_flags[vid] = True
                print(f"[TEST] veh={vid} terminal reason={env.last_term_reason}, "
                      f"loc={vehicle.current_location}, steps_since_delivery={env.steps_since_delivery}")

        t_loop += 1
        if all(done_flags):
            break

    # Final segment
    log_and_cut_segments(event_label="FINAL", end_global_step=global_step)

    # Özet
    print("\n=== SCENARIO RESULT ===")
    print(f"Total shaped reward: {total_points:.3f} | deliveries={deliveries_done} | backtracks={backtracks_test}")
    print("All vehicles at depot? ", all(v.current_location == init_location for v in vehicles))
    for vi, v in enumerate(vehicles):
        delivered_vi = v.initial_capacity - v.capacity
        print(f"  veh {vi}: end_loc={v.current_location}, "
              f"cap_left={v.capacity}, delivered={delivered_vi}, distance={v.total_distance:.2f}")
        print(f"  route (full nodes): {v.visited_locations}")
        print(f"  route (full edges): {list(zip(v.visited_locations[:-1], v.visited_locations[1:]))}")

    print("\n=== ROUTE SEGMENTS SUMMARY ===")
    for vi in range(n_vehicles):
        print(f"[veh={vi}]")
        for k, seg in enumerate(segment_info[vi]["segments"], 1):
            print(f"  seg#{k} label={seg['label']} | end_step={seg['end_step']} "
                  f"| dist={seg['distance']:.2f} | cum={seg['cum_distance']:.2f} "
                  f"| path={seg['path']} | edges={seg['edges']}")

    if PROTECT_CRITICAL and PROTECT_CRITICAL_MODE == 'min_degree':
        crits = [init_location] + sorted([n for n,_ in ep_demands])
        degs = {int(c): int(np.sum(env.dst[int(c), :] > 0)) for c in crits}
        print("[RISK] critical degrees after protection:", degs)

    return {
        "reward": total_points,
        "deliveries": deliveries_done,
        "backtracks": backtracks_test,
        "all_home": all(v.current_location == init_location for v in vehicles),
        "vehicles": [{
            "id": vi,
            "end_loc": v.current_location,
            "cap_left": v.capacity,
            "delivered": v.initial_capacity - v.capacity,
            "distance": v.total_distance,
            "route_nodes": v.visited_locations,
            "route_edges": list(zip(v.visited_locations[:-1], v.visited_locations[1:])),
            "segments": segment_info[vi]["segments"]
        } for vi, v in enumerate(vehicles)]
    }

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    if tf.config.list_physical_devices('GPU'):
        print("GPU kullanılabilir.")
    else:
        print("GPU kullanılabilir değil, CPU kullanılıyor.")

    start = time.time()
    results = []
    for sid in range(N_SCENARIOS):
        res = run_single_scenario(sid)
        results.append(res)

    tot_time = time.time() - start
    print(f"\nTotal Inference Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
