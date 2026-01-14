import time
from collections import deque, namedtuple
import heapq
import random
import os

import numpy as np
import tensorflow as tf
import pandas as pd

from keras.layers import Dense, Input, Lambda
from keras.models import Model

# ============================
# Kullanıcı parametreleri (senaryoyu buradan ayarla)
# ============================
MODEL_PATH = r"cit_q_network_kcekmece.h5"   # eğitimde kaydettiğin dosya

DATA_DISTANCE_XLSX = "kcekmece_distance.xlsx"
DATA_RISK_XLSX     = "kcekmece_risk_matris.xlsx"
SHEET_NAME = "Sheet1"

# Aksiyon uzayı (eğitimdeki ile aynı olmalı)
MAX_ACTIONS = 8

# Senaryo adedi (kaç farklı senaryo koşulsun?)
N_SCENARIOS =  1 #3

# --- DEPOT ayarları ---
RANDOM_DEPOT_ACTIVE   = False#True     # True: her senaryoda rastgele depo seç
DEPOT_DEGREE_MIN      = 3
DEPOT_CANDIDATES      = None     # Örn [0, 316, 10] (None: tüm düğümler)
DEPOT_WEIGHTED_BY_DEG = True     # Out-degree ile ağırlıklı seçim
FIXED_DEPOT_IDX       = 663    # RANDOM_DEPOT_ACTIVE=False ise burası kullanılır

# --- DEMAND (talepler) ---
# 1) Manuel liste: None değilse kullanılır, curriculum devre dışı kalır.
#    Örn: FIXED_DEMAND_NODES = [316, 464, 607, 748]
FIXED_DEMAND_NODES = [464, 607, 748, 316] #None
DEMAND_QTY_PER_NODE_TEST = 1     # manuel veya curriculum fark etmez

# 2) Curriculum+Öncelik: FIXED_DEMAND_NODES None ise çalışır
CURRICULUM_TEST_ACTIVE    = True
CURRICULUM_DEMANDS_RANGE  = (2, 10)
PRIORITY_NODES            = None #[316, 464, 607, 748]
PRIORITY_MIN              = 2     # her senaryoda en az bu kadar öncelikli düğüm

# --- Araçlar ---
VEHICLE_COUNT        = 3
VEHICLE_CAPACITIES   = None  # Liste vermezsen toplam talebe göre otomatik eşitlenir [2,2,1] vb.

# ============================
# RİSK / KAPANMA KONTROL PARAMETRELERİ
# ============================
# Hedef kapanma oranı modu (önerilir): her senaryoda kenarların %5–%15'i kapansın.
TARGET_CLOSURE_ACTIVE   = False
CLOSE_TARGET_RANGE      = (0.05, 0.15)  # min%, max% (0.05= %5, 0.15= %15)
CLOSURE_ATTEMPTS        = 8             # hedeften sapıyorsa r'ı 8 kez yeniden dener
CLOSURE_TOL             = 0.02          # bant toleransı (±%2 gibi düşün)

# Eski mantık (fallback) için parametreler:
RISK_MAX_ATTEMPTS = 6
RISK_BASE_RANGE   =(2.00, 2.00) #(0.00, 1.00)    # hedef kapanma bulunamazsa bu aralıktan r seç
RISK_SOFT_RANGE   = (2.00, 2.00)#(0.00, 1.00)    # daha da esnek denemeler için
RISK_R_FIXED      = 2#None            # None ise sabit eşik yok; sayı verirsen sabit r
FORCE_SYMMETRY    = True            # i->j veya j->i kapandıysa ikisini de kapat
# Kabul kriteri: minimum açık oran (%); hedef modundaysak otomatik ayarlanır.
MIN_OPEN_RATIO_BASE = 0.40          # hedef modunda otomatik 1 - CLOSE_TARGET_RANGE[1]

# Kritik düğümler = depo + talep (koruma modu)
PROTECT_CRITICAL        = True
PROTECT_CRITICAL_MODE   = 'min_degree'   # 'all' | 'min_degree'
MIN_DEGREE_PER_CRIT     = 1              # 'min_degree' modunda min açık derecesi

# Kopuk hedef için orijinal SP üzerindeki kapalı kenarları aç (yakınsadığı kanıtlı pratik)
REOPEN_ON_DEMAND = True

# --- Shield ve test davranışları ---
SHIELD_ACTIVE     = True
SHIELD_TEST_FRAC  = 1.0       # testte shield’ın devreye girme olasılığı (1.0: hep açık koşullarda)
SHIELD_KICKIN_STEPS = 20
NP_STREAK_THRESHOLD  = 15
TABU_RECENT          = 16
LOOP_WINDOW          = 32
LOOP_UNIQUE_MAX      = 7
PROGRESS_TOL         = 1e-6

# Top‑K karıştırma (sadece eğitimde önerilir) -> testte kapalı
TRAIN_TOPK_SHUFFLE = False
TOPK_SHUFFLE_PROB  = 0.0

# Adım/ödül parametreleri (eğitimle tutarlı)
STALL_LIMIT            = 400
STALL_PENALTY          = -20.0
BACKTRACK_PENALTY      = -0.30
NO_PROGRESS_PENALTY    = -0.40
RECENT_REVISIT_PENALTY = -0.20
RECENT_WINDOW          = 8
C_STEP                 = 0.02
LAMBDA_PBRS            = 2.5
ALPHA_NEAR             = 1.2
ALPHA_HOME             = 2.5
UNREACHABLE_NEAR_MULT  = 50.0
ALPHA_HOME_ESCAPE      = 6.0
ILLEGAL_PENALTY        = -1.0
DELIVERY_BONUS         =  1.2
FINAL_BONUS            =  7.5

# Maks adım
MAX_TIMESTEPS_TEST = 500

# Rastgelelik
SEED = None  # tam rassallık için None; istersen sabit bir int ver

# ============================
# Yardımcı fonksiyonlar
# ============================
def setup_seeds(seed):
    """seed=None ise tam rastgele; TF'de seed ayarlamayız."""
    if seed is None:
        np.random.seed(None)   # OS entropy
        random.seed()          # system time
        # tf.random.set_seed() AYARLAMADIK -> tam rassallık
    else:
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

setup_seeds(SEED)

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

def count_edges(mat, symmetric=True):
    """Toplam kenar sayısı: symmetric=True ise üst üçgen (i<j) üzerinden sayar."""
    m = (mat > 0)
    if symmetric:
        return int(np.sum(np.triu(m, k=1)))
    else:
        return int(np.sum(m))

def apply_risk_to_distance_safe(dm, rm, r, critical_nodes=None, symmetric=True,
                                protect_mode='all', min_degree_per_critical=1):
    dm = np.array(dm, dtype=np.float32)
    rm = np.nan_to_num(np.array(rm, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)

    # 1) Eşikle kapatılacak yerler (yönlü)
    cond = (dm > 0) & (rm > float(r))
    if symmetric:
        cond = np.logical_or(cond, cond.T)

    # 2) Kapatmaları uygula
    new_dm = np.where(cond, 0.0, dm)

    # 3) Kritik düğümleri koru
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
                    new_dm[pick, c] = dm[pick, c]  # simetri

    # 4) Nihai kapalı kenar sayısı (bilgi amaçlı)
    eff_closed = (dm > 0) & (new_dm <= 0)
    if symmetric:
        closed_count = int(np.sum(np.triu(np.logical_or(eff_closed, eff_closed.T), k=1)))
    else:
        closed_count = int(np.sum(eff_closed))

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
# Ortam sınıfları (eğitimdeki ile aynı mantık)
# ============================
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
        v = np.zeros(size)
        v[index] = 1
        return v

    def one_hot_encode_demands(self):
        v = np.zeros(self.n_locations)
        for node in self.demands:
            v[node] = 1
        return v

    def get_state_from_external(self, vehicle):
        vehicle_capacity = vehicle.capacity / max(1, vehicle.initial_capacity)
        self.one_hot_location = self.one_hot_encode(vehicle.current_location, self.n_locations)
        self.one_hot_demands = self.one_hot_encode_demands()
        return np.concatenate((self.one_hot_location, [vehicle_capacity], self.one_hot_demands))

    def attach_vehicles(self, vehicles):
        self.vehicles_ref = vehicles

    def all_vehicles_at_depot(self):
        if not self.vehicles_ref:
            return False
        return all(v.current_location == init_location for v in self.vehicles_ref)

    def _can_reach_any_demand(self, node: int) -> bool:
        if len(self.demands) == 0:
            return True
        dem_idx = list(self.demands.keys())
        row = self.sp_dist[node, dem_idx]
        return np.isfinite(row).any()

    def _nearest_demand_dist(self, from_node: int) -> float:
        if len(self.demands) == 0:
            return 0.0
        idxs = list(self.demands.keys())
        d = self.sp_dist[from_node, idxs]
        m = float(np.min(d)) if len(d) else np.inf
        if not np.isfinite(m):
            return self.d_scale * UNREACHABLE_NEAR_MULT
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
        if len(self.demands) == 0:
            return None
        dem_idx = list(self.demands.keys())
        drow = self.sp_dist[src, dem_idx]
        if not np.isfinite(drow).any():
            return None
        tgt = dem_idx[int(np.argmin(drow))]
        G = build_graph(self.dst)
        dist, path = dijkstra_path_with_parents(G, src, tgt)
        if not np.isfinite(dist) or len(path) < 2:
            return None
        return int(path[1])

    def next_hop_to_home(self, src: int):
        G = build_graph(self.dst)
        dist, path = dijkstra_path_with_parents(G, src, init_location)
        if not np.isfinite(dist) or len(path) < 2:
            return None
        return int(path[1])

    def is_looping(self, visited: list) -> bool:
        if len(visited) < LOOP_WINDOW:
            return False
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
                    slots = slots[keep]
                    neighbor_nodes = neighbor_nodes[keep]
                    best = best[keep]
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

        slots = slots[order]
        neighbor_nodes = neighbor_nodes[order]

        if tabu_nodes:
            mask = np.array([n not in tabu_nodes for n in neighbor_nodes], dtype=bool)
            if force_first is not None:
                mask = np.logical_or(mask, neighbor_nodes == force_first)
            slots = slots[mask]
            neighbor_nodes = neighbor_nodes[mask]
            if len(slots) == 0:
                slots = np.arange(min(n_all, NUM_ACTIONS), dtype=np.int32)[order]
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
        invalid = (
            action_index is None or
            action_index >= n_valid or
            action is None or
            action < 0 or action >= self.n_locations or
            self.dst[vehicle.current_location][action] <= 0 or
            action == vehicle.current_location
        )

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
        if will_backtrack:
            reward += BACKTRACK_PENALTY
        if recent_revisit:
            reward += RECENT_REVISIT_PENALTY

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
# Dueling‑DQN mimarisi ve model yükleme
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
    # Mimarinin aynısını kur
    model = build_dueling_dqn(state_size, num_actions)
    # Önce doğrudan ağırlıkları yüklemeyi dene
    try:
        model.load_weights(model_path)
        print(f"[MODEL] Weights loaded via load_weights() from: {model_path}")
        return model
    except Exception as e1:
        print(f"[MODEL] load_weights failed: {e1}. Trying load_model() fallback...")
        try:
            full = tf.keras.models.load_model(
                model_path,
                custom_objects={"dueling_combine": dueling_combine}
            )
            model.set_weights(full.get_weights())
            print(f"[MODEL] Loaded via load_model() and transferred weights.")
            return model
        except Exception as e2:
            raise RuntimeError(f"Could not load model from {model_path}. "
                               f"First error: {e1} | Fallback error: {e2}")

# ============================
# Veri yükle
# ============================
dst = load_square_matrix_xlsx(DATA_DISTANCE_XLSX, sheet_name=SHEET_NAME)
risk = load_square_matrix_xlsx(DATA_RISK_XLSX, sheet_name=SHEET_NAME)
assert dst.shape == risk.shape, "Mesafe ve risk matrisleri aynı boyutta olmalı!"
n_locations = len(dst)
N_LOC = n_locations

# Ölçek
_arr = np.array(dst, dtype=np.float32)
_pos = _arr[_arr > 0]
D_SCALE = float(np.percentile(_pos, 95)) if _pos.size > 0 else 1.0

# Risk yüzdelikleri (kalibrasyon için 1 kez yazdır)
edge_mask = (dst > 0)
risk_vals = risk[edge_mask]
if risk_vals.size > 0:
    p80 = np.percentile(risk_vals, 80)
    p85 = np.percentile(risk_vals, 85)
    p90 = np.percentile(risk_vals, 90)
    p95 = np.percentile(risk_vals, 95)
    p99 = np.percentile(risk_vals, 99)
    print(f"[RISK STATS] p80={p80:.3f} p85={p85:.3f} p90={p90:.3f} p95={p95:.3f} p99={p99:.3f} max={risk_vals.max():.3f}")

# Modeli hazırla + yükle
state_size = 2 * n_locations + 1
num_actions = MAX_ACTIONS
q_network = load_trained_q_network(MODEL_PATH, state_size, num_actions)
NUM_ACTIONS = int(q_network.output_shape[-1])
assert NUM_ACTIONS == MAX_ACTIONS, f"Model çıkış sayısı ({NUM_ACTIONS}) MAX_ACTIONS ({MAX_ACTIONS}) ile uyumlu olmalı!"

# ============================
# Test koşturucu
# ============================
def run_single_scenario(sid: int):
    global init_location

    # --- Depo seçimi ---
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
        ep_demands = []  # hiç talep yoksa, araçlar eve dönerek bitecek

    total_demand = int(sum(a for _, a in ep_demands))
    # --- Araçlar & kapasiteler ---
    if VEHICLE_CAPACITIES is not None and len(VEHICLE_CAPACITIES) == VEHICLE_COUNT:
        caps = [int(c) for c in VEHICLE_CAPACITIES]
        if sum(caps) < total_demand:
            # toplam talebi taşıyamıyorsa kalan otomatik bölünsün
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
    NONZERO_MASK = (dst_initial > 0).astype(np.uint8)
    total_edges_undirected = count_edges(dst_initial, symmetric=FORCE_SYMMETRY)

    # Otomatik minimum açık oran (hedef modunda)
    MIN_OPEN_RATIO = (1.0 - CLOSE_TARGET_RANGE[1]) if TARGET_CLOSURE_ACTIVE else MIN_OPEN_RATIO_BASE

    # --- EPİZOT BAŞI: Risk uygula ---
    attempt = 0
    applied = False

    # Hedef kapanma modu: risk eşiklerini yüzdeliklere göre örnekle
    if TARGET_CLOSURE_ACTIVE and risk_vals.size > 0 and RISK_R_FIXED is None:
        q_low  = 100.0 * (1.0 - CLOSE_TARGET_RANGE[1])  # örn 85
        q_high = 100.0 * (1.0 - CLOSE_TARGET_RANGE[0])  # örn 95
        # İlk r: orta nokta yüzdelik
        r_center = np.percentile(risk_vals, 0.5 * (q_low + q_high))

        while attempt < CLOSURE_ATTEMPTS:
            # denemede r'ı q_low..q_high aralığından örnekle (merkez etrafında)
            if attempt == 0:
                r_used = float(r_center)
            else:
                q_try = np.random.uniform(q_low, q_high)
                r_used = float(np.percentile(risk_vals, q_try))

            critical_nodes = [init_location] + sorted([int(n) for n,_ in ep_demands]) if PROTECT_CRITICAL else None
            dst_risk, closed_before = apply_risk_to_distance_safe(
                dst_initial, risk, r_used,
                critical_nodes=critical_nodes,
                symmetric=FORCE_SYMMETRY,
                protect_mode=PROTECT_CRITICAL_MODE,
                min_degree_per_critical=MIN_DEGREE_PER_CRIT
            )

            # Gerekirse bağlanırlığı zorla
            if REOPEN_ON_DEMAND and total_demand > 0:
                dst_risk = ensure_connectivity(dst_risk, dst_initial, init_location, [n for n,_ in ep_demands])

            # Son açık / kapalı sayıları (final grafik)
            open_edges_now  = count_edges(dst_risk, symmetric=FORCE_SYMMETRY)
            closed_after    = total_edges_undirected - open_edges_now
            open_ratio      = open_edges_now / max(1, total_edges_undirected)
            closed_ratio    = 1.0 - open_ratio

            print(f"[RISK@try{attempt}] r={r_used:.4f} | closed_before={closed_before} | "
                  f"closed_after={closed_after} | open_ratio={open_ratio*100:.1f}%")

            # Kriter: minimum açık oran ve hedef bandı
            if open_ratio >= MIN_OPEN_RATIO and (CLOSE_TARGET_RANGE[0] - CLOSURE_TOL) <= closed_ratio <= (CLOSE_TARGET_RANGE[1] + CLOSURE_TOL):
                applied = True
                break

            attempt += 1

        if not applied:
            print(f"[RISK] Targeted closure not met in {CLOSURE_ATTEMPTS} tries → fallback to base/random strategy.")

    # Eğer hedef modunda uygulanamadıysa, eski strateji ile dene
    if not applied:
        attempt = 0
        while True:
            if RISK_R_FIXED is not None:
                r_used = float(RISK_R_FIXED)
            else:
                r_lo, r_hi = RISK_BASE_RANGE if attempt < (RISK_MAX_ATTEMPTS // 2) else RISK_SOFT_RANGE
                r_used = float(np.random.uniform(r_lo, r_hi))

            critical_nodes = [init_location] + sorted([int(n) for n,_ in ep_demands]) if PROTECT_CRITICAL else None
            dst_risk, closed_before = apply_risk_to_distance_safe(
                dst_initial, risk, r_used,
                critical_nodes=critical_nodes,
                symmetric=FORCE_SYMMETRY,
                protect_mode=PROTECT_CRITICAL_MODE,
                min_degree_per_critical=MIN_DEGREE_PER_CRIT
            )

            if REOPEN_ON_DEMAND and total_demand > 0:
                dst_risk = ensure_connectivity(dst_risk, dst_initial, init_location, [n for n,_ in ep_demands])

            open_edges_now  = count_edges(dst_risk, symmetric=FORCE_SYMMETRY)
            closed_after    = total_edges_undirected - open_edges_now
            open_ratio      = open_edges_now / max(1, total_edges_undirected)

            print(f"[RISK@fallback{attempt}] r={r_used:.4f} | closed_before={closed_before} | "
                  f"closed_after={closed_after} | open_ratio={open_ratio*100:.1f}%")

            if open_ratio >= MIN_OPEN_RATIO:
                break

            attempt += 1
            if attempt >= RISK_MAX_ATTEMPTS:
                dst_risk = np.array(dst_initial, dtype=np.float32)
                open_edges_now  = count_edges(dst_risk, symmetric=FORCE_SYMMETRY)
                closed_after    = total_edges_undirected - open_edges_now
                open_ratio      = open_edges_now / max(1, total_edges_undirected)
                print(f"[RISK] fallback: using base graph (open_ratio<{MIN_OPEN_RATIO:.2f}). "
                      f"closed_after={closed_after} | open_ratio={open_ratio*100:.1f}%")
                break

    # SP / action list güncelle
    arr_now = np.array(dst_risk)
    mask_now = (arr_now > 0).astype(np.uint8)
    if np.any(mask_now != NONZERO_MASK):
        changed_edges = int(np.sum(mask_now != NONZERO_MASK))
        print(f"[Graph change] nonzero edges changed: {changed_edges}. Recompute SP ...")
        SP_DIST = dijkstra_all_sources(dst_risk)

    # Ulaşılamayan hedef var mı?
    unreachable = check_connectivity_and_routes(dst_risk, init_location, [n for n,_ in ep_demands])
    if len(unreachable) > 0:
        print(f"[WARN] Unreachable after risk+reopen: {unreachable}")

    # Ortam/araçları kur
    env = Environment(dst_risk, ep_demands, n_vehicles, caps, n_locations, SP_DIST)
    vehicles = [Vehicle(i, caps[i], env) for i in range(n_vehicles)]
    env.attach_vehicles(vehicles)
    env.reset()
    for v in vehicles:
        v.reset()

    total_points = 0.0
    deliveries_done = 0
    backtracks_test = 0

    # --- Shield oranı (testte sabit frac) ---
    shield_frac = float(np.clip(SHIELD_TEST_FRAC, 0.0, 1.0))

    # Araçları sırayla çalıştır
    for vid, vehicle in enumerate(vehicles):
        done_vehicle = False
        env.steps_since_delivery = 0
        env.no_progress_streak   = 0

        for t in range(MAX_TIMESTEPS_TEST):
            state = env.get_state_from_external(vehicle)

            # Zero-capacity & evde ise hareket etmesin
            if vehicle.capacity <= 0 and vehicle.current_location == init_location:
                env.last_term_reason = "empty_home"
                print(f"[TEST] veh={vid} early terminal: empty_home(idle)")
                break

            history = vehicle.visited_locations
            prev_prev = history[-2] if len(history) >= 2 else None
            tabu_nodes = set(history[-TABU_RECENT:])

            # Shield kapısı
            shield_gate = (random.random() < shield_frac)
            use_shield = SHIELD_ACTIVE and shield_gate and (
                env.steps_since_delivery >= SHIELD_KICKIN_STEPS or
                env.no_progress_streak   >= NP_STREAK_THRESHOLD or
                env.is_looping(history)
            )

            # Force SP‑hop (eve veya en yakın talebe)
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

            # 2‑cycle (u‑turn) slotunu çıkar (birden fazla seçenek varsa)
            if n_valid > 1 and prev_prev is not None:
                filtered = [(s, n) for s, n in zip(slots, neigh_nodes) if n != prev_prev]
                if len(filtered) > 0:
                    slots, neigh_nodes = zip(*filtered)
                    slots, neigh_nodes = list(slots), list(neigh_nodes)
                    n_valid = len(slots)

            # Top‑K shuffle → testte kapalı
            if TRAIN_TOPK_SHUFFLE and n_valid > 1 and random.random() < TOPK_SHUFFLE_PROB:
                if force_first is not None:
                    head = [slots[0]]
                    tail = list(slots[1:])
                    random.shuffle(tail)
                    slots = head + tail
                else:
                    slots = list(slots)
                    random.shuffle(slots)

            # Aksiyon seçimi (greedy)
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

            # İlerleme kapısı (shield yokken)
            if (vehicle.capacity > 0) and (len(env.demands) > 0) and (not use_shield) and (action is not None):
                cur_near  = env._nearest_demand_dist(vehicle.current_location)
                cand_near = env._nearest_demand_dist(action)
                if cand_near >= cur_near - PROGRESS_TOL and env.steps_since_delivery >= SHIELD_KICKIN_STEPS//2:
                    action_index = int(slots[0])
                    action = int(env.action_list[vehicle.current_location][action_index])

            cap_before = vehicle.capacity
            next_state, reward, done, done_vehicle = env.step(vehicle, action, action_index)
            total_points += reward

            if len(vehicle.visited_locations) >= 3:
                if vehicle.visited_locations[-1] == vehicle.visited_locations[-3]:
                    backtracks_test += 1
            if vehicle.capacity < cap_before:
                deliveries_done += 1

            if done_vehicle:
                print(f"[TEST] veh={vid} terminal reason={env.last_term_reason}, "
                      f"loc={vehicle.current_location}, steps_since_delivery={env.steps_since_delivery}")
                break

        if done_vehicle:
            continue

    # Özet çıktı
    print("\n=== SCENARIO RESULT ===")
    print(f"Total shaped reward: {total_points:.3f} | deliveries={deliveries_done} | backtracks={backtracks_test}")
    print("All vehicles at depot? ", all(v.current_location == init_location for v in vehicles))
    for vi, v in enumerate(vehicles):
        delivered_vi = v.initial_capacity - v.capacity
        print(f"  veh {vi}: end_loc={v.current_location}, "
              f"cap_left={v.capacity}, delivered={delivered_vi}, distance={v.total_distance:.2f}")
        print(f"  route: {v.visited_locations}")

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
            "route": v.visited_locations
        } for vi, v in enumerate(vehicles)]
    }

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    # Donanım bilgisi (bilgi amaçlı)
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
