import time
from collections import deque, namedtuple
import heapq
import random
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
import pandas as pd

import myutils as utils

from keras.layers import Dense, Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam

# ----------------------------
# Donanım bilgisi
# ----------------------------
if tf.config.list_physical_devices('GPU'):
    print("GPU kullanılabilir.")
else:
    print("GPU kullanılabilir değil, CPU kullanılıyor.")

# ============================================================
# Global config (sabitler)
# ============================================================
MAX_ACTIONS = 8

STALL_LIMIT = 400
STALL_PENALTY = -20.0
BACKTRACK_PENALTY = -0.30

NO_PROGRESS_PENALTY    = -0.40
RECENT_REVISIT_PENALTY = -0.20
RECENT_WINDOW          = 8
C_STEP                 = 0.02

LAMBDA_PBRS        = 2.5
ALPHA_NEAR         = 1.2
ALPHA_HOME         = 2.5
UNREACHABLE_NEAR_MULT = 50.0
ALPHA_HOME_ESCAPE  = 6.0

ILLEGAL_PENALTY = -1.0
DELIVERY_BONUS  =  1.2
FINAL_BONUS     =  7.5

CONNECTIVITY_CHECK_EVERY_EPISODE = True
CONNECTIVITY_CHECK_PERIOD = 1
RECOMPUTE_SP_IF_GRAPH_CHANGED = True

TABU_RECENT = 16
LOOP_WINDOW = 32
LOOP_UNIQUE_MAX = 7
NP_STREAK_THRESHOLD = 15

SHIELD_ACTIVE = True
SHIELD_KICKIN_STEPS = 20
# PROGRESS_TOL sabit yerine dinamik tolerans kullanacağız (aşağıda step() içinde)

# ---- Random DEPOT (başlangıç düğümü) ayarları ----
RANDOM_DEPOT_ACTIVE = True        # her epizotta depoyu rastgele seç
DEPOT_DEGREE_MIN = 3              # min 3 çıkış kenarı
DEPOT_CANDIDATES = None           # örn: [0, 10, 25, 316]
DEPOT_WEIGHTED_BY_DEGREE = True   # derece ağırlıklı örnekleme

# ---------- Curriculum + Öncelikli node örnekleme ----------
CURRICULUM_ACTIVE = True
CURRICULUM_DEMANDS_RANGE = (6, 10)  # epizotta talep sayısı min..max
PRIORITY_NODES = [316, 464, 607, 748]
PRIORITY_MIN   = 2
DEMAND_QTY_PER_NODE = 1

# ---------- Shield annealing ----------
SHIELD_ANNEAL = True
SHIELD_ANNEAL_END_EP = 5000
SHIELD_ANNEAL_MIN_FRAC = 0.20  # eğitim sonunda shield kullanma olasılık hedefi

# ---------- Eğitimde top‑K sırasını ara sıra karıştır ----------
TRAIN_TOPK_SHUFFLE = True
TOPK_SHUFFLE_PROB  = 0.20

# ---------- Kritik koruma modu ----------
PROTECT_CRITICAL_MODE  = 'min_degree'   # 'all' | 'min_degree'
MIN_DEGREE_PER_CRIT    = 1
PROTECT_CRITICAL       = True

# ---------- RISK: hedef kapama bandı + arama ayarları ----------
RISK_TARGET_OPEN_FRAC   = (0.85, 0.95)   # %5–%15 kapama (open_ratio=85–95%)
RISK_TRIES_PRIMARY      = 8              # dar r aralığında deneme sayısı
RISK_TRIES_FALLBACK     = 8              # geniş r aralığında deneme sayısı
RISK_R_PRIMARY_RANGE    = (0.42, 0.48)   # önce denenir
RISK_R_FALLBACK_RANGE   = (0.30, 0.60)   # sonra denenir
REOPEN_ON_DEMAND        = True
REOPEN_LIMIT_PER_TARGET = 30             # << 10 → 30 (daha agresif bağlanırlık)
FORCE_SYMMETRY          = True

# ---------- Ulaşılamayan hedef politikası ----------
# 'skip_learn' : epizodu oynat ama replay/öğrenmeyi kapat
# 'resample'   : risk örneklemeyi yeniden dene (UNREACHABLE_RESAMPLE_MAX kadar)
# 'allow'      : uyarı ver, normal eğitime devam et
UNREACHABLE_POLICY       = 'allow'
UNREACHABLE_RESAMPLE_MAX = 2

# ---------- Epsilon azaltım modu ----------
# Varsayılan step-bazlı azaltımı koruyoruz; isterseniz 'episode' yapabilirsiniz.
EPSILON_DECAY_MODE = 'episode'   # 'step' | 'episode'
E_DECAY_EPISODE    = 0.999    # episode bazlı modda kullanılacak çarpan

# ---------- Reproducibility ----------
SEED = None
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ============================
# Grafik yardımcıları
# ============================
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

# Nihai kapalı kenar sayısını (undirected) say
def count_closed_undirected(dm_before, dm_after, symmetric=True):
    mask_before = (np.array(dm_before) > 0)
    mask_after  = (np.array(dm_after)  > 0)
    eff_closed  = mask_before & (~mask_after)
    if symmetric:
        return int(np.sum(np.triu(np.logical_or(eff_closed, eff_closed.T), k=1)))
    else:
        return int(np.sum(eff_closed))

# ============================
# Double‑DQN kaybı (illegal mask + Huber)
# ============================
def compute_loss(experiences, gamma, q_network, target_q_network):
    states, actions, rewards, next_states, terminals = experiences

    q_next_target = target_q_network(next_states)  # (B, NUM_ACTIONS)
    next_loc_idx = tf.argmax(next_states[:, :N_LOC], axis=1, output_type=tf.int32)
    valid_counts = tf.gather(ACTION_COUNTS, next_loc_idx)

    maxlen_actions = tf.shape(q_next_target)[1]
    mask = tf.sequence_mask(valid_counts, maxlen=maxlen_actions, dtype=tf.float32)
    q_next_target_masked = q_next_target + (1.0 - mask) * (-1e9)

    q_next_online = q_network(next_states)
    q_next_online_masked = q_next_online + (1.0 - mask) * (-1e9)
    next_best_actions = tf.argmax(q_next_online_masked, axis=1, output_type=tf.int32)
    idx_next = tf.stack([tf.range(tf.shape(q_next_target)[0]), next_best_actions], axis=1)
    max_qsa = tf.gather_nd(q_next_target_masked, idx_next)

    y_targets = rewards + gamma * max_qsa * (1.0 - terminals)

    q_pred_all = q_network(states)
    idx_pred = tf.stack([tf.range(tf.shape(q_pred_all)[0]), tf.cast(actions, tf.int32)], axis=1)
    q_pred = tf.gather_nd(q_pred_all, idx_pred)

    loss_fn = tf.keras.losses.Huber(delta=1.0)
    loss = tf.reduce_mean(loss_fn(y_targets, q_pred))
    return loss

@tf.function
def agent_learn(experiences, gamma):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)
    grads = tape.gradient(loss, q_network.trainable_variables)
    grads = [tf.clip_by_norm(g, 10.0) for g in grads]
    optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
    return loss

# ============================
# Ortam
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

        # --- Dinamik ilerleme toleransı (ölçekli) ---
        tol = max(1e-6, 1e-3 * self.d_scale)

        if len(self.demands) > 0:
            near_next = self._nearest_demand_dist(vehicle.current_location)
            if (not delivered_now) and (near_next >= near_prev - tol):
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
        self.total_distance = 0
        self.total_duration = 0
        self.total_fuel = 0.0
        self.visited_location_time_window = [0]
        self.visited_time = [0.0]
        self.current_time_window = 0
        self.visited_locations = [init_location]
        self.visited_time_window = [0]
        self.done_vehicle = False

    def reset(self):
        self.current_location = init_location
        self.total_distance = 0
        self.total_duration = 0
        self.total_fuel = 0.0
        self.visited_location_time_window = [0]
        self.visited_time = [0.0]
        self.visited_locations = [init_location]
        self.visited_time_window = [0]
        self.current_time_window = 0
        self.done_vehicle = False
        self.capacity = self.initial_capacity


# ============================
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

    # 3) Kritik düğümleri koru (tam koruma veya min-degree)
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

    # 4) Nihai kapalı kenarları (koruma sonrası) say (geçici)
    eff_closed = (dm > 0) & (new_dm <= 0)
    if symmetric:
        closed_count = int(np.sum(np.triu(np.logical_or(eff_closed, eff_closed.T), k=1)))
    else:
        closed_count = int(np.sum(eff_closed))

    return new_dm, closed_count


# ============================
# Bağlanırlığı zorla: orijinal SP üzerindeki kapalı kenarları aç
# ============================
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

# ============================
# Veri yükle (sağlamlaştırılmış)
# ============================
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

# Mesafe ve risk
dst = load_square_matrix_xlsx('kcekmece_distance.xlsx', sheet_name='Sheet1')
risk = load_square_matrix_xlsx('kcekmece_risk_matris.xlsx', sheet_name='Sheet1')
assert dst.shape == risk.shape, "Mesafe ve risk matrisleri aynı boyutta olmalı!"

dst_initial = np.array(dst, dtype=np.float32)

# ============================
# Problem tanımı
# ============================
action_list, padded_action_list = action_matrix(dst, MAX_ACTIONS)
n_locations = len(dst)
N_LOC = n_locations
init_location = 0

# (Başlangıç) Talepler — curriculum episodik olarak güncellenecek
demands = [[316, 1], [464, 1], [607, 1], [748, 1]]

SP_DIST = dijkstra_all_sources(dst)

_arr = np.array(dst, dtype=np.float32)
_pos = _arr[_arr > 0]
D_SCALE = float(np.percentile(_pos, 95)) if _pos.size > 0 else 1.0

# ============================
# Hyperparametreler
# ============================
MEMORY_SIZE = 1_000_000
GAMMA = 0.995
ALPHA = 1e-4
NUM_STEPS_FOR_UPDATE = 2

epsilon = 1.0
E_MIN = 0.01
E_DECAY_STEP = 0.9998   # step-bazlı modda kullanılır

save_models_period = 20
test_period = 20

# ============================
# Ortam / araç
# ============================
n_vehicles = 3
vehicle_capacities = [2, 1, 1]  # curriculum ile epizot başında güncellenecek

env = Environment(dst, demands, n_vehicles, vehicle_capacities, n_locations, SP_DIST)
vehicles = [Vehicle(i, vehicle_capacities[i], env) for i in range(n_vehicles)]
env.attach_vehicles(vehicles)
env.gamma_rl = GAMMA
env.reset()

# ============================
# Ağlar — Dueling‑DQN
# ============================
state_size = 2 * n_locations + 1
num_actions = MAX_ACTIONS

print('State Shape:', state_size)
print('Number of actions:', num_actions)

def build_dueling_dqn(state_size, num_actions):
    inputs = Input(shape=(state_size,))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(1024, activation='relu')(x)
    V = Dense(1, activation='linear', name='V')(x)
    A = Dense(num_actions, activation='linear', name='A')(x)
    Q = Lambda(lambda va: va[0] + (va[1] - tf.reduce_mean(va[1], axis=1, keepdims=True)))([V, A])
    return Model(inputs=inputs, outputs=Q)

q_network = build_dueling_dqn(state_size, num_actions)
target_q_network = build_dueling_dqn(state_size, num_actions)

NUM_ACTIONS = int(q_network.output_shape[-1])
assert NUM_ACTIONS == int(target_q_network.output_shape[-1]), "q ve target çıktı sayıları farklı!"

ACTION_COUNTS = tf.Variable(
    [min(len(a), NUM_ACTIONS) for a in action_list], dtype=tf.int32, trainable=False
)

print("NUM_ACTIONS (from model):", NUM_ACTIONS)
optimizer = Adam(ALPHA)

# ============================
# Eğitim hazırlık
# ============================
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

start = time.time()

num_episodes = 100001
max_num_timesteps = 50_000

total_rewards = []
total_rewards_test = []

memory_buffer = deque(maxlen=MEMORY_SIZE)

# --- SOFT TARGET (Polyak/EMA) ---
TAU_SOFT = 0.005
def soft_update(target_net, source_net, tau=TAU_SOFT):
    tw = target_net.get_weights()
    sw = source_net.get_weights()
    target_net.set_weights([(1.0 - tau) * t + tau * s for t, s in zip(tw, sw)])

target_q_network.set_weights(q_network.get_weights())

NONZERO_MASK = (_arr > 0).astype(np.uint8)  # başlangıç (yönlü) nonzero mask

# ---------- yardımcılar (curriculum/priority + kapasite) ----------
def split_sum_positive(total, parts):
    if parts <= 0:
        return []
    base = [total // parts] * parts
    for i in range(total % parts):
        base[i] += 1
    return base

def choose_demands_with_priority(n_loc, depot, k, priority_nodes, priority_min):
    pr = [p for p in priority_nodes if 0 <= p < n_loc and p != depot]
    rest = [i for i in range(n_loc) if i != depot and i not in pr]
    k_pr = max(0, min(k, min(priority_min, len(pr))))
    chosen_pr = random.sample(pr, k_pr) if k_pr > 0 else []
    remaining = max(0, k - len(chosen_pr))
    chosen_rest = random.sample(rest, remaining) if remaining > 0 else []
    chosen = chosen_pr + chosen_rest
    return [[int(n), int(DEMAND_QTY_PER_NODE)] for n in chosen]

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

# ============================
# === CHECKPOINT / RESUME KURULUMU ===
# ============================
CKPT_DIR = r"cit_ckpt"
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
BUFFER_PKL = os.path.join(CKPT_DIR, "replay_buffer.pkl")
RNG_PKL    = os.path.join(CKPT_DIR, "rng.pkl")

# Episode ve epsilon'u checkpoint'e dahil edebilmek için değişkenleştiriyoruz
episode_var = tf.Variable(0, dtype=tf.int64, name="episode_var")
epsilon_var = tf.Variable(float(epsilon), dtype=tf.float32, name="epsilon_var")
global_step = tf.Variable(0, dtype=tf.int64, name="global_step")

ckpt = tf.train.Checkpoint(
    q=q_network,
    target=target_q_network,
    opt=optimizer,
    episode=episode_var,
    epsilon=epsilon_var,
    gstep=global_step
)
manager = tf.train.CheckpointManager(ckpt, CKPT_DIR, max_to_keep=5)

# === VARSA CHECKPOINT'TEN DEVAM ET ===
start_episode = 0
if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    start_episode = int(episode_var.numpy()) + 1  # kaldığı ep + 1
    epsilon = float(epsilon_var.numpy())

    # Replay buffer'ı yükle (varsa)
    if os.path.exists(BUFFER_PKL):
        try:
            with open(BUFFER_PKL, "rb") as f:
                memory_buffer = pickle.load(f)
            print(f"[RESUME] Replay buffer yüklendi: {len(memory_buffer)} kayıt")
        except Exception as e:
            print("[RESUME] Replay buffer yüklenemedi:", e)

    # RNG durumlarını yükle (opsiyonel)
    if os.path.exists(RNG_PKL):
        try:
            with open(RNG_PKL, "rb") as f:
                rng = pickle.load(f)
            random.setstate(rng.get("py", random.getstate()))
            np.random.set_state(rng.get("np", np.random.get_state()))
        except Exception as e:
            print("[RESUME] RNG durumu yüklenemedi:", e)

    print(f"[RESUME] {manager.latest_checkpoint} yüklendi | start_episode={start_episode} | epsilon={epsilon:.4f}")
else:
    # (Opsiyonel) Eski .h5 modeliniz varsa sıcak başlangıç yapın
    WARMSTART_H5 = r"cit_q_network_kcekmece.h5"
    if os.path.exists(WARMSTART_H5):
        try:
            full = load_model(WARMSTART_H5, compile=False)
            q_network.set_weights(full.get_weights())
            target_q_network.set_weights(q_network.get_weights())
            print("[RESUME-MIN] H5 model ağırlıkları yüklendi ve target senkronlandı.")
        except Exception as e:
            print("[RESUME-MIN] H5 yüklenemedi:", e)
    print("[RESUME] Checkpoint bulunamadı; eğitim sıfırdan başlayacak.")

# ============================
# Eğitim döngüsü
# ============================
for episode in range(start_episode, num_episodes):

    # --- RANDOM DEPOT (per-episode) ---
    if RANDOM_DEPOT_ACTIVE:
        init_location = pick_random_depot(
            dst_initial,
            degree_min=DEPOT_DEGREE_MIN,
            candidates=DEPOT_CANDIDATES,
            weighted=DEPOT_WEIGHTED_BY_DEGREE
        )
        print(f"[DEPOT] ep={episode} | init_location={init_location}")

    # ---------- Curriculum / priority demands + kapasite eşleme ----------
    if CURRICULUM_ACTIVE:
        k_demands = random.randint(CURRICULUM_DEMANDS_RANGE[0], CURRICULUM_DEMANDS_RANGE[1])
        ep_demands = choose_demands_with_priority(n_locations, init_location, k_demands, PRIORITY_NODES, PRIORITY_MIN)
        env.demands_init = [(int(n), int(a)) for n, a in ep_demands]
        env.demands = {int(n): int(a) for n, a in ep_demands}
        env.initial_total_demand = int(sum(a for _, a in ep_demands))

        caps = split_sum_positive(env.initial_total_demand, len(vehicles))
        for i, v in enumerate(vehicles):
            v.initial_capacity = int(caps[i])
        env.vehicle_capacities = [int(c) for c in caps]

        print(f"[CURR] ep={episode} | demands={sorted([n for n,_ in ep_demands])} "
              f"| total_demand={env.initial_total_demand} | capacities={env.vehicle_capacities}")

    # ---- Risk fit + (opsiyonel) unreachable resample döngüsü ----
    skip_learning = False
    resample_rounds = 0
    while True:
        total_edges_undir = int(np.sum(np.triu((dst_initial > 0), k=1)))  # tekil (i<j) kenar
        accepted = False
        tries = 0
        dst_risk = np.array(dst_initial, dtype=np.float32)  # default
        r_used = None
        closed_after = 0

        # Önce dar aralık
        while (not accepted) and (tries < RISK_TRIES_PRIMARY):
            r_used = float(np.random.uniform(RISK_R_PRIMARY_RANGE[0], RISK_R_PRIMARY_RANGE[1]))
            critical_nodes = [init_location] + sorted(env.demands.keys()) if PROTECT_CRITICAL else None
            dm_tmp, closed_before = apply_risk_to_distance_safe(
                dst_initial, risk, r_used,
                critical_nodes=critical_nodes,
                symmetric=FORCE_SYMMETRY,
                protect_mode=PROTECT_CRITICAL_MODE,
                min_degree_per_critical=MIN_DEGREE_PER_CRIT
            )
            if REOPEN_ON_DEMAND and len(env.demands) > 0:
                dm_final = ensure_connectivity(dm_tmp, dst_initial, init_location, env.demands.keys(),
                                               reopen_limit_per_target=REOPEN_LIMIT_PER_TARGET)
            else:
                dm_final = dm_tmp

            closed_after_local = count_closed_undirected(dst_initial, dm_final, symmetric=FORCE_SYMMETRY)
            reopened = closed_before - closed_after_local
            open_ratio = 1.0 - (closed_after_local / max(1, total_edges_undir))
            print(f"[RISK@try{tries}] r={r_used:.4f} | closed_before={closed_before} | "
                  f"closed_after={closed_after_local} | reopened={reopened} | "
                  f"open_ratio={open_ratio*100:.1f}% (primary)")
            if (RISK_TARGET_OPEN_FRAC[0] <= open_ratio <= RISK_TARGET_OPEN_FRAC[1]):
                dst_risk = dm_final
                closed_after = closed_after_local
                accepted = True
            tries += 1

        # Geniş aralık
        while (not accepted) and (tries < (RISK_TRIES_PRIMARY + RISK_TRIES_FALLBACK)):
            r_used = float(np.random.uniform(RISK_R_FALLBACK_RANGE[0], RISK_R_FALLBACK_RANGE[1]))
            critical_nodes = [init_location] + sorted(env.demands.keys()) if PROTECT_CRITICAL else None
            dm_tmp, closed_before = apply_risk_to_distance_safe(
                dst_initial, risk, r_used,
                critical_nodes=critical_nodes,
                symmetric=FORCE_SYMMETRY,
                protect_mode=PROTECT_CRITICAL_MODE,
                min_degree_per_critical=MIN_DEGREE_PER_CRIT
            )
            if REOPEN_ON_DEMAND and len(env.demands) > 0:
                dm_final = ensure_connectivity(dm_tmp, dst_initial, init_location, env.demands.keys(),
                                               reopen_limit_per_target=REOPEN_LIMIT_PER_TARGET)
            else:
                dm_final = dm_tmp

            closed_after_local = count_closed_undirected(dst_initial, dm_final, symmetric=FORCE_SYMMETRY)
            reopened = closed_before - closed_after_local
            open_ratio = 1.0 - (closed_after_local / max(1, total_edges_undir))
            print(f"[RISK@try{tries}] r={r_used:.4f} | closed_before={closed_before} | "
                  f"closed_after={closed_after_local} | reopened={reopened} | "
                  f"open_ratio={open_ratio*100:.1f}% (fallback)")
            if (RISK_TARGET_OPEN_FRAC[0] <= open_ratio <= RISK_TARGET_OPEN_FRAC[1]):
                dst_risk = dm_final
                closed_after = closed_after_local
                accepted = True
            tries += 1

        # Kabul edildiyse grafiği ve SP'yi yaz
        if accepted:
            arr_now = np.array(dst_risk)
            mask_now = (arr_now > 0).astype(np.uint8)
            if RECOMPUTE_SP_IF_GRAPH_CHANGED and mask_now.shape == NONZERO_MASK.shape and np.any(mask_now != NONZERO_MASK):
                changed_edges = int(np.sum(mask_now != NONZERO_MASK))
                print(f"[Graph change] nonzero edges changed: {changed_edges}. Recompute SP ...")
                SP_DIST = dijkstra_all_sources(dst_risk)
                env.sp_dist = SP_DIST
                action_list, padded_action_list = action_matrix(dst_risk, MAX_ACTIONS)
                env.action_list, env.padded_action_list = action_list, padded_action_list
                ACTION_COUNTS.assign([min(len(a), NUM_ACTIONS) for a in action_list])
                NONZERO_MASK = mask_now

            env.dst = dst_risk
            env.reset()
            for vehicle in vehicles:
                vehicle.reset()

            unreachable = []
            if CONNECTIVITY_CHECK_EVERY_EPISODE and (episode % CONNECTIVITY_CHECK_PERIOD == 0):
                unreachable = check_connectivity_and_routes(env.dst, init_location, sorted(env.demands.keys()))

            final_open_ratio = 1.0 - (closed_after / max(1, total_edges_undir))
            print(f"[RISK] ACCEPTED | r={r_used:.4f} | closed_after={closed_after} | "
                  f"open_ratio={final_open_ratio*100:.1f}%")

            # --- Ulaşılamayan hedef politikası ---
            if len(unreachable) > 0:
                if UNREACHABLE_POLICY == 'resample' and resample_rounds < UNREACHABLE_RESAMPLE_MAX:
                    resample_rounds += 1
                    print(f"[RISK] Unreachable after acceptance → RESAMPLE (round {resample_rounds}/{UNREACHABLE_RESAMPLE_MAX})")
                    continue  # risk fit bloğunu tekrar çalıştır
                elif UNREACHABLE_POLICY == 'skip_learn':
                    skip_learning = True
                    print("[RISK] Unreachable after acceptance → SKIP LEARNING for this episode.")
                else:
                    print("[RISK] Unreachable after acceptance → ALLOW (training continues).")

        else:
            # Hâlâ kabul yoksa: base graph
            env.dst = np.array(dst_initial, dtype=np.float32)
            SP_DIST = dijkstra_all_sources(env.dst)
            env.sp_dist = SP_DIST
            action_list, padded_action_list = action_matrix(env.dst, MAX_ACTIONS)
            env.action_list, env.padded_action_list = action_list, padded_action_list
            ACTION_COUNTS.assign([min(len(a), NUM_ACTIONS) for a in action_list])

            env.reset()
            for vehicle in vehicles:
                vehicle.reset()
            print(f"[RISK] Fallback to base graph (no target-fit in {tries} tries). "
                  f"closed_after=0 | open_ratio=100.0%")

        break  # risk/resample döngüsünden çık

    total_points = 0.0
    done = False
    deliveries_done = 0
    backtracks = 0
    train_paths = [[] for _ in vehicles]

    # ---------- Shield anneal oranı (epizot bazlı) ----------
    if SHIELD_ANNEAL:
        prog = min(1.0, episode / max(1, SHIELD_ANNEAL_END_EP))
        shield_frac = 1.0 - (1.0 - SHIELD_ANNEAL_MIN_FRAC) * prog
    else:
        shield_frac = 1.0

    for vehicle_number, vehicle in enumerate(vehicles):
        done_vehicle = False
        env.steps_since_delivery = 0
        env.no_progress_streak = 0

        for t in range(max_num_timesteps):

            state = env.get_state_from_external(vehicle)
            if vehicle.capacity <= 0 and vehicle.current_location == init_location:
                env.last_term_reason = "empty_home"
                train_paths[vehicle_number] = vehicle.visited_locations[:]
                print(f"Terminal (k={episode}, i={t}) reason=empty_home(idle), "
                      f"loc={vehicle.current_location}, steps_since_delivery={env.steps_since_delivery}")
                break

            history = vehicle.visited_locations
            prev_prev = history[-2] if len(history) >= 2 else None
            tabu_nodes = set(history[-TABU_RECENT:])

            # --- annealed shield kapısı ---
            shield_gate = (random.random() < shield_frac)

            use_shield = SHIELD_ACTIVE and shield_gate and (
                env.steps_since_delivery >= SHIELD_KICKIN_STEPS or
                env.no_progress_streak   >= NP_STREAK_THRESHOLD or
                env.is_looping(history)
            )

            # Kapasite 0 / talepler bitti: eve SP‑hop
            if (vehicle.capacity <= 0):
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

            # Top‑K karıştır (eğitimde)
            if TRAIN_TOPK_SHUFFLE and n_valid > 1 and random.random() < TOPK_SHUFFLE_PROB:
                if force_first is not None:
                    head = [slots[0]]
                    tail = list(slots[1:])
                    random.shuffle(tail)
                    slots = head + tail
                else:
                    slots = list(slots)
                    random.shuffle(slots)

            # Aksiyon seçimi
            if n_valid > 0:
                if (vehicle.capacity <= 0) or (len(env.demands) == 0):
                    action_index = int(slots[0])
                elif use_shield:
                    action_index = int(slots[0])
                else:
                    q_vals = q_network(state.reshape(1, -1)).numpy()[0]
                    if random.random() > epsilon:
                        action_index = int(slots[int(np.argmax(q_vals[slots]))])
                    else:
                        action_index = int(random.choice(slots))
                action = int(env.action_list[vehicle.current_location][action_index])
            else:
                action, action_index = None, None

            # İlerleme kapısı
            if (vehicle.capacity > 0) and (len(env.demands) > 0) and (not use_shield) and (action is not None):
                cur_near = env._nearest_demand_dist(vehicle.current_location)
                cand_near = env._nearest_demand_dist(action)
                # dinamik tol
                tol = max(1e-6, 1e-3 * env.d_scale)
                if cand_near >= cur_near - tol and env.steps_since_delivery >= SHIELD_KICKIN_STEPS//2:
                    action_index = int(slots[0])
                    action = int(env.action_list[vehicle.current_location][action_index])

            cap_before = vehicle.capacity
            next_state, reward, done, done_vehicle = env.step(vehicle, action, action_index)

            # Replay’e yaz (skip_learning ise HİÇ yazma)
            if (not skip_learning) and (action is not None) and (action_index is not None):
                terminal = float(done or done_vehicle)  # << kritik düzeltme
                memory_buffer.append(experience(
                    state,
                    action_index,
                    reward,
                    next_state,
                    terminal
                ))

            # Güncelleme (skip_learning ise öğrenme yok)
            update = (not skip_learning) and utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
            if update:
                experiences = utils.get_experiences(memory_buffer)
                _ = agent_learn(experiences, GAMMA)
                # Soft target update
                soft_update(target_q_network, q_network, TAU_SOFT)

            total_points += reward
            train_paths[vehicle_number] = vehicle.visited_locations[:]

            # Epsilon decay
            if EPSILON_DECAY_MODE == 'step' and epsilon > E_MIN:
                epsilon = max(E_MIN, epsilon * E_DECAY_STEP)

            # Sayaçlar
            if len(vehicle.visited_locations) >= 3:
                if vehicle.visited_locations[-1] == vehicle.visited_locations[-3]:
                    backtracks += 1
            if vehicle.capacity < cap_before:
                deliveries_done += 1

            if done_vehicle:
                print(f"Terminal (k={episode}, i={t}) reason={env.last_term_reason}, "
                      f"loc={vehicle.current_location}, steps_since_delivery={env.steps_since_delivery}")
                break

        if done:
            if env.last_term_reason == "solved":
                print("Training SOLVED")
            else:
                print("Training TERMINATED (stall)")
                epsilon = max(0.10, epsilon)
            break

    # Episode-bazlı epsilon azaltımı (opsiyonel mod)
    if EPSILON_DECAY_MODE == 'episode':
        epsilon = max(E_MIN, epsilon * E_DECAY_EPISODE)

    # Eğitim çıktıları
    total_rewards.append(total_points)
    total_path_length = sum(len(v.visited_locations) for v in vehicles)
    print(f"Episode {episode} | Training total reward: {total_points:.3f} | eps={epsilon:.4f} "
          f"| deliveries={deliveries_done} | backtracks={backtracks} | skip_learning={skip_learning}")
    print("Training time_steps:", total_path_length)
    print("All vehicles at depot? ", all(v.current_location == init_location for v in vehicles))
    for vi, v in enumerate(vehicles):
        delivered_vi = v.initial_capacity - v.capacity
        print(f"  veh {vi}: loc={v.current_location}, cap_left={v.capacity}, delivered={delivered_vi}")

    # ---------------- TEST (her 20 epizotta) ----------------
    if episode % test_period == 0:
        env.reset()
        for vehicle in vehicles:
            vehicle.reset()
        total_points_TrainedNetwork = 0.0
        max_timesteps_test = 500

        test_paths = [[] for _ in vehicles]
        test_time_steps = 0

        deliveries_done_test = 0
        backtracks_test = 0

        for vehicle_number, vehicle in enumerate(vehicles):
            done_vehicle = False
            env.steps_since_delivery = 0
            env.no_progress_streak = 0

            for t in range(max_timesteps_test):

                state = env.get_state_from_external(vehicle)
                # --- HARD FREEZE (TEST): zero-capacity vehicle at depot -> no movement
                if vehicle.capacity <= 0 and vehicle.current_location == init_location:
                    env.last_term_reason = "empty_home"
                    test_paths[vehicle_number] = vehicle.visited_locations[:]
                    print(f"[TEST] Terminal (k={episode}, i={t}) reason=empty_home(idle), "
                          f"loc={vehicle.current_location}, steps_since_delivery={env.steps_since_delivery}")
                    break

                history = vehicle.visited_locations
                prev_prev = history[-2] if len(history) >= 2 else None
                tabu_nodes = set(history[-TABU_RECENT:])
                use_shield = SHIELD_ACTIVE and (
                    env.steps_since_delivery >= SHIELD_KICKIN_STEPS or
                    env.no_progress_streak   >= NP_STREAK_THRESHOLD or
                    env.is_looping(history)
                )

                if (vehicle.capacity <= 0):
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

                if n_valid > 0:
                    if (vehicle.capacity <= 0) or (len(env.demands) == 0):
                        action_index = int(slots[0])
                    elif use_shield:
                        action_index = int(slots[0])
                    else:
                        q_vals = q_network(state.reshape(1, -1)).numpy()[0]
                        best_slot = slots[int(np.argmax(q_vals[slots]))]
                        action_index = int(best_slot)
                    action = int(env.action_list[vehicle.current_location][action_index])
                else:
                    action, action_index = None, None

                if (vehicle.capacity > 0) and (len(env.demands) > 0) and (not use_shield) and (action is not None):
                    cur_near = env._nearest_demand_dist(vehicle.current_location)
                    cand_near = env._nearest_demand_dist(action)
                    tol = max(1e-6, 1e-3 * env.d_scale)
                    if cand_near >= cur_near - tol and env.steps_since_delivery >= SHIELD_KICKIN_STEPS//2:
                        action_index = int(slots[0])
                        action = int(env.action_list[vehicle.current_location][action_index])

                cap_before = vehicle.capacity
                next_state, reward, done, done_vehicle = env.step(vehicle, action, action_index)
                total_points_TrainedNetwork += reward

                if len(vehicle.visited_locations) >= 3:
                    if vehicle.visited_locations[-1] == vehicle.visited_locations[-3]:
                        backtracks_test += 1
                if vehicle.capacity < cap_before:
                    deliveries_done_test += 1

                test_paths[vehicle_number] = vehicle.visited_locations[:]

                if done_vehicle:
                    print(f"[TEST] Terminal (k={episode}, i={t}) reason={env.last_term_reason}, "
                          f"loc={vehicle.current_location}, steps_since_delivery={env.steps_since_delivery}")
                    break

            test_time_steps += len(vehicle.visited_locations)

        total_rewards_test.append(total_points_TrainedNetwork)

        print("TEST RESULT")
        print(f"Episode {episode} | Test total reward: {total_points_TrainedNetwork:.3f} "
              f"| deliveries={deliveries_done_test} | backtracks={backtracks_test}")
        print("Test time_steps:", test_time_steps)
        for vi, p in enumerate(test_paths):
            print(f"Test route (veh {vi}): {p}")
        print("All vehicles at depot? ", all(v.current_location == init_location for v in vehicles))
        for vi, v in enumerate(vehicles):
            delivered_vi = v.initial_capacity - v.capacity
            print(f"  [TEST] veh {vi}: loc={v.current_location}, cap_left={v.capacity}, delivered={delivered_vi}")

        if PROTECT_CRITICAL and PROTECT_CRITICAL_MODE == 'min_degree':
            crits = [init_location] + sorted(env.demands.keys())
            # env.dst son kabul edilen riskli grafiktir
            degs = {int(c): int(np.sum(env.dst[int(c), :] > 0)) for c in crits}
            print("[RISK] critical degrees after protection:", degs)

    # ---------------- KAYDET (Checkpoint + buffer + RNG) ----------------
    # epizot sonunda güncel değerleri checkpoint değişkenlerine yazın
    episode_var.assign(episode)
    epsilon_var.assign(float(epsilon))
    global_step.assign_add(1)

    if episode % save_models_period == 0 and episode > 0:
        # 1) TF checkpoint (ağlar + optimizer + sayaçlar)
        path = manager.save()
        print(f"[CKPT] Kaydedildi: {path}")

        # 2) Replay buffer'ı güvenli (atomic) yaz
        try:
            tmp = BUFFER_PKL + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(memory_buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, BUFFER_PKL)
        except Exception as e:
            print("[CKPT] Replay buffer kaydedilemedi:", e)

        # 3) RNG durumları (python + numpy)
        try:
            tmp_rng = RNG_PKL + ".tmp"
            with open(tmp_rng, "wb") as f:
                pickle.dump({"py": random.getstate(), "np": np.random.get_state()},
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_rng, RNG_PKL)
        except Exception as e:
            print("[CKPT] RNG durumu kaydedilemedi:", e)

        # (İsteğe bağlı) .h5 yedekleri ve ödül log’ları
        try:
            q_network.save(r"cit_q_network_kcekmece.h5")
        except Exception as e:
            print("[CKPT] .h5 model kaydedilemedi:", e)
        try:
            with open(r"cash_in_transit_rewards_train_5.txt", "w", encoding="utf-8") as fp:
                fp.write('\n'.join(str(item) for item in total_rewards))
            with open(r"cash_in_transit_rewards_test_5.txt", "w", encoding="utf-8") as fp:
                fp.write('\n'.join(str(item) for item in total_rewards_test))
        except Exception as e:
            print("[CKPT] ödül dosyaları kaydedilemedi:", e)

tot_time = time.time() - start
print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

