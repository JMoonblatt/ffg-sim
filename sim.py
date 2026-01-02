import json, random, math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# ----------------------------
# Utilities
# ----------------------------
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def gini(values: List[float]) -> float:
    # Gini coefficient for nonnegative values
    vals = [v for v in values if v >= 0]
    if not vals:
        return 0.0
    s = sum(vals)
    if s == 0:
        return 0.0
    vals = sorted(vals)
    n = len(vals)
    cum = 0.0
    for i, v in enumerate(vals, start=1):
        cum += i * v
    return (2 * cum) / (n * s) - (n + 1) / n

def weighted_choice(items: List[int], weights: List[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if r <= acc:
            return item
    return items[-1]

# ----------------------------
# Model
# ----------------------------
@dataclass
class Agent:
    id: int
    motivation: str  # selfish | neutral | prosocial
    competence: float
    availability: float
    visibility_bias: float
    fatigue: float = 0.0
    responsibility_load: float = 0.0
    active: bool = True

    def utility_weights(self) -> Tuple[float, float, float]:
        """
        Returns weights (w_group, w_self_cost, w_overhead_aversion).
        """
        if self.motivation == "selfish":
            return (0.25, 1.00, 0.90)
        if self.motivation == "prosocial":
            return (1.00, 0.35, 0.45)
        # neutral
        return (0.60, 0.60, 0.65)

@dataclass
class Task:
    id: int
    complexity: int        # 1..10
    coordination: int      # number of agents needed
    ambiguity: float       # 0..1
    payoff: float          # group payoff proxy

@dataclass
class Fork:
    task_id: int
    remaining: int
    contenders: Tuple[str, str]  # ("A","B")

@dataclass
class SimResult:
    completed: int
    attempted: int
    failed: int
    exits: int
    avg_latency_proxy: float
    responsibility_gini: float
    steps_survived: int

# ----------------------------
# Governance regimes
# ----------------------------
class Regime:
    HERO = "hero"
    FFG = "ffg"

# ----------------------------
# Simulation
# ----------------------------
class Simulator:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.rng = random.Random(cfg["seed"])
        self.step = 0
        self.task_id = 0
        self.forks: List[Fork] = []

        self.agents: List[Agent] = self._init_agents()
        self.visible_agent_id: Optional[int] = None  # chosen later

        self.completed = 0
        self.attempted = 0
        self.failed = 0

        self.latency_samples: List[float] = []

    def _init_agents(self) -> List[Agent]:
        n = self.cfg["n_agents"]
        mix = self.cfg["motivation_mix"]
        types = []
        for k, p in mix.items():
            types += [k] * int(round(p * n))
        while len(types) < n:
            types.append("neutral")
        while len(types) > n:
            types.pop()
        self.rng.shuffle(types)

        agents = []
        for i in range(n):
            competence = clamp(self.rng.gauss(0.6, 0.15), 0.1, 0.95)
            availability = clamp(self.rng.gauss(0.75, 0.15), 0.2, 0.98)
            visibility_bias = clamp(self.rng.gauss(0.5, 0.25), 0.0, 1.0)
            agents.append(Agent(i, types[i], competence, availability, visibility_bias))

        # Optionally inject a toxic hero archetype (high competence, high visibility, selfish, high availability).
        if self.cfg.get("enable_toxic_hero", False) and agents:
            idx = self.rng.randrange(len(agents))
            agents[idx] = Agent(
                id=agents[idx].id,
                motivation="selfish",
                competence=self.cfg.get("toxic_hero_competence", 0.92),
                availability=self.cfg.get("toxic_hero_availability", 0.92),
                visibility_bias=self.cfg.get("toxic_hero_visibility", 0.98),
            )

        return agents

    def _alive_agents(self) -> List[Agent]:
        return [a for a in self.agents if a.active]

    def _choose_visible_agent(self):
        alive = self._alive_agents()
        if not alive:
            self.visible_agent_id = None
            return
        scores = [(a.visibility_bias * 0.6 + a.competence * 0.4, a.id) for a in alive]
        scores.sort(reverse=True)
        self.visible_agent_id = scores[0][1]

    def _fatigue_multiplier(self, a: Agent) -> float:
        # If toxic hero enabled and this agent matches the archetype, reduce fatigue accumulation.
        if self.cfg.get("enable_toxic_hero", False):
            if a.motivation == "selfish" and a.competence >= 0.9 and a.visibility_bias >= 0.9:
                return self.cfg.get("toxic_hero_burnout_resistance", 0.55)
        return 1.0

    def _shock_remove_top(self, k: int = 1):
        alive = self._alive_agents()
        if not alive:
            return
        topk = sorted(alive, key=lambda a: a.competence, reverse=True)[:k]
        for a in topk:
            a.active = False

    def _shock_attack_visible(self):
        if self.visible_agent_id is None:
            return
        a = self.agents[self.visible_agent_id]
        if a.active:
            a.availability = clamp(a.availability * self.cfg["attack_availability_multiplier"], 0.05, 0.98)

    def _random_dropout(self, count: int):
        alive = self._alive_agents()
        if len(alive) <= count:
            return
        weights = []
        ids = []
        for a in alive:
            w = 1.0
            if a.motivation == "selfish":
                w = 1.25
            elif a.motivation == "prosocial":
                w = 0.9
            ids.append(a.id)
            weights.append(w)
        for _ in range(count):
            pick_id = weighted_choice(ids, weights, self.rng)
            self.agents[pick_id].active = False
            j = ids.index(pick_id)
            ids.pop(j)
            weights.pop(j)

    def _spawn_task(self) -> Optional[Task]:
        if self.rng.random() > self.cfg["task_rate"]:
            return None
        self.task_id += 1
        c_lo, c_hi = self.cfg["task_complexity_range"]
        k_lo, k_hi = self.cfg["task_coordination_range"]
        a_lo, a_hi = self.cfg["task_ambiguity_range"]
        complexity = self.rng.randint(c_lo, c_hi)
        coordination = self.rng.randint(k_lo, k_hi)
        ambiguity = self.rng.random() * (a_hi - a_lo) + a_lo
        payoff = complexity * (1.0 + (1.0 - ambiguity) * 0.5)
        return Task(self.task_id, complexity, coordination, ambiguity, payoff)

    def _available_pool(self) -> List[Agent]:
        pool = []
        for a in self._alive_agents():
            if self.rng.random() < a.availability:
                pool.append(a)
        return pool

    def _hero_select_team(self, task: Task, pool: List[Agent]) -> List[Agent]:
        if len(pool) < task.coordination:
            return []
        scores = []
        for a in pool:
            s = (0.75 * a.competence + 0.25 * a.visibility_bias)
            scores.append((s, a))
        scores.sort(reverse=True, key=lambda x: x[0])

        hero = scores[0][1]
        team = [hero] + [a for _, a in scores[1:task.coordination]]
        return team

    def _ffg_select_team(self, task: Task, pool: List[Agent], steward_id: int) -> Tuple[List[Agent], float, bool]:
        if len(pool) < task.coordination:
            return ([], 0.0, False)

        overhead = self.cfg["ffg_overhead_base"] + self.cfg["ffg_overhead_ambiguity_scale"] * task.ambiguity
        latency_proxy = overhead * task.coordination

        base_disagree = task.ambiguity * 0.6
        committee = self.rng.sample(pool, k=min(task.coordination + 2, len(pool)))
        preference_spread = 0.0
        for a in committee:
            w_group, w_cost, w_ov = a.utility_weights()
            stance = w_group * task.payoff - w_ov * overhead - w_cost * (task.coordination * 0.08)
            preference_spread += stance
        preference_spread /= max(1, len(committee))

        contention = 1.0 - clamp(abs(preference_spread) / 5.0, 0.0, 1.0)
        p_fork = clamp(base_disagree * contention, 0.0, 1.0)

        forked = (p_fork > self.cfg["fork_threshold"])

        pool_ids = [a.id for a in pool]
        if steward_id in pool_ids:
            steward = self.agents[steward_id]
        else:
            steward = max(pool, key=lambda a: a.competence)

        scored = []
        for a in pool:
            fatigue_pen = a.fatigue * 0.35
            scored.append(((a.competence - fatigue_pen), a))
        scored.sort(reverse=True, key=lambda x: x[0])

        team = [steward]
        for _, a in scored:
            if a.id == steward.id:
                continue
            if len(team) >= task.coordination:
                break
            team.append(a)
        if len(team) < task.coordination:
            return ([], latency_proxy, forked)
        return (team, latency_proxy, forked)

    def _attempt_task(self, regime: str, task: Task, pool: List[Agent], steward_id: int) -> None:
        self.attempted += 1

        if regime == Regime.HERO:
            team = self._hero_select_team(task, pool)
            if not team:
                self.failed += 1
                return

            total_comp = sum(a.competence for a in team)
            p_success = clamp((total_comp / task.coordination) * 0.95 - task.ambiguity * 0.25, 0.05, 0.98)

            hero_load = self.cfg.get("hero_hero_load", 1.0)
            other_load = self.cfg.get("hero_other_load", 0.35)

            for i, a in enumerate(team):
                load = hero_load if i == 0 else other_load
                a.responsibility_load += load
                a.fatigue += (load * self.cfg["burnout_load_scale"]) * self._fatigue_multiplier(a)

            self.latency_samples.append(0.15 * task.coordination)

            if self.rng.random() < p_success:
                self.completed += 1
            else:
                self.failed += 1

        else:
            team, latency_proxy, forked = self._ffg_select_team(task, pool, steward_id)
            self.latency_samples.append(latency_proxy)

            if not team:
                self.failed += 1
                return

            split_penalty = 0.12 if forked else 0.0

            total_comp = sum(a.competence for a in team)
            p_success = clamp((total_comp / task.coordination) * 0.90 - task.ambiguity * 0.18 - split_penalty, 0.05, 0.98)

            steward = team[0]
            steward_load = self.cfg.get("ffg_steward_load", 0.55)
            member_load = self.cfg.get("ffg_member_load", 0.45)

            for a in team:
                load = steward_load if a.id == steward.id else member_load
                a.responsibility_load += load
                a.fatigue += (load * self.cfg["burnout_load_scale"]) * self._fatigue_multiplier(a)

            if forked:
                self.forks.append(Fork(task_id=task.id, remaining=self.cfg["fork_trial_length"], contenders=("A", "B")))

            if self.rng.random() < p_success:
                self.completed += 1
            else:
                self.failed += 1

    def _resolve_forks(self):
        kept = []
        for f in self.forks:
            f.remaining -= 1
            if f.remaining <= 0:
                if self.rng.random() < self.cfg.get("merge_success_prob", 0.65):
                    self.completed += 1
                else:
                    self.failed += 1
            else:
                kept.append(f)
        self.forks = kept

    def _apply_exits(self):
        exits = 0
        thr = self.cfg["exit_fatigue_threshold"]
        for a in self._alive_agents():
            if a.fatigue >= thr:
                a.active = False
                exits += 1
        return exits

    def run(self, regime: str) -> SimResult:
        self._choose_visible_agent()

        steward_cycle = list(range(len(self.agents)))
        steward_idx = 0

        steps_survived = 0
        total_exits = 0

        for t in range(self.cfg["steps"]):
            self.step = t
            alive = self._alive_agents()
            if len(alive) < 3:
                break
            steps_survived += 1

            if t == self.cfg["shock_remove_top_agent_step"]:
                self._shock_remove_top(k=self.cfg.get("shock_remove_top_k", 1))

            if t == self.cfg["shock_attack_visible_agent_step"]:
                self._shock_attack_visible()

            if t == self.cfg.get("shock_attack_visible_agent_step_2", -1):
                self._shock_attack_visible()

            if t == self.cfg.get("random_dropout_step", -1):
                self._random_dropout(self.cfg.get("random_dropout_count", 1))

            if self.visible_agent_id is None or not self.agents[self.visible_agent_id].active:
                self._choose_visible_agent()

            pool = self._available_pool()
            task = self._spawn_task()

            steward_id = steward_cycle[steward_idx % len(steward_cycle)]
            steward_idx += 1

            if task is not None:
                self._attempt_task(regime, task, pool, steward_id)

            if regime == Regime.FFG and self.forks:
                self._resolve_forks()

            total_exits += self._apply_exits()

        loads = [a.responsibility_load for a in self.agents]
        return SimResult(
            completed=self.completed,
            attempted=self.attempted,
            failed=self.failed,
            exits=total_exits,
            avg_latency_proxy=(sum(self.latency_samples) / len(self.latency_samples)) if self.latency_samples else 0.0,
            responsibility_gini=gini(loads),
            steps_survived=steps_survived,
        )

def run_ab(cfg: Dict):
    simA = Simulator(cfg)
    resA = simA.run(Regime.HERO)

    simB = Simulator(cfg)
    resB = simB.run(Regime.FFG)

    return resA, resB

def summarize(results: List[SimResult]) -> Dict[str, float]:
    def mean(attr):
        vals = [getattr(r, attr) for r in results]
        return sum(vals) / max(1, len(vals))

    completion_rate = sum(r.completed for r in results) / max(1, sum(r.attempted for r in results))

    return {
        "steps_survived": mean("steps_survived"),
        "completion_rate": completion_rate,
        "exits": mean("exits"),
        "avg_latency_proxy": mean("avg_latency_proxy"),
        "responsibility_gini": mean("responsibility_gini"),
    }

if __name__ == "__main__":
    with open("config.json", "r") as f:
        cfg = json.load(f)

    seeds = cfg.get("sweep_seeds", [cfg.get("seed", 7)])

    hero_runs: List[SimResult] = []
    ffg_runs: List[SimResult] = []

    for s in seeds:
        cfg_run = dict(cfg)
        cfg_run["seed"] = s
        resA, resB = run_ab(cfg_run)
        hero_runs.append(resA)
        ffg_runs.append(resB)

    H = summarize(hero_runs)
    F = summarize(ffg_runs)

    print("\n=== SWEEP SUMMARY ===")
    print("seeds:", len(seeds))

    print("\nHERO:")
    for k, v in H.items():
        print(f"  {k:18s} {v:.3f}")

    print("\nFFG:")
    for k, v in F.items():
        print(f"  {k:18s} {v:.3f}")

    print("\nDIFF (FFG - HERO):")
    print("  d_steps_survived      :", round(F["steps_survived"] - H["steps_survived"], 3))
    print("  d_completion_rate     :", round(F["completion_rate"] - H["completion_rate"], 3))
    print("  d_exits               :", round(F["exits"] - H["exits"], 3))
    print("  d_avg_latency_proxy   :", round(F["avg_latency_proxy"] - H["avg_latency_proxy"], 3))
    print("  d_responsibility_gini :", round(F["responsibility_gini"] - H["responsibility_gini"], 3))
