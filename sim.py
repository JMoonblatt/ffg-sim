import json, random, math, statistics
from dataclasses import dataclass, field
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
    # In v0, a fork is just a temporary parallel attempt; we don't model branch content deeply.
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
        # adjust to exact n
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
        return agents

    def _alive_agents(self) -> List[Agent]:
        return [a for a in self.agents if a.active]

    def _choose_visible_agent(self):
        # visible = high visibility_bias + competence
        alive = self._alive_agents()
        if not alive:
            self.visible_agent_id = None
            return
        scores = [(a.visibility_bias * 0.6 + a.competence * 0.4, a.id) for a in alive]
        scores.sort(reverse=True)
        self.visible_agent_id = scores[0][1]

    def _shock_remove_top(self):
        alive = self._alive_agents()
        if not alive:
            return
        top = max(alive, key=lambda a: a.competence)
        top.active = False

    def _shock_attack_visible(self):
        if self.visible_agent_id is None:
            return
        a = self.agents[self.visible_agent_id]
        if a.active:
            a.availability = clamp(a.availability * self.cfg["attack_availability_multiplier"], 0.05, 0.98)

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
        # deference weights: competence + visibility, skewed
        strength = self.cfg["hero_deference_strength"]
        scores = []
        for a in pool:
            # visibility makes them more likely to be chosen / deferred to
            s = (0.75 * a.competence + 0.25 * a.visibility_bias)
            scores.append((s, a))
        scores.sort(reverse=True, key=lambda x: x[0])

        # choose a "hero" then pick others by competence
        hero = scores[0][1]
        team = [hero] + [a for _, a in scores[1:task.coordination]]
        return team

    def _ffg_select_team(self, task: Task, pool: List[Agent], steward_id: int) -> Tuple[List[Agent], float, bool]:
        """
        Returns (team, latency_proxy, forked)
        """
        if len(pool) < task.coordination:
            return ([], 0.0, False)

        # Overhead scales with ambiguity (more debate, more logging)
        overhead = self.cfg["ffg_overhead_base"] + self.cfg["ffg_overhead_ambiguity_scale"] * task.ambiguity
        latency_proxy = overhead * task.coordination

        # Disagreement probability rises with ambiguity and motivation mix.
        # We model it as: probability someone objects strongly enough to fork.
        # selfish agents object more to personal cost; prosocial object more to group harm.
        base_disagree = task.ambiguity * 0.6
        # sample a small "committee" preference spread
        committee = self.rng.sample(pool, k=min(task.coordination + 2, len(pool)))
        preference_spread = 0.0
        for a in committee:
            w_group, w_cost, w_ov = a.utility_weights()
            # crude "stance": prefers payoff vs overhead & personal load
            stance = w_group * task.payoff - w_ov * overhead - w_cost * (task.coordination * 0.08)
            preference_spread += stance
        preference_spread /= max(1, len(committee))

        # Lower spread (near zero) = more contention
        contention = 1.0 - clamp(abs(preference_spread) / 5.0, 0.0, 1.0)
        p_fork = clamp(base_disagree * contention, 0.0, 1.0)

        forked = (p_fork > self.cfg["fork_threshold"])
        # Team selection under FFG: steward is included if available; otherwise best-fit by competence
        pool_ids = [a.id for a in pool]
        if steward_id in pool_ids:
            steward = self.agents[steward_id]
        else:
            steward = max(pool, key=lambda a: a.competence)

        # pick remaining by competence but avoid always selecting same top person by adding fatigue penalty
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
            # Success probability increases with total competence; ambiguity hurts; hero bottleneck adds fatigue.
            total_comp = sum(a.competence for a in team)
            p_success = clamp((total_comp / task.coordination) * 0.95 - task.ambiguity * 0.25, 0.05, 0.98)

            hero = team[0]
            # hero takes disproportionate load
            hero_load = 1.0
            other_load = 0.35
            for i, a in enumerate(team):
                load = hero_load if i == 0 else other_load
                a.responsibility_load += load
                a.fatigue += load * self.cfg["burnout_load_scale"]

            # latency proxy lower
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

            # If forked, create a fork trial; you still attempt one path now with slightly reduced success (split attention),
            # then the fork resolves later.
            split_penalty = 0.12 if forked else 0.0

            total_comp = sum(a.competence for a in team)
            p_success = clamp((total_comp / task.coordination) * 0.90 - task.ambiguity * 0.18 - split_penalty, 0.05, 0.98)

            # load more evenly distributed, steward slightly higher but bounded
            steward = team[0]
            for i, a in enumerate(team):
                load = 0.55 if a.id == steward.id else 0.45
                a.responsibility_load += load
                a.fatigue += load * self.cfg["burnout_load_scale"]

            if forked:
                self.forks.append(Fork(task_id=task.id, remaining=self.cfg["fork_trial_length"], contenders=("A", "B")))

            if self.rng.random() < p_success:
                self.completed += 1
            else:
                self.failed += 1

    def _resolve_forks(self):
        # Simple model: forks consume attention for a while; when they resolve, you recover some benefit (extra completions)
        # but not always. This is intentionally conservative.
        kept = []
        for f in self.forks:
            f.remaining -= 1
            if f.remaining <= 0:
                # merge success chance depends on current ambiguity climate; we approximate with a constant
                if self.rng.random() < 0.65:
                    self.completed += 1  # merged improvement
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
        # choose visible at start
        self._choose_visible_agent()

        # rotate steward schedule = simple round-robin over agent IDs
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
                self._shock_remove_top()
            if t == self.cfg["shock_attack_visible_agent_step"]:
                self._shock_attack_visible()

            # refresh visible agent if previous is gone
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
        res = SimResult(
            completed=self.completed,
            attempted=self.attempted,
            failed=self.failed,
            exits=total_exits,
            avg_latency_proxy=(sum(self.latency_samples) / len(self.latency_samples)) if self.latency_samples else 0.0,
            responsibility_gini=gini(loads),
            steps_survived=steps_survived,
        )
        return res

def run_ab(cfg: Dict):
    # Run HERO
    simA = Simulator(cfg)
    resA = simA.run(Regime.HERO)

    # Run FFG
    simB = Simulator(cfg)
    resB = simB.run(Regime.FFG)

    return resA, resB

def print_result(name: str, r: SimResult):
    attempted = max(1, r.attempted)
    print(f"\n=== {name} ===")
    print(f"steps_survived       : {r.steps_survived}")
    print(f"attempted            : {r.attempted}")
    print(f"completed            : {r.completed}")
    print(f"failed               : {r.failed}")
    print(f"completion_rate      : {r.completed/attempted:.3f}")
    print(f"exits (burnout)      : {r.exits}")
    print(f"avg_latency_proxy    : {r.avg_latency_proxy:.3f}")
    print(f"responsibility_gini  : {r.responsibility_gini:.3f}")

if __name__ == "__main__":
    with open("config.json", "r") as f:
        cfg = json.load(f)

    resA, resB = run_ab(cfg)
    print_result("HERO", resA)
    print_result("FFG", resB)

    # A crude "win" report that matches our stated objective: resilience and load diffusion.
    print("\n=== Quick comparison (interpretation) ===")
print("Prefer: higher steps_survived, lower responsibility_gini, fewer exits.")
print("Completion rate matters, but not at the expense of collapse risk.")
print("d_steps_survived      :", resB.steps_survived - resA.steps_survived)
print("d_responsibility_gini :", round(resB.responsibility_gini - resA.responsibility_gini, 3))
print("d_exits               :", resB.exits - resA.exits)
print("d_completion_rate     :", round((resB.completed/max(1,resB.attempted)) - (resA.completed/max(1,resA.attempted)), 3))

