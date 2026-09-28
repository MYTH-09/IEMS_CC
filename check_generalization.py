import runpy
import numpy as np
from src.environment.energy_env import EnergyEnv
from src.grid.grid_model import GridModel
from src.controllers.dqn_controller import DQNController
from src.controllers.mpc_controller import MPCController
from src.controllers.rule_based import RuleBasedController

ns = runpy.run_path('scripts/07_train_dqn.py')
build_household_series = ns['build_household_series']

battery_config = dict(
    capacity_kwh=10.0,
    max_charge_rate_kw=3.0,
    max_discharge_rate_kw=3.0,
    round_trip_efficiency=0.90,
    min_soc=0.10,
    max_soc=0.95,
    initial_soc=0.50,
)

def make_env(load, solar, timestamps):
    grid = GridModel(
        fallback_outage_hours_per_day=4.0,
        outage_peak_hours=[10, 14, 18, 22],
    )
    return EnergyEnv(
        load_kw=load,
        solar_kw=solar,
        timestamps=timestamps,
        battery_config=battery_config,
        grid=grid,
        episode_days=30,
        seed=999,
    )

def run_controller(controller, load, solar, timestamps, seed):
    env = make_env(load, solar, timestamps)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    while True:
        action = controller.select_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return {
        'reward': total_reward,
        'import': env.episode_grid_import_kwh,
        'export': env.episode_grid_export_kwh,
        'unmet': env.episode_unmet_load_kwh,
    }

households = [(888, 0.9), (777, 1.4), (666, 0.7)]
all_results = []

for household_seed, load_scale in households:
    load, solar, timestamps = build_household_series(
        n_days=60, seed=household_seed, load_scale=load_scale
    )
    controllers = {
        'DQN': DQNController(model_path='models/checkpoints/best_model.zip'),
        'MPC': MPCController(horizon_hours=24.0),
        'Rule-based': RuleBasedController(),
    }
    results = {}
    for name, controller in controllers.items():
        results[name] = run_controller(controller, load, solar, timestamps, household_seed)

    baseline = results['Rule-based']['reward']
    dqn_improvement = ((results['DQN']['reward'] - baseline) / abs(baseline) * 100
                        if baseline != 0 else float('nan'))
    mpc_improvement = ((results['MPC']['reward'] - baseline) / abs(baseline) * 100
                        if baseline != 0 else float('nan'))
    all_results.append((dqn_improvement, mpc_improvement))

    print(f'Household seed={household_seed}, load_scale={load_scale}')
    for name in ['DQN', 'MPC', 'Rule-based']:
        r = results[name]
        print(f'  {name}: reward={r["reward"]:.4f}, import={r["import"]:.4f}, '
              f'export={r["export"]:.4f}, unmet={r["unmet"]:.4f}')
    print(f'  DQN improvement over rule-based: {dqn_improvement:.2f}%')
    print(f'  MPC improvement over rule-based: {mpc_improvement:.2f}%')
    print()

print('=== Average improvements across households ===')
print(f'  DQN: {np.mean([x[0] for x in all_results]):.2f}%')
print(f'  MPC: {np.mean([x[1] for x in all_results]):.2f}%')
