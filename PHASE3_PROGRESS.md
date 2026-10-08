## Full pipeline validation (11_validate_sizing.py --city Mathura, quick NSGA-II)
- DQN correctly loads models/checkpoints/best_model.zip through the full chain: 
  11_validate_sizing.py -> validate_candidates.py -> DQNController, and the app.py /api/size 
  endpoint (validation is shared, candidate generation is separate).
- All three controllers (rule-based, MPC, DQN) agreed on candidate ranking: 
  best_npv > most_balanced > best_backup. Result: STABLE, confidence HIGH.
- On the recommended candidate (best_npv): DQN NPV ₹319,139 vs rule-based ₹316,903 vs MPC 
  ₹313,970. DQN unmet load 69.49 kWh vs rule-based 131.25 kWh vs MPC 144.66 kWh.
- Fairness warning still applies for very small-battery candidates (3.89/1.28 kWh), since 
  training used a fixed 10 kWh battery (Issue 6, not yet done).
- Quick-mode NSGA-II run used for this check (~10 min), not the full 1.5-2hr search.
