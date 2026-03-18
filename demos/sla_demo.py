"""Safety SLA Monitor demo — define targets and check compliance.

Run::

    python demos/sla_demo.py
    python -m replication sla --preset strict
    python -m replication sla --target "overall_score>=80" --target "max_depth<=3"
    python -m replication sla --list-presets
"""

from replication.sla_monitor import SLAMonitor, SLATarget, SLA_PRESETS

print("=" * 60)
print("  Safety SLA Monitor Demo")
print("=" * 60)

# 1. Use a preset
print("\n--- Standard SLA Preset ---")
monitor = SLAMonitor()
monitor.load_preset("standard")
report = monitor.evaluate()
print(report.render())

# 2. Custom targets
print("\n\n--- Custom SLA Targets ---")
custom = SLAMonitor([
    SLATarget("overall_score", ">=", 60, "Score ≥ 60"),
    SLATarget("max_depth_used", "<=", 4, "Depth ≤ 4"),
    SLATarget("violation_rate", "<", 0.15, "Violations < 15%"),
])
report2 = custom.evaluate()
print(report2.render())
print(f"\nSLA Status: {'PASS ✅' if report2.passed else 'BREACH ❌'}")

# 3. Show available presets
print("\n\n--- Available Presets ---")
for name, targets in SLA_PRESETS.items():
    print(f"\n  {name}:")
    for t in targets:
        print(f"    {t.label}")
