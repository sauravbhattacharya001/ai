"""Safety Profiles — save, load, list, compare, and delete named configurations.

Manage named safety configuration profiles so users can quickly switch
between different safety postures (e.g., "production", "testing",
"high-risk-research") without manually remembering parameter values.

Profiles are stored as JSON files in a configurable directory, each
capturing a full ``ScenarioConfig`` plus optional metadata (author,
description, tags, creation date).

Features
--------
- **Save** the current configuration as a named profile.
- **Load** a profile by name and get back a ``ScenarioConfig``.
- **List** all saved profiles with summary info.
- **Compare** two or more profiles side-by-side.
- **Delete** profiles that are no longer needed.
- **Built-in profiles** for common safety postures.
- **Validate** profiles against a safety policy before saving.

Usage (CLI)::

    python -m replication.profiles list                           # list all profiles
    python -m replication.profiles show production                # show one profile
    python -m replication.profiles save myprofile --strategy greedy --max-depth 3
    python -m replication.profiles save myprofile --from-preset balanced --desc "Balanced dev"
    python -m replication.profiles load myprofile --run           # load & run simulation
    python -m replication.profiles compare production testing     # side-by-side diff
    python -m replication.profiles delete old-experiment          # remove a profile
    python -m replication.profiles validate myprofile --policy strict
    python -m replication.profiles export myprofile               # print JSON to stdout
    python -m replication.profiles import profile.json            # import from file
    python -m replication.profiles --dir ./my-profiles list       # custom directory

Programmatic::

    from replication.profiles import ProfileManager, ProfileMeta
    mgr = ProfileManager("./profiles")

    # Save
    mgr.save("production", ScenarioConfig(max_depth=2, strategy="conservative"),
             desc="Locked-down production config", tags=["prod", "safe"])

    # Load
    config, meta = mgr.load("production")
    sim = Simulator(config)
    report = sim.run()

    # Compare
    diff = mgr.compare(["production", "testing"])
    print(diff.render())
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .simulator import ScenarioConfig, Simulator, PRESETS as SIM_PRESETS
from .policy import SafetyPolicy, POLICY_PRESETS
from ._helpers import box_header as _box_header


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class ProfileMeta:
    """Metadata attached to a saved profile."""

    name: str
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    based_on: str = ""  # preset or parent profile name

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "based_on": self.based_on,
        }.items() if v}


@dataclass
class ProfileDiff:
    """Differences between profiles for a single parameter."""

    param: str
    values: Dict[str, Any]  # profile_name -> value

    @property
    def is_different(self) -> bool:
        vals = list(self.values.values())
        return len(set(str(v) for v in vals)) > 1


@dataclass
class ComparisonResult:
    """Side-by-side comparison of multiple profiles."""

    profiles: List[str]
    diffs: List[ProfileDiff]
    metas: Dict[str, ProfileMeta]

    @property
    def changed_params(self) -> List[ProfileDiff]:
        return [d for d in self.diffs if d.is_different]

    @property
    def identical_params(self) -> List[ProfileDiff]:
        return [d for d in self.diffs if not d.is_different]

    def render(self) -> str:
        lines: List[str] = []
        lines.extend(_box_header("Profile Comparison"))
        lines.append("")

        # Header with profile names and descriptions
        for name in self.profiles:
            meta = self.metas.get(name)
            desc = f" — {meta.description}" if meta and meta.description else ""
            lines.append(f"  📋 {name}{desc}")
        lines.append("")

        # Changed parameters
        changed = self.changed_params
        if changed:
            lines.append(f"  ⚡ Differences ({len(changed)} parameters):")
            lines.append("")

            # Column widths
            param_w = max(len(d.param) for d in changed)
            col_w = max(max(len(n) for n in self.profiles), 8)

            header = f"    {'Parameter':<{param_w}}  " + "  ".join(
                f"{n:>{col_w}}" for n in self.profiles
            )
            lines.append(header)
            lines.append(f"    {'─' * param_w}  " + "  ".join(
                "─" * col_w for _ in self.profiles
            ))

            for d in changed:
                row = f"    {d.param:<{param_w}}  " + "  ".join(
                    f"{str(d.values.get(n, '—')):>{col_w}}" for n in self.profiles
                )
                lines.append(row)

            lines.append("")

        # Identical parameters
        identical = self.identical_params
        if identical:
            lines.append(f"  ✓ Identical ({len(identical)} parameters):")
            for d in identical:
                val = list(d.values.values())[0]
                lines.append(f"    {d.param}: {val}")
            lines.append("")

        return "\n".join(lines)


# ── Built-in profiles ──────────────────────────────────────────────────

BUILTIN_PROFILES: Dict[str, Tuple[ScenarioConfig, ProfileMeta]] = {
    "lockdown": (
        ScenarioConfig(
            max_depth=1,
            max_replicas=2,
            cooldown_seconds=5.0,
            strategy="conservative",
            tasks_per_worker=1,
            replication_probability=0.1,
        ),
        ProfileMeta(
            name="lockdown",
            description="Maximum restrictions — minimal replication allowed",
            tags=["production", "safe", "builtin"],
        ),
    ),
    "production": (
        ScenarioConfig(
            max_depth=2,
            max_replicas=5,
            cooldown_seconds=2.0,
            strategy="conservative",
            tasks_per_worker=2,
            replication_probability=0.3,
        ),
        ProfileMeta(
            name="production",
            description="Production-safe defaults with conservative replication",
            tags=["production", "builtin"],
        ),
    ),
    "balanced": (
        ScenarioConfig(
            max_depth=3,
            max_replicas=10,
            cooldown_seconds=1.0,
            strategy="random",
            tasks_per_worker=2,
            replication_probability=0.5,
        ),
        ProfileMeta(
            name="balanced",
            description="Balanced configuration for general development",
            tags=["development", "builtin"],
        ),
    ),
    "permissive": (
        ScenarioConfig(
            max_depth=5,
            max_replicas=25,
            cooldown_seconds=0.0,
            strategy="greedy",
            tasks_per_worker=3,
            replication_probability=0.7,
        ),
        ProfileMeta(
            name="permissive",
            description="Permissive settings for stress testing and research",
            tags=["testing", "research", "builtin"],
        ),
    ),
    "chaos": (
        ScenarioConfig(
            max_depth=8,
            max_replicas=50,
            cooldown_seconds=0.0,
            strategy="greedy",
            tasks_per_worker=4,
            replication_probability=0.9,
        ),
        ProfileMeta(
            name="chaos",
            description="Maximum chaos — for testing safety limits and boundaries",
            tags=["testing", "chaos", "builtin"],
        ),
    ),
}


# ── Profile Manager ─────────────────────────────────────────────────────


class ProfileManager:
    """Manage named safety configuration profiles on disk."""

    def __init__(self, directory: str = "./safety-profiles"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.directory / f"{safe}.json"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def save(
        self,
        name: str,
        config: ScenarioConfig,
        desc: str = "",
        author: str = "",
        tags: Optional[List[str]] = None,
        based_on: str = "",
        overwrite: bool = True,
    ) -> ProfileMeta:
        """Save a profile to disk. Returns the metadata."""
        path = self._path(name)
        now = self._now()

        # Preserve creation date if overwriting
        existing_created = ""
        if path.exists() and overwrite:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                existing_created = data.get("meta", {}).get("created_at", "")
            except (json.JSONDecodeError, OSError):
                pass
        elif path.exists() and not overwrite:
            raise FileExistsError(f"Profile '{name}' already exists. Use overwrite=True.")

        meta = ProfileMeta(
            name=name,
            description=desc,
            author=author,
            tags=tags or [],
            created_at=existing_created or now,
            updated_at=now,
            based_on=based_on,
        )

        payload = {
            "meta": meta.to_dict(),
            "config": {
                "max_depth": config.max_depth,
                "max_replicas": config.max_replicas,
                "cooldown_seconds": config.cooldown_seconds,
                "expiration_seconds": config.expiration_seconds,
                "strategy": config.strategy,
                "tasks_per_worker": config.tasks_per_worker,
                "replication_probability": config.replication_probability,
                "seed": config.seed,
                "cpu_limit": config.cpu_limit,
                "memory_limit_mb": config.memory_limit_mb,
            },
        }

        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return meta

    def load(self, name: str) -> Tuple[ScenarioConfig, ProfileMeta]:
        """Load a profile by name. Checks builtins first, then disk."""
        if name in BUILTIN_PROFILES:
            config, meta = BUILTIN_PROFILES[name]
            return config, meta

        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Profile '{name}' not found. "
                f"Available: {', '.join(self.list_names())}"
            )

        data = json.loads(path.read_text(encoding="utf-8"))
        cfg_data = data.get("config", {})
        meta_data = data.get("meta", {})

        config = ScenarioConfig(
            max_depth=cfg_data.get("max_depth", 3),
            max_replicas=cfg_data.get("max_replicas", 10),
            cooldown_seconds=cfg_data.get("cooldown_seconds", 0.0),
            expiration_seconds=cfg_data.get("expiration_seconds"),
            strategy=cfg_data.get("strategy", "greedy"),
            tasks_per_worker=cfg_data.get("tasks_per_worker", 2),
            replication_probability=cfg_data.get("replication_probability", 0.5),
            seed=cfg_data.get("seed"),
            cpu_limit=cfg_data.get("cpu_limit", 0.5),
            memory_limit_mb=cfg_data.get("memory_limit_mb", 256),
        )

        meta = ProfileMeta(
            name=meta_data.get("name", name),
            description=meta_data.get("description", ""),
            author=meta_data.get("author", ""),
            tags=meta_data.get("tags", []),
            created_at=meta_data.get("created_at", ""),
            updated_at=meta_data.get("updated_at", ""),
            based_on=meta_data.get("based_on", ""),
        )

        return config, meta

    def delete(self, name: str) -> bool:
        """Delete a saved profile. Returns True if deleted."""
        if name in BUILTIN_PROFILES:
            raise ValueError(f"Cannot delete built-in profile '{name}'.")
        path = self._path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_names(self) -> List[str]:
        """List all available profile names (builtins + saved)."""
        names = sorted(BUILTIN_PROFILES.keys())
        for p in sorted(self.directory.glob("*.json")):
            n = p.stem
            if n not in names:
                names.append(n)
        return names

    def list_all(self) -> List[Tuple[str, ProfileMeta, bool]]:
        """List all profiles with metadata. Returns (name, meta, is_builtin)."""
        results: List[Tuple[str, ProfileMeta, bool]] = []

        for name, (_, meta) in sorted(BUILTIN_PROFILES.items()):
            results.append((name, meta, True))

        for p in sorted(self.directory.glob("*.json")):
            name = p.stem
            if name in BUILTIN_PROFILES:
                continue
            try:
                _, meta = self.load(name)
                results.append((name, meta, False))
            except (json.JSONDecodeError, OSError):
                results.append((name, ProfileMeta(name=name, description="(corrupt)"), False))

        return results

    def compare(self, names: List[str]) -> ComparisonResult:
        """Compare two or more profiles side-by-side."""
        if len(names) < 2:
            raise ValueError("Need at least 2 profiles to compare.")

        configs: Dict[str, ScenarioConfig] = {}
        metas: Dict[str, ProfileMeta] = {}

        for name in names:
            config, meta = self.load(name)
            configs[name] = config
            metas[name] = meta

        # Extract all config parameters
        params = [
            "max_depth", "max_replicas", "cooldown_seconds",
            "expiration_seconds", "strategy", "tasks_per_worker",
            "replication_probability", "seed", "cpu_limit", "memory_limit_mb",
        ]

        diffs: List[ProfileDiff] = []
        for param in params:
            values = {}
            for name in names:
                cfg = configs[name]
                values[name] = getattr(cfg, param, None)
            diffs.append(ProfileDiff(param=param, values=values))

        return ComparisonResult(profiles=names, diffs=diffs, metas=metas)

    def validate(self, name: str, policy_preset: str = "standard") -> str:
        """Validate a profile against a safety policy. Returns rendered report."""
        config, meta = self.load(name)
        policy = SafetyPolicy.from_preset(policy_preset)
        sim = Simulator(config)
        report = sim.run()
        result = policy.evaluate(report)
        return result.render()

    def export_json(self, name: str) -> str:
        """Export a profile as a JSON string."""
        if name in BUILTIN_PROFILES:
            config, meta = BUILTIN_PROFILES[name]
            payload = {
                "meta": meta.to_dict(),
                "config": {
                    "max_depth": config.max_depth,
                    "max_replicas": config.max_replicas,
                    "cooldown_seconds": config.cooldown_seconds,
                    "expiration_seconds": config.expiration_seconds,
                    "strategy": config.strategy,
                    "tasks_per_worker": config.tasks_per_worker,
                    "replication_probability": config.replication_probability,
                    "seed": config.seed,
                    "cpu_limit": config.cpu_limit,
                    "memory_limit_mb": config.memory_limit_mb,
                },
            }
            return json.dumps(payload, indent=2)

        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"Profile '{name}' not found.")
        return path.read_text(encoding="utf-8")

    def import_json(self, json_path: str) -> str:
        """Import a profile from a JSON file. Returns the profile name."""
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        meta_data = data.get("meta", {})
        name = meta_data.get("name", Path(json_path).stem)
        cfg_data = data.get("config", {})

        config = ScenarioConfig(
            max_depth=cfg_data.get("max_depth", 3),
            max_replicas=cfg_data.get("max_replicas", 10),
            cooldown_seconds=cfg_data.get("cooldown_seconds", 0.0),
            expiration_seconds=cfg_data.get("expiration_seconds"),
            strategy=cfg_data.get("strategy", "greedy"),
            tasks_per_worker=cfg_data.get("tasks_per_worker", 2),
            replication_probability=cfg_data.get("replication_probability", 0.5),
            seed=cfg_data.get("seed"),
            cpu_limit=cfg_data.get("cpu_limit", 0.5),
            memory_limit_mb=cfg_data.get("memory_limit_mb", 256),
        )

        self.save(
            name, config,
            desc=meta_data.get("description", ""),
            author=meta_data.get("author", ""),
            tags=meta_data.get("tags", []),
            based_on=meta_data.get("based_on", ""),
        )
        return name

    def render_list(self) -> str:
        """Render a formatted list of all profiles."""
        lines: List[str] = []
        lines.extend(_box_header("Safety Profiles"))
        lines.append("")

        all_profiles = self.list_all()
        if not all_profiles:
            lines.append("  (no profiles found)")
            return "\n".join(lines)

        for name, meta, is_builtin in all_profiles:
            marker = "🔒" if is_builtin else "📄"
            desc = f" — {meta.description}" if meta.description else ""
            tags = f"  [{', '.join(meta.tags)}]" if meta.tags else ""
            lines.append(f"  {marker} {name}{desc}{tags}")

        lines.append("")
        lines.append(f"  Total: {len(all_profiles)} profiles "
                      f"({sum(1 for _, _, b in all_profiles if b)} built-in, "
                      f"{sum(1 for _, _, b in all_profiles if not b)} custom)")
        lines.append("")
        return "\n".join(lines)

    def render_show(self, name: str) -> str:
        """Render detailed view of a single profile."""
        config, meta = self.load(name)
        is_builtin = name in BUILTIN_PROFILES

        lines: List[str] = []
        lines.extend(_box_header(f"Profile: {name}"))
        lines.append("")

        marker = "🔒 Built-in" if is_builtin else "📄 Custom"
        lines.append(f"  Type:        {marker}")
        if meta.description:
            lines.append(f"  Description: {meta.description}")
        if meta.author:
            lines.append(f"  Author:      {meta.author}")
        if meta.tags:
            lines.append(f"  Tags:        {', '.join(meta.tags)}")
        if meta.based_on:
            lines.append(f"  Based on:    {meta.based_on}")
        if meta.created_at:
            lines.append(f"  Created:     {meta.created_at}")
        if meta.updated_at:
            lines.append(f"  Updated:     {meta.updated_at}")

        lines.append("")
        lines.append("  Configuration:")
        lines.append(f"    max_depth:                {config.max_depth}")
        lines.append(f"    max_replicas:             {config.max_replicas}")
        lines.append(f"    cooldown_seconds:         {config.cooldown_seconds}")
        lines.append(f"    expiration_seconds:       {config.expiration_seconds}")
        lines.append(f"    strategy:                 {config.strategy}")
        lines.append(f"    tasks_per_worker:         {config.tasks_per_worker}")
        lines.append(f"    replication_probability:  {config.replication_probability}")
        lines.append(f"    seed:                     {config.seed}")
        lines.append(f"    cpu_limit:                {config.cpu_limit}")
        lines.append(f"    memory_limit_mb:          {config.memory_limit_mb}")
        lines.append("")
        return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m replication.profiles",
        description="Safety Profiles — manage named configuration profiles",
    )
    p.add_argument("--dir", default="./safety-profiles",
                   help="Directory for profile storage (default: ./safety-profiles)")

    sub = p.add_subparsers(dest="command", help="Available commands")

    # list
    sub.add_parser("list", help="List all profiles")

    # show
    s = sub.add_parser("show", help="Show profile details")
    s.add_argument("name", help="Profile name")

    # save
    s = sub.add_parser("save", help="Save a new profile")
    s.add_argument("name", help="Profile name")
    s.add_argument("--from-preset", choices=list(SIM_PRESETS.keys()),
                   help="Base on a simulator preset")
    s.add_argument("--strategy", help="Override strategy")
    s.add_argument("--max-depth", type=int, help="Override max_depth")
    s.add_argument("--max-replicas", type=int, help="Override max_replicas")
    s.add_argument("--cooldown", type=float, help="Override cooldown_seconds")
    s.add_argument("--probability", type=float, help="Override replication_probability")
    s.add_argument("--desc", default="", help="Description")
    s.add_argument("--author", default="", help="Author")
    s.add_argument("--tags", nargs="*", default=[], help="Tags")

    # load
    s = sub.add_parser("load", help="Load a profile")
    s.add_argument("name", help="Profile name")
    s.add_argument("--run", action="store_true", help="Run simulation after loading")
    s.add_argument("--json", action="store_true", help="JSON output")

    # compare
    s = sub.add_parser("compare", help="Compare profiles")
    s.add_argument("names", nargs="+", help="Profile names (2 or more)")

    # delete
    s = sub.add_parser("delete", help="Delete a profile")
    s.add_argument("name", help="Profile name")

    # validate
    s = sub.add_parser("validate", help="Validate profile against policy")
    s.add_argument("name", help="Profile name")
    s.add_argument("--policy", default="standard", help="Policy preset")

    # export
    s = sub.add_parser("export", help="Export profile as JSON")
    s.add_argument("name", help="Profile name")

    # import
    s = sub.add_parser("import", help="Import profile from JSON file")
    s.add_argument("file", help="Path to JSON file")

    return p


def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    mgr = ProfileManager(args.dir)

    if args.command == "list" or args.command is None:
        print(mgr.render_list())

    elif args.command == "show":
        print(mgr.render_show(args.name))

    elif args.command == "save":
        if args.from_preset:
            config = SIM_PRESETS[args.from_preset]
        else:
            config = ScenarioConfig()

        if args.strategy:
            config = ScenarioConfig(
                max_depth=config.max_depth,
                max_replicas=config.max_replicas,
                cooldown_seconds=config.cooldown_seconds,
                expiration_seconds=config.expiration_seconds,
                strategy=args.strategy,
                tasks_per_worker=config.tasks_per_worker,
                replication_probability=config.replication_probability,
                seed=config.seed,
                cpu_limit=config.cpu_limit,
                memory_limit_mb=config.memory_limit_mb,
            )
        if args.max_depth is not None:
            config.max_depth = args.max_depth
        if args.max_replicas is not None:
            config.max_replicas = args.max_replicas
        if args.cooldown is not None:
            config.cooldown_seconds = args.cooldown
        if args.probability is not None:
            config.replication_probability = args.probability

        meta = mgr.save(
            args.name, config,
            desc=args.desc,
            author=args.author,
            tags=args.tags,
            based_on=args.from_preset or "",
        )
        print(f"✓ Saved profile '{args.name}'")
        print(mgr.render_show(args.name))

    elif args.command == "load":
        config, meta = mgr.load(args.name)
        if args.json:
            print(mgr.export_json(args.name))
        else:
            print(mgr.render_show(args.name))

        if args.run:
            print("\nRunning simulation with loaded profile...\n")
            sim = Simulator(config)
            report = sim.run()
            print(report.render_tree())
            print(report.render_timeline())

    elif args.command == "compare":
        result = mgr.compare(args.names)
        print(result.render())

    elif args.command == "delete":
        if mgr.delete(args.name):
            print(f"✓ Deleted profile '{args.name}'")
        else:
            print(f"✗ Profile '{args.name}' not found")

    elif args.command == "validate":
        print(f"Validating '{args.name}' against '{args.policy}' policy...\n")
        print(mgr.validate(args.name, args.policy))

    elif args.command == "export":
        print(mgr.export_json(args.name))

    elif args.command == "import":
        name = mgr.import_json(args.file)
        print(f"✓ Imported profile '{name}'")
        print(mgr.render_show(name))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
