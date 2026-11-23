"""Wrapper around predgate.benchmark.run_benchmark (single source of truth in docs/benchmark.md)."""

from __future__ import annotations

from rich.console import Console

from predgate.benchmark import run_benchmark


def main() -> None:
    console = Console()
    run_benchmark(console, n_seeds=30, start_seed=1000, command_str="predgate bench")


if __name__ == "__main__":
    main()
