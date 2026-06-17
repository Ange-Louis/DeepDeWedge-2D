"""
Décorateurs pour mesurer les temps d'exécution dans DDW2.
Support CPU et GPU avec synchronisation automatique.
"""

import time
import torch
from functools import wraps
from typing import Callable, Any, Dict, Optional, List
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TimingResult:
    """Structure pour stocker les résultats de timing."""
    function_name: str
    execution_time: float
    device: str
    input_size: Optional[Dict] = None
    output_size: Optional[Dict] = None
    call_count: int = 1

# Liste globale pour stocker tous les résultats
_timing_results: List[TimingResult] = []

def pytorch_timeit(func: Callable) -> Callable:
    """
    Décorateur PyTorch pour mesurer le temps d'exécution avec support CPU/GPU.
    Synchronise automatiquement le GPU pour des mesures précises.

    Exemple:
        @pytorch_timeit
        def ma_fonction(x):
            return x * 2
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Synchronisation GPU initiale si disponible
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        # Synchronisation GPU finale si disponible
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Détection du device et taille des données
        device = "CPU"
        input_size = None
        output_size = None

        # Analyser les arguments d'entrée
        all_args = list(args) + list(kwargs.values())
        for arg in all_args:
            if isinstance(arg, torch.Tensor):
                device = f"GPU:{arg.device.index}" if arg.is_cuda else "CPU"
                if input_size is None:
                    input_size = {
                        "shape": list(arg.shape),
                        "dtype": str(arg.dtype),
                        "size_bytes": arg.numel() * arg.element_size()
                    }
            elif isinstance(arg, dict):
                for key, value in arg.items():
                    if isinstance(value, torch.Tensor):
                        device = f"GPU:{value.device.index}" if value.is_cuda else "CPU"
                        if input_size is None:
                            input_size = {}
                        input_size[key] = {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "size_bytes": value.numel() * value.element_size()
                        }

        # Analyser le résultat
        if isinstance(result, torch.Tensor):
            output_size = {
                "shape": list(result.shape),
                "dtype": str(result.dtype),
                "size_bytes": result.numel() * result.element_size()
            }
        elif isinstance(result, dict):
            output_size = {}
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    output_size[key] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "size_bytes": value.numel() * value.element_size()
                    }

        # Stocker le résultat
        timing_result = TimingResult(
            function_name=func.__name__,
            execution_time=execution_time,
            device=device,
            input_size=input_size,
            output_size=output_size
        )
        _timing_results.append(timing_result)

        # Affichage formaté
        print(f"⏱️  {func.__name__:30} | {device:10} | {execution_time:8.4f}s", end="")

        if input_size:
            if isinstance(input_size, dict) and "size_bytes" in input_size:
                print(f" | Input: {input_size['shape']} ({input_size['size_bytes']/1024/1024:.2f} Mo)", end="")
            elif isinstance(input_size, dict):
                total_bytes = sum(v.get('size_bytes', 0) for v in input_size.values())
                print(f" | Input: {total_bytes/1024/1024:.2f} Mo", end="")

        if output_size:
            if isinstance(output_size, dict) and "size_bytes" in output_size:
                print(f" | Output: {output_size['shape']} ({output_size['size_bytes']/1024/1024:.2f} Mo)")
            elif isinstance(output_size, dict):
                total_bytes = sum(v.get('size_bytes', 0) for v in output_size.values())
                print(f" | Output: {total_bytes/1024/1024:.2f} Mo")
            else:
                print()
        else:
            print()

        return result
    return wrapper

def print_timing_summary() -> None:
    """Affiche un résumé complet des temps mesurés."""
    if not _timing_results:
        print("Aucune mesure de temps enregistrée.")
        return

    # Regrouper par fonction
    stats = defaultdict(lambda: {"count": 0, "total": 0.0, "times": [], "devices": set()})

    for result in _timing_results:
        stats[result.function_name]["count"] += 1
        stats[result.function_name]["total"] += result.execution_time
        stats[result.function_name]["times"].append(result.execution_time)
        stats[result.function_name]["devices"].add(result.device)

    print("\n" + "=" * 100)
    print("📊 DDW2 - RÉSUMÉ DES TEMPS D'EXÉCUTION")
    print("=" * 100)
    print(f"{'Fonction':<35} {'Appels':>6} {'Total (s)':>10} {'Moyenne (s)':>12} {'Min (s)':>10} {'Max (s)':>10} {'Device':<15}")
    print("-" * 100)

    for func_name, data in sorted(stats.items()):
        avg = data["total"] / data["count"]
        min_t = min(data["times"])
        max_t = max(data["times"])
        devices = ", ".join(sorted(data["devices"]))
        print(f"{func_name:<35} {data['count']:>6} {data['total']:>10.4f} {avg:>12.4f} {min_t:>10.4f} {max_t:>10.4f} {devices:<15}")

    print("=" * 100)

def reset_timing_stats() -> None:
    """Réinitialise les statistiques de timing."""
    global _timing_results
    _timing_results = []

def get_timing_stats() -> List[TimingResult]:
    """Retourne toutes les statistiques de timing."""
    return _timing_results.copy()