"""
Décorateurs pour mesurer les temps d'exécution dans DDW2.
Version compatible avec multiprocessing et DDP.
"""

import time
import torch
import os
from functools import wraps
from typing import Callable, Any, Dict, Optional, List
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class TimingResult:
    function_name: str
    execution_time: float
    device: str
    input_size: Optional[Dict] = None
    output_size: Optional[Dict] = None

_timing_results: List[TimingResult] = []

def _is_in_worker_process() -> bool:
    """Vérifie si on est dans un processus worker DataLoader."""
    # Méthode 1: Vérifier si on est dans un worker DataLoader
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        return True

    # Méthode 2: Vérifier si c'est un processus enfant
    if os.getenv("LOCAL_RANK", "0") != "0":
        return True

    return False

def pytorch_timeit(func: Callable) -> Callable:
    """
    Décorateur PyTorch compatible avec multiprocessing et DDP.
    Désactive la synchronisation CUDA dans les workers DataLoader.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # NE PAS synchroniser CUDA dans les workers DataLoader
        in_worker = _is_in_worker_process()

        if not in_worker and torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        if not in_worker and torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # Détection du device (sans synchronisation dans les workers)
        device = "CPU"
        input_size = None
        output_size = None

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

        timing_result = TimingResult(
            function_name=func.__name__,
            execution_time=execution_time,
            device=device,
            input_size=input_size,
            output_size=output_size
        )
        _timing_results.append(timing_result)

        # N'affiche que sur le processus principal (rank 0)
        if os.getenv("LOCAL_RANK", "0") == "0":
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
    """
    Affiche un résumé COMPLET de toutes les mesures de timing.
    - Temps total par fonction
    - Pourcentage du temps total
    - Nombre d'appels
    - Moyenne, min, max
    - Device utilisé
    """
    if os.getenv("LOCAL_RANK", "0") != "0":
        return  # N'affiche que sur le processus principal

    if not _timing_results:
        print("⚠️  Aucune mesure de temps enregistrée.")
        return

    # Calculer le temps total de TOUTES les fonctions
    total_time_all_functions = sum(r.execution_time for r in _timing_results)

    # Regrouper par fonction
    stats = defaultdict(lambda: {
        "count": 0,
        "total": 0.0,
        "times": [],
        "devices": set(),
        "input_sizes": [],
        "output_sizes": []
    })

    for result in _timing_results:
        func_name = result.function_name
        stats[func_name]["count"] += 1
        stats[func_name]["total"] += result.execution_time
        stats[func_name]["times"].append(result.execution_time)
        stats[func_name]["devices"].add(result.device)
        if result.input_size:
            stats[func_name]["input_sizes"].append(result.input_size)
        if result.output_size:
            stats[func_name]["output_sizes"].append(result.output_size)

    # Trier par temps total décroissant
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)

    print("\n" + "=" * 120)
    print("📊 DDW2 - RÉSUMÉ COMPLET DES TEMPS D'EXÉCUTION")
    print("=" * 120)
    print(f"{'Fonction':<35} {'Appels':>7} {'Total (s)':>11} {'% Total':>8} {'Moy (s)':>11} {'Min (s)':>11} {'Max (s)':>11} {'Device':<12}")
    print("-" * 120)

    for func_name, data in sorted_stats:
        count = data["count"]
        total = data["total"]
        avg = total / count if count > 0 else 0
        min_t = min(data["times"]) if data["times"] else 0
        max_t = max(data["times"]) if data["times"] else 0
        percentage = (total / total_time_all_functions * 100) if total_time_all_functions > 0 else 0
        devices = ", ".join(sorted(data["devices"]))

        print(f"{func_name:<35} {count:>7} {total:>11.4f} {percentage:>7.1f}% {avg:>11.4f} {min_t:>11.4f} {max_t:>11.4f} {devices:<12}")

    print("-" * 120)
    print(f"{'TOTAL':<35} {'':<7} {total_time_all_functions:>11.4f} {'100.0%':>8} {'':<11} {'':<11} {'':<11}")
    print("=" * 120)

    # Section détaillée : Temps par étape principale
    print("\n📈 DÉTAIL PAR ÉTAPE PRINCIPALE:")
    print("-" * 60)

    # Identifier les étapes principales
    main_steps = {
        "Préparation des données": ["prepare_data"],
        "Chargement des données": ["__getitem__"],
        "Forward pass": ["forward"],
        "Training step": ["training_step"],
        "Validation step": ["validation_step"],
        "Calcul de perte": ["masked_loss"],
        "Mise à jour missing wedges": ["update_subtomo_missing_wedges"],
        "Autres": []
    }

    # Calculer le temps par catégorie
    step_totals = {step: 0.0 for step in main_steps}
    assigned_functions = set()

    for func_name, data in stats:
        for step, functions in main_steps.items():
            if func_name in functions:
                step_totals[step] += data["total"]
                assigned_functions.add(func_name)
                break
        else:
            step_totals["Autres"] += data["total"]

    for step, total in step_totals.items():
        if total > 0:
            percentage = (total / total_time_all_functions * 100) if total_time_all_functions > 0 else 0
            print(f"  {step:<30}: {total:>8.4f}s ({percentage:>5.1f}%)")

    print("-" * 60)

    # Section : Quantité de données
    print("\n💾 QUANTITÉ DE DONNÉES TRANSITÉES:")
    print("-" * 60)

    data_volume = {
        "Entrée (Mo)": 0,
        "Sortie (Mo)": 0
    }

    for result in _timing_results:
        if result.input_size:
            if isinstance(result.input_size, dict) and "size_bytes" in result.input_size:
                data_volume["Entrée (Mo)"] += result.input_size["size_bytes"] / (1024 * 1024)
            elif isinstance(result.input_size, dict):
                for v in result.input_size.values():
                    data_volume["Entrée (Mo)"] += v.get("size_bytes", 0) / (1024 * 1024)

        if result.output_size:
            if isinstance(result.output_size, dict) and "size_bytes" in result.output_size:
                data_volume["Sortie (Mo)"] += result.output_size["size_bytes"] / (1024 * 1024)
            elif isinstance(result.output_size, dict):
                for v in result.output_size.values():
                    data_volume["Sortie (Mo)"] += v.get("size_bytes", 0) / (1024 * 1024)

    print(f"  Données d'entrée totales: {data_volume['Entrée (Mo)']:.2f} Mo")
    print(f"  Données de sortie totales: {data_volume['Sortie (Mo)']:.2f} Mo")
    print(f"  Total: {sum(data_volume.values()):.2f} Mo")
    print("=" * 120)

def reset_timing_stats() -> None:
    global _timing_results
    _timing_results = []