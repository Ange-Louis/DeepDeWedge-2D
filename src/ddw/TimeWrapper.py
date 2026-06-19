import time
from functools import update_wrapper
import os
import json
import uuid
import tempfile
import atexit

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Abandon du mp.Manager() qui est incompatible avec les "daemons" PyTorch.
# Utilisation de dictionnaires locaux et de sauvegarde sur disque.
_STATS_REGISTRY = {}
_WORKER_ID = str(uuid.uuid4())
_STATS_DIR = os.path.join(tempfile.gettempdir(), "ddw_chrono_stats")

os.makedirs(_STATS_DIR, exist_ok=True)

def _save_local_stats():
    """Sauvegarde les statistiques locales de CE processus dans un fichier JSON."""
    if not _STATS_REGISTRY:
        return
    filepath = os.path.join(_STATS_DIR, f"stats_{_WORKER_ID}.json")
    try:
        with open(filepath, "w") as f:
            json.dump(_STATS_REGISTRY, f)
    except Exception:
        pass

# S'assure que chaque worker PyTorch enregistre ses stats juste avant de se fermer
atexit.register(_save_local_stats)

class Chrono:
    def __init__(self, function):
        self.function = function
        self._stats_key = function.__name__
        update_wrapper(self, function)

    def _init_in_registry(self):
        # Initialisation purement locale, sans aucun processus d'arrière-plan
        if self._stats_key not in _STATS_REGISTRY:
            _STATS_REGISTRY[self._stats_key] = {
                "period": 0.0,
                "calls": 0,
                "device": "cpu"
            }
        return _STATS_REGISTRY

    def _detect_device(self, registry, *args, **kwds):
        stats = registry[self._stats_key]
        if not HAS_TORCH:
            stats["device"] = "cpu (PyTorch non installé)"
        else:
            device_found = False
            for item in args + tuple(kwds.values()):
                if isinstance(item, torch.Tensor):
                    stats["device"] = str(item.device)
                    device_found = True
                    break

            if not device_found:
                if torch.cuda.is_available():
                    stats["device"] = "cuda (disponible, mais tenseurs non détectés dans les args)"
                else: 
                    stats["device"] = "cpu"

    def __call__(self, *args, **kwds):
        registry = self._init_in_registry()
        
        start = time.time()
        result = self.function(*args, **kwds)
        end = time.time()

        stats = registry[self._stats_key]
        stats["period"] += (end - start)
        stats["calls"] += 1

        self._detect_device(registry, *args, **kwds)

        # Sauvegarde périodique (tous les 50 appels) au cas où le worker 
        # PyTorch serait forcé de se fermer brutalement sans appeler 'atexit'.
        if stats["calls"] % 50 == 0:
            _save_local_stats()

        return result
    
    def print_stats(self):
        # Forcer la sauvegarde des stats locales du processus principal avant de tout lire
        _save_local_stats()

        total_period = 0.0
        total_calls = 0
        last_device = "cpu"

        # Lecture et agrégation de tous les fichiers JSON laissés par les différents workers PyTorch
        for filename in os.listdir(_STATS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(_STATS_DIR, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        if self._stats_key in data:
                            total_period += data[self._stats_key]["period"]
                            total_calls += data[self._stats_key]["calls"]
                            last_device = data[self._stats_key]["device"]
                except Exception:
                    continue

        avg = total_period / total_calls if total_calls > 0 else 0
        print (
            f"Chrono Stats pour '\033[34m{self.function.__name__}\033[0m':\n"
            f"\t- Temps total: \033[36m{total_period:.6f}\033[0m s\n"
            f"\t- Appels totaux (tous workers confondus): \033[36m{total_calls}\033[0m\n"
            f"\t- Temps moyen/appel: \033[36m{avg:.6f}\033[0m\n"
            f"\t- Dernier Device détecté: \033[36m{last_device}\033[0m"
        )

    def reset_stats(self):
        registry = self._init_in_registry()
        registry[self._stats_key] = {
            "period": 0.0,
            "calls": 0,
            "device": "cpu"
        }
        _save_local_stats()
        
        # Nettoyer les fichiers JSON existants pour faire table rase
        for filename in os.listdir(_STATS_DIR):
            if filename.endswith(".json"):
                try:
                    os.remove(os.path.join(_STATS_DIR, filename))
                except Exception:
                    pass
