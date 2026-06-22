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

    def _get_device(self, *args, **kwds):
        """Détecte et retourne le device sous forme de chaîne de caractères."""
        if not HAS_TORCH:
            return "cpu (PyTorch non installé)"
        
        for item in args + tuple(kwds.values()):
            if isinstance(item, torch.Tensor):
                return str(item.device)

        if torch.cuda.is_available():
            return "cuda (disponible, mais tenseurs non détectés dans les args)"
        
        return "cpu"

    def __call__(self, *args, **kwds):
        # Initialisation de la fonction dans le registre si elle n'existe pas
        if self._stats_key not in _STATS_REGISTRY:
            _STATS_REGISTRY[self._stats_key] = {}
        
        # On détermine le device AVANT de lancer l'exécution (ou juste pour l'indexation)
        device = self._get_device(*args, **kwds)
        
        # Initialisation du sous-dictionnaire pour ce device spécifique
        if device not in _STATS_REGISTRY[self._stats_key]:
            _STATS_REGISTRY[self._stats_key][device] = {
                "period": 0.0,
                "calls": 0
            }

        start = time.time()
        result = self.function(*args, **kwds)
        end = time.time()

        # Mise à jour des statistiques pour CE device
        stats = _STATS_REGISTRY[self._stats_key][device]
        stats["period"] += (end - start)
        stats["calls"] += 1

        # Sauvegarde périodique pour éviter de perdre des données 
        # en cas de fermeture brutale du worker
        total_calls_all_devices = sum(d["calls"] for d in _STATS_REGISTRY[self._stats_key].values())
        if total_calls_all_devices % 50 == 0:
            _save_local_stats()

        return result
    
    def print_stats(self):
        # Forcer la sauvegarde des stats locales avant lecture
        _save_local_stats()

        # Dictionnaire pour agréger les résultats de tous les workers par device
        # Format: { "cpu": {"period": X, "calls": Y}, "cuda:0": {"period": W, "calls": Z} }
        aggregated_stats = {}

        # Lecture et agrégation de tous les fichiers JSON
        for filename in os.listdir(_STATS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(_STATS_DIR, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                        if self._stats_key in data:
                            func_data = data[self._stats_key]
                            for device, device_stats in func_data.items():
                                if device not in aggregated_stats:
                                    aggregated_stats[device] = {"period": 0.0, "calls": 0}
                                
                                aggregated_stats[device]["period"] += device_stats["period"]
                                aggregated_stats[device]["calls"] += device_stats["calls"]
                except Exception:
                    continue

        print(f"\nChrono Stats pour '\033[34m{self.function.__name__}\033[0m':")
        
        if not aggregated_stats:
            print("\tAucune donnée enregistrée.")
            return

        # Affichage détaillé par device
        for device, stats in aggregated_stats.items():
            total_period = stats["period"]
            total_calls = stats["calls"]
            avg = total_period / total_calls if total_calls > 0 else 0
            
            print (
                f"\t[\033[33mDevice: {device}\033[0m]\n"
                f"\t  - Temps total: \033[36m{total_period:.6f}\033[0m s\n"
                f"\t  - Appels totaux: \033[36m{total_calls}\033[0m\n"
                f"\t  - Temps moyen/appel: \033[36m{avg:.6f}\033[0m\n"
            )

    def reset_stats(self):
        # Réinitialisation locale
        _STATS_REGISTRY[self._stats_key] = {}
        _save_local_stats()
        
        # Nettoyer les fichiers JSON existants pour faire table rase
        for filename in os.listdir(_STATS_DIR):
            if filename.endswith(".json"):
                try:
                    os.remove(os.path.join(_STATS_DIR, filename))
                except Exception:
                    pass
