import time
from functools import update_wrapper
import torch.multiprocessing as mp

try:
	import torch
	HAS_TORCH= True
except:
	HAS_TORCH= False

# Initialise un manager pour partager les stats entre processus
if mp.get_start_method() != 'spawn':
	mp.set_start_method('spawn', force=True)
_manager = mp.Manager()
_STATS_REGISTRY = _manager.dict()

class Chrono:
	def __init__(self, function):
		# Sauvegarde la fonction initiale
		self.function= function

		# Utilise le nom de la fonction comme clef unique
		self._stats_key= function.__name__
		
		# Initialise les stats dans le registre global si elles n'existent pas
		if self._stats_key not in _STATS_REGISTRY:
			_STATS_REGISTRY[self._stats_key]= {
				"period": 0.0,
				"calls": 0,
				"device": "cpu",
				"printed":  False
			}

		# Remplace @wraps pour une classe : copie de l'identité de la fonction
		update_wrapper(self, function)

	def _detect_device(self, *args, **kwds):
		"""
		Détecte si l'exéctution se fait sur CPU ou GPU
		"""
		if not HAS_TORCH:
			_STATS_REGISTRY[self._stats_key]["device"] = "cpu (PyTorch non installé)"
			return
		
		# Vérifie les tenseurs dans les arguments 
		for item in args + tuple(kwds.values()):
			if isinstance(item, torch.Tensor):
				_STATS_REGISTRY[self._stats_key]["device"] = str(item.device)
				return
			
		# Si aucun tenseur dans les args, vérifie si CUDA est utilisé
		if torch.cuda.is_available():
			# Vérifie si des tenseurs ont été créés sur GPU dans la fonction
			_STATS_REGISTRY[self._stats_key]["device"] = "cuda (disponible, mais tenseurs non détectés dans les args)"
		else: 
			_STATS_REGISTRY[self._stats_key]["device"] = "cpu"

	def __call__(self, *args, **kwds):
		"""
		L'architecture du wrapper pour une classe.
		"""
		start = time.time()
		# Exécute la fonction stockée dans l'instance.
		result= self.function(*args, **kwds)
		end= time.time()

		# Mise à jour l'état de l'instance
		stats = _STATS_REGISTRY[self._stats_key]
		stats["period"] += (end-start)
		stats["calls"] += 1

		self._detect_device(*args, **kwds)

		return result
	
	def print_stats(self):
		"""
		Affiche les stats.
		"""
		stats = _STATS_REGISTRY[self._stats_key]
		if stats["printed"]:
			return

		avg= stats["period"] / stats["calls"] if stats["calls"] > 0 else 0
		print (
			f"Chrono Stats poour '\033[34m{self.function.__name__}\033[0m':\n"
			f"	- Temps total: \033[36m{stats["period"]:.6f}\033[0m s\n"
			f"	- Appels: \033[36m{stats["calls"]}\033[0m\n"
			f"	- Temps moyen/appel: \033[36m{avg:.6f}\033[0m\n"
			f"	- Device: \033[36m{stats["device"]}\033[0m"
		)
		stats["printed"]= True


	def reset_stats(self):
		"""
		Réinitialise les stats et le flag (pour un nouveau run)
		"""
		_STATS_REGISTRY[self._stats_key] = {
			"period": 0.0,
			"calls": 0,
			"device": "cpu",
			"printed":  False
		}
		self.period= 0.0
		self.calls= 0
		self._stats_printed= False
