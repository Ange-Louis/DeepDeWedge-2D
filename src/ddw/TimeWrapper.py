import time
from functools import update_wrapper

try:
	import torch
	HAS_TORCH= True
except:
	HAS_TORCH= False


class Chrono:
	def __init__(self, function):
		# Sauvegarde la fonction initiale
		self.function= function
		
		# Initialise les variables d'état
		self.period= 0.0
		self.calls= 0
		self.device= "cpu" # Par défaut

		# Remplace @wraps pour une classe : copie de l'identité de la fonction
		update_wrapper(self, function)

	def _detect_device(self, *args, **kwds):
		"""
		Détecte si l'exéctution se fait sur CPU ou GPU
		"""
		if not HAS_TORCH:
			self.device= "cpu (PyTorch non installé)"
			return
		
		# Vérifie les tenseurs dans les arguments 
		for item in args + tuple(kwds.values()):
			if isinstance(item, torch.Tensor):
				self.device = str(item.device)
				return
			
		# Si aucun tenseur dans les args, vérifie si CUDA est utilisé
		if torch.cuda.is_available():
			# Vérifie si des tenseurs ont été créés sur GPU dans la fonction
			self.device = "cuda (disponible, mais tenseurs non détectés dans les args)"
		else: 
			self.device = "cpu"

	def __call__(self, *args, **kwds):
		"""
		L'architecture du wrapper pour une classe.
		"""
		start = time.time()
		# Exécute la fonction stockée dans l'instance.
		result= self.function(*args, **kwds)
		end= time.time()

		# Mise à jour l'état de l'instance
		self.period += (end-start)
		self.calls += 1

		self._detect_device(*args, **kwds)

		return result
	
	def print_stats(self):
		"""
		Affiche les stats.
		"""
		avg= self.period / self.calls if self.calls > 0 else 0
		print (
			f"Chrono Stats poour '\033[34m{self.function.__name__}\033[0m':\n"
			f"	- Temps total: \033[36m{self.period:.6f}\033[0m s\n"
			f"	- Appels: \033[36m{self.calls}\033[0m\n"
			f"	- Temps moyen/appel: \033[36m{avg}\033[0m\n"
			f"	- Device: \033[36m{self.device}\033[0m"
		)
