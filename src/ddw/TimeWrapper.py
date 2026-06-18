import time
from functools import update_wrapper

class Chrono:
	def __init__(self, function):
		# Sauvegarde la fonction initiale
		self.function= function
		
		# Initialise les variables d'état
		self.period= 0.0
		self.calls= 0

		# Remplace @wraps pour une classe : copie de l'identité de la fonction
		update_wrapper(self, function)

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

		return result
	
