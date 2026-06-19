import torch
from src.ddw.TimeWrapper import Chrono

@Chrono
def ma_fonction(tensor):
	return tensor *2

if __name__ == '__main__':

	# Test avec un tensor CPU
	t_cpu = torch.randn(10, device="cpu")
	ma_fonction(t_cpu)
	ma_fonction.print_stats() # Affiche : "cpu"

	# Test avec un tensor GPU
	t_gpu = torch.randn(10, device="cuda")
	ma_fonction(t_gpu)
	ma_fonction.print_stats() # Affiche : "cuda"