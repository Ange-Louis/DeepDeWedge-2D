import pytest
import torch
from pathlib import Path
from unittest.mock import patch
# =============================================================================
# L'utilisation de unittest.mock.patch permet de pallier les contraintes liées
# aux dépendances externes d'une fonction (accès aux fichiers, calculs
# intensifs, etc.). Dans le cas de prepare_data, cette méthode est essentielle
# pour les raisons suivantes :
# - Isolation du système de fichiers : La fonction appelle initialement 
# load_data() pour lire des fichiers réels. Le "mocking" permet de 
# s'affranchir de la présence de ces fichiers sur le disque.
# - Interception des appels : Le décorateur @patch remplace temporairement les
# fonctions réelles (load_data, extract_subtomos) par des objets simulés
# appelés "Mocks".
# - Contrôle du flux : Lorsqu'un appel est effectué durant le test, Python
# redirige l'exécution vers le Mock au lieu de la fonction d'origine,
# permettant ainsi de tester la logique interne sans exécuter les processus
# lourds ou dépendants de l'environnement.
# =============================================================================

from ddw.prepare_data2 import prepare_data, setup_subtomo_dir


@patch ('ddw.prepare_data2.extract_subtomos')
@patch ('ddw.prepare_data2.load_data')
def test_prepare_data_2D_Image(mock_load_data, mock_extract_subtomos, tmp_path):
	"""
	For a 2D Image:
	Test that the `prepare_data` function extracts, splits (train/val sets) and
    saves the 2D subtomograms correctly to the appropriate folders. 
	"""

	# --- 1. *args PREPARATION ---

	mock_load_data.return_value = torch.zeros((100, 100))

	nb_fake_subtomos = 10
	fake_subtomos = [torch.zeros((10, 10)) for _ in range(nb_fake_subtomos)]
	fake_coords = [(0, 0) for _ in range(nb_fake_subtomos)]
	mock_extract_subtomos.return_value = (fake_subtomos, fake_coords)

	fake_pict0_files = [Path("fake_pict0.png")]
	fake_pict1_files = [Path("fake_pict1.png")]


	# --- 2. FUNCTION EXECUTION ---

	prepare_data(
		tomo0_files=fake_pict0_files,
		tomo1_files=fake_pict1_files,
		subtomo_size=10,
		val_fraction=0.2,
		project_dir=tmp_path,
		seed=42,
		verbose=False
	)


	# --- 3. ASSERTS ---
	
	# Check that `load_data` & `extract_subtomos` have been called
	assert mock_load_data.called
	assert mock_extract_subtomos.called


	# Check that the directory structure has been created
	expected_subtomo_dir = tmp_path/"subtomos"

	assert (expected_subtomo_dir / "fitting_subtomos" / "subtomo0").exists()
	assert (expected_subtomo_dir / "fitting_subtomos" / "subtomo1").exists()
	assert (expected_subtomo_dir / "val_subtomos" / "subtomo0").exists()
	assert (expected_subtomo_dir / "val_subtomos" / "subtomo1").exists()


	# Check that the correct number of subtomos files has been generated
	train_files_0 = list((expected_subtomo_dir / "fitting_subtomos" / "subtomo0").glob("*.pt"))
	val_files_0 = list((expected_subtomo_dir / "val_subtomos" / "subtomo0").glob("*.pt"))

	assert len(train_files_0) == 8, f"Expected 8 training files, obtained {len(train_files_0)}"
	assert len(val_files_0) == 2, f"Expected 2 training files, obtained {len(val_files_0)}"

	