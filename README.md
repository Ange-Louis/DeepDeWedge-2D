# DeepDeWedge-2D

> **Optimisation des reconstructions tomographiques par IA : évaluation d'une approche 2D pour réduire la complexité calculatoire**

Ce projet est une implémentation 2D inspirée de [DeepDeWedge](https://github.com/MLI-lab/DeepDeWedge.git), développé dans le cadre d'un stage de recherche. L'objectif principal est d'optimiser les reconstructions tomographiques en utilisant des méthodes d'apprentissage profond pour traiter le problème du *missing wedge* (coin manquant) tout en réduisant significativement la complexité calculatoire par rapport aux approches 3D.

## À propos

DeepDeWedge-2D s'inspire directement des travaux de :

> Simon Wiedemann and Reinhard Heckel. **A deep learning method for simultaneous denoising and missing wedge reconstruction in cryogenic electron tomography.** *Nature Communications*, 15(1):8255, 2024.  
> [DOI:10.1038/s41467-024-46582-2](https://doi.org/10.1038/s41467-024-46582-2)  
> [GitHub original : MLI-lab/DeepDeWedge](https://github.com/MLI-lab/DeepDeWedge.git)

### Objectifs du projet

- **Réduction de la complexité calculatoire** : Passer d'une approche 3D à une approche 2D pour accélérer le traitement
- **Reconstruction du *missing wedge*** : Corriger les artefacts causés par les angles d'acquisition limités en tomographie
- **Débruitage simultané** : Améliorer la qualité des reconstructions tout en réduisant le bruit
- **Accessibilité** : Rendre les méthodes avancées de reconstruction accessibles avec des ressources computationnelles limitées

---

## Prérequis

- **Python** ≥ 3.12
- [**uv**](https://github.com/astral-sh/uv) (gestionnaire de dépendances et environnement)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Ange-Louis/DeepDeWedge-2D.git
cd DeepDeWedge-2D
```

### 2. Installer l'environnement et les dépendances

```bash
uv sync
```

C'est tout ! L'environnement virtuel `.venv` est créé automatiquement avec toutes les dépendances.

---

## 🚀 Vérification et utilisation

### Activer l'environnement

```bash
source .venv/bin/activate
```

### Vérifier que l'installation fonctionne

```bash
ddw --help
```

Si la commande affiche l'aide de DeepDeWedge-2D, l'installation est réussie !

---


### Structure du projet

```
DeepDeWedge-2D/
├── src/
│   └── ddw/
│       ├── __init__.py
│       ├── app.py              # Point d'entrée principal
│       ├── prepare_data2.py    # Préparation des données tomographiques
│       ├── fit_model2.py       # Entraînement du modèle 2D
│       ├── refine_tomogram2.py # Correction tomographique (missing wedge)
│       └── utils/              # Fonctions utilitaires
├── tutorial/                  # Exemples et tutoriels
├── tests/                     # Tests unitaires
└── pyproject.toml             # Configuration du projet
```

---

## Performances

L'approche 2D de DeepDeWedge-2D est à revoir, les résultats ne sont pas encore concluants.

---

*© 2024-2026 Ange-Louis Sammarcelli. Tous droits réservés.*
