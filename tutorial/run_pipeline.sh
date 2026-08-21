#!/bin/bash

# ==================================================================
# 1. ACTIVATION DE L'ENVIRONNEMENT UV (REMPLACE CONDA)
# ==================================================================
# On s'assure d'être dans l'environnement virtuel créé par uv
source .venv/bin/activate

# ==================================================================
# 2. CONFIGURATION DES CHEMINS ET VARIABLES
# ==================================================================
CONFIG_FILE="./tutorial/config.yaml"

# Tailles des carrés à tester (doivent être divisibles par 2^3 = 8)
SIZES=(128)

# ==================================================================
# 3. BOUCLE PRINCIPALE
# ==================================================================
for SIZE in "${SIZES[@]}"; do
    echo "=================================================================="
    echo "DÉMARRAGE DU PIPELINE 2D AVEC SUBTOMO_SIZE = $SIZE"
    echo "=================================================================="

    # Création d'un dossier de projet unique pour chaque taille
    PROJECT_DIR="./Taillebraintutorial_ddw2project_${SIZE}"

    # 1. Préparation des données (--overwrite permet d'écraser si le dossier existe déjà)
    echo ">>> 1. prepare-data (size: $SIZE)..."
    ddw prepare-data \
        --config "$CONFIG_FILE" \
        --subtomo-size "$SIZE" \
        --project-dir "$PROJECT_DIR" \
        --overwrite

    # 2. Entraînement du modèle
    echo ">>> 2. fit-model (size: $SIZE)..."
    ddw fit-model2 \
        --config "$CONFIG_FILE" \
        --subtomo-size "$SIZE" \
        --project-dir "$PROJECT_DIR"

    # 3. Trouver le meilleur checkpoint généré (val_loss la plus basse)
    echo ">>> Recherche du meilleur modèle entraîné..."
    BEST_MODEL=$(ls -t "$PROJECT_DIR"/logs/version_*/checkpoints/val_loss/*.ckpt 2>/dev/null | head -n 1)

    if [ -z "$BEST_MODEL" ]; then
        echo "ERREUR : Aucun modèle trouvé pour la taille $SIZE. L'entraînement a peut-être échoué."
        continue
    fi

    echo ">>> Modèle trouvé : $BEST_MODEL"

    # 4. Raffinement des tomogrammes
    echo ">>> 3. refine-tomogram (size: $SIZE)..."
    ddw refine-tomogram \
        --config "$CONFIG_FILE" \
        --subtomo-size "$SIZE" \
        --project-dir "$PROJECT_DIR" \
        --model-checkpoint-file "$BEST_MODEL"

    echo "Terminé pour subtomo_size = $SIZE"
    echo ""
done

echo "Toutes les tailles ont été traitées avec succès !"