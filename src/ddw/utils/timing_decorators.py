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