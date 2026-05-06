"""Valide localement le runtime final a partir des artefacts deployables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_utils import ensure_paths_exist, relative_to_project
from scripts.prediction_adjustment import AdjustedYieldService


RUNTIME_REQUIRED_ARTIFACTS = [
    Path("artifacts/models/p1_historical_pipeline.joblib"),
    Path("artifacts/models/p1_historical_metadata.json"),
    Path("artifacts/models/p23_simulation_pipeline.joblib"),
    Path("artifacts/models/p23_simulation_metadata.json"),
    Path("artifacts/experiments/experience_1/dataset_consolide_historique_colonnes.csv"),
]


def parse_args() -> argparse.Namespace:
    """Construit l'interface en ligne de commande du validateur runtime."""
    parser = argparse.ArgumentParser(
        description="Run a local smoke test against the final adjusted-yield service.",
    )
    parser.add_argument(
        "--country",
        help="Optional country used for the smoke test. Defaults to the first available area.",
    )
    parser.add_argument(
        "--crop",
        help="Optional crop used for the smoke test. Must belong to the selected country.",
    )
    parser.add_argument(
        "--max-candidate-crops",
        type=int,
        default=3,
        help="Maximum number of crops included in the recommendation smoke test.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validation summary as JSON.",
    )
    return parser.parse_args()


def pick_area_and_crop(
    service: AdjustedYieldService,
    *,
    country: str | None = None,
    crop: str | None = None,
) -> tuple[str, str]:
    """Selectionne un couple pays/culture valide pour le smoke test.

    Args:
        service: Service metier deja initialise.
        country: Pays optionnel impose.
        crop: Culture optionnelle imposee.

    Returns:
        tuple[str, str]: Couple pays/culture compatible avec le moteur final.
    """
    selected_country = country or service.available_areas[0]
    if selected_country not in service.crops_by_area:
        raise ValueError(f"Unknown country for runtime validation: {selected_country}")

    available_crops = service.crops_by_area[selected_country]
    selected_crop = crop or available_crops[0]
    if selected_crop not in available_crops:
        raise ValueError(
            f"Crop {selected_crop!r} is not available for country {selected_country!r}."
        )

    return selected_country, selected_crop


def _pick_distinct_option(options: list[str], current_value: Any) -> Any:
    """Choisit une valeur differente de la reference si possible."""
    for option in options:
        if option != current_value:
            return option
    return current_value


def build_smoke_user_conditions(
    service: AdjustedYieldService,
    *,
    reference_profile: dict[str, Any],
) -> dict[str, Any]:
    """Construit des conditions utilisateur legerement differentes de la reference.

    Args:
        service: Service metier expose par l'application finale.
        reference_profile: Profil de reference retourne par le baseline.

    Returns:
        dict[str, Any]: Conditions candidates pour un smoke test realiste.
    """
    return {
        "region": _pick_distinct_option(service.simulation_options["regions"], reference_profile["region"]),
        "soil_type": _pick_distinct_option(service.simulation_options["soil_types"], reference_profile["soil_type"]),
        "rainfall_mm": float(reference_profile["rainfall_mm"]) + 25.0,
        "temperature_celsius": float(reference_profile["temperature_celsius"]) + 1.5,
        "fertilizer_used": not bool(reference_profile["fertilizer_used"]),
        "irrigation_used": not bool(reference_profile["irrigation_used"]),
        "weather_condition": _pick_distinct_option(
            service.simulation_options["weather_conditions"],
            reference_profile["weather_condition"],
        ),
        "days_to_harvest": max(1.0, float(reference_profile["days_to_harvest"]) + 7.0),
    }


def validate_runtime(
    *,
    country: str | None = None,
    crop: str | None = None,
    max_candidate_crops: int = 3,
) -> dict[str, Any]:
    """Execute un smoke test complet sur la pile metier finale.

    Args:
        country: Pays optionnel impose.
        crop: Culture optionnelle imposee.
        max_candidate_crops: Nombre maximum de cultures a comparer.

    Returns:
        dict[str, Any]: Resume du test et des artefacts verifies.
    """
    ensure_paths_exist(RUNTIME_REQUIRED_ARTIFACTS, label="runtime artifacts")
    service = AdjustedYieldService()
    selected_country, selected_crop = pick_area_and_crop(service, country=country, crop=crop)

    baseline = service.get_baseline(selected_country, selected_crop)
    smoke_conditions = build_smoke_user_conditions(
        service,
        reference_profile=baseline["reference_profile"],
    )
    prediction = service.predict_adjusted_yield(selected_country, selected_crop, smoke_conditions)

    candidate_crops = service.crops_by_area[selected_country][: max(1, max_candidate_crops)]
    recommendations = service.recommend_crops(
        selected_country,
        smoke_conditions,
        candidate_crops=candidate_crops,
    )
    if recommendations.empty:
        raise RuntimeError("Runtime validation produced no recommendations.")

    top_recommendation = recommendations.iloc[0]
    return {
        "country": selected_country,
        "crop": selected_crop,
        "target_year": baseline["target_year"],
        "candidate_crop_count": int(len(candidate_crops)),
        "baseline_prediction": float(baseline["p1_historical_prediction"]),
        "final_prediction": float(prediction["final_prediction"]),
        "top_recommendation": str(top_recommendation["crop"]),
        "top_recommendation_prediction": float(top_recommendation["final_prediction"]),
        "validated_artifacts": [relative_to_project(path) for path in RUNTIME_REQUIRED_ARTIFACTS],
    }


def main() -> None:
    """Execute le validateur runtime depuis la CLI."""
    args = parse_args()
    summary = validate_runtime(
        country=args.country,
        crop=args.crop,
        max_candidate_crops=args.max_candidate_crops,
    )
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return

    print(
        "[runtime] Validation passed "
        f"(country={summary['country']}, crop={summary['crop']}, "
        f"final_prediction={summary['final_prediction']:.4f}, "
        f"top_recommendation={summary['top_recommendation']})"
    )


if __name__ == "__main__":
    main()
