from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.config import load_and_validate  # noqa: E402
from shield.registry import production_promotion_decision  # noqa: E402

PRODUCTION_ALIAS = "production"
STAGING_ALIAS = "staging"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promozione staging → production nel Model Registry."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="config.toml dell'esperimento (fornisce [registry].model_name e [mlflow].tracking_uri).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Nome del registered model (alternativa a --config).",
    )
    parser.add_argument(
        "--tracking-uri", default=None, help="Override del tracking URI MLflow."
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Versione da promuovere (default: quella con alias @staging).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Promuovi anche se il gate d'accettazione era 'hold'.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Mostra la decisione senza applicarla."
    )
    return parser.parse_args()


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PROJECT_ROOT / path


def _version_by_alias(client, model_name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except Exception:
        return None


def main() -> int:
    args = parse_args()

    model_name = args.model_name
    tracking_uri = args.tracking_uri
    if args.config is not None:
        config = load_and_validate(_resolve(args.config))
        registry = config.get("registry") or {}
        model_name = model_name or registry.get("model_name")
        mlflow_cfg = config.get("mlflow", {})
        tracking_uri = tracking_uri or mlflow_cfg.get("tracking_uri")
    if not model_name:
        raise SystemExit(
            "[promote] serve --model-name oppure --config con la sezione [registry]."
        )

    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    target = (
        str(args.version)
        if args.version
        else _version_by_alias(client, model_name, STAGING_ALIAS)
    )
    if target is None:
        raise SystemExit(
            f"[promote] nessun alias @{STAGING_ALIAS} su '{model_name}' e nessuna --version: "
            f"registra prima un modello (scripts/mlflow/register_model.py) o indica la versione."
        )
    try:
        target_tags = dict(client.get_model_version(model_name, target).tags)
    except Exception as exc:
        raise SystemExit(
            f"[promote] versione {target} di '{model_name}' non trovata: {exc}"
        ) from exc

    production = _version_by_alias(client, model_name, PRODUCTION_ALIAS)

    action, reason = production_promotion_decision(
        target, target_tags, production, force=args.force
    )
    print(f"[promote] decisione: {action.upper()} — {reason}")
    if action != "promote":
        return 0 if action == "skip" else 1
    if args.dry_run:
        print("[promote] dry-run: nessuna modifica applicata.")
        return 0

    staging = _version_by_alias(client, model_name, STAGING_ALIAS)

    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, target)
    client.set_model_version_tag(
        model_name, target, "lifecycle_stage", PRODUCTION_ALIAS
    )
    print(f"[promote] {model_name} v{target} → alias @{PRODUCTION_ALIAS}")

    if production is not None and production != target:
        client.set_model_version_tag(
            model_name, production, "lifecycle_stage", "archived"
        )
        print(
            f"[promote] {model_name} v{production} (production precedente) → archived"
        )

    if staging == target:
        client.delete_registered_model_alias(model_name, STAGING_ALIAS)
        print(f"[promote] alias @{STAGING_ALIAS} rimosso da v{target} (graduata).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
