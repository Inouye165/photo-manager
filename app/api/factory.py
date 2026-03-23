"""Flask application factory.

Creates and configures the Flask app, wires up all services, and
registers route blueprints.  ``serve.py`` calls ``create_app()`` to boot.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask

from app.config.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> Flask:
    """Build the fully-wired Flask application."""
    cfg = settings or get_settings()

    # ── Logging ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
    )
    log = logging.getLogger("photointel")

    # ── Flask ────────────────────────────────────────────────────────────
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_upload_mb * 1024 * 1024
    app.secret_key = cfg.secret_key

    # Store settings on the app for easy access in routes
    app.config["SETTINGS"] = cfg

    # ── Ensure workspace dirs ────────────────────────────────────────────
    for d in (cfg.vault_root, cfg.derivatives_dir, cfg.vector_db_dir,
              cfg.metadata_dir, cfg.crops_dir, cfg.logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Build services ───────────────────────────────────────────────────
    from app.catalog.registry import AssetRegistry
    from app.identity.manager import IdentityManager
    from app.search.engine import SearchEngine
    from app.search.vector_store import build_index
    from app.storage.derivatives import DerivativeStore
    from app.storage.vault import Vault
    from app.vision.detector import Detector
    from app.vision.embedder import Embedder

    vault = Vault(cfg.vault_root)
    registry = AssetRegistry(cfg.metadata_dir / "asset_registry.json")
    derivatives = DerivativeStore(cfg.derivatives_dir, cfg.crops_dir)
    identity_mgr = IdentityManager(cfg.metadata_dir / "labels.json")
    detector = Detector(
        model_path=cfg.detector_model,
        conf_person=cfg.detector_confidence_person,
        conf_animal=cfg.detector_confidence_animal,
        conf_general=cfg.detector_confidence_general,
    )
    embedder = Embedder(
        model_name=cfg.embedding_model,
        dim=cfg.embedding_dim,
        device=cfg.embedding_device,
    )
    scene_index = build_index(
        collection_name=cfg.qdrant_scene_collection,
        dim=cfg.embedding_dim,
        provider=cfg.vector_provider,
        location=cfg.vector_db_dir / "scene",
        host=cfg.qdrant_host,
        port=cfg.qdrant_port,
    )
    region_index = build_index(
        collection_name=cfg.qdrant_region_collection,
        dim=cfg.embedding_dim,
        provider=cfg.vector_provider,
        location=cfg.vector_db_dir / "region",
        host=cfg.qdrant_host,
        port=cfg.qdrant_port,
    )
    search_engine = SearchEngine(
        embedder=embedder,
        scene_index=scene_index,
        region_index=region_index,
    )

    # Attach services so blueprints can grab them via current_app
    app.config["VAULT"] = vault
    app.config["REGISTRY"] = registry
    app.config["DERIVATIVES"] = derivatives
    app.config["IDENTITY"] = identity_mgr
    app.config["DETECTOR"] = detector
    app.config["EMBEDDER"] = embedder
    app.config["SEARCH_ENGINE"] = search_engine

    # ── Register blueprints ──────────────────────────────────────────────
    from app.api.routes_assets import bp as assets_bp
    from app.api.routes_search import bp as search_bp
    from app.api.routes_identity import bp as identity_bp

    app.register_blueprint(assets_bp)
    app.register_blueprint(search_bp)
    app.register_blueprint(identity_bp)

    # ── Optional scan-on-startup ─────────────────────────────────────────
    if cfg.scan_on_startup:
        with app.app_context():
            from app.catalog.scanner import scan
            log.info("Running startup scan of %s", cfg.resolved_scan_roots)
            result = scan(cfg.resolved_scan_roots, registry)
            log.info(
                "Scan complete: %d discovered, %d new, %d updated",
                result.discovered, result.created, result.updated,
            )

    log.info("PhotoIntel ready — %d assets in registry", len(registry))
    return app
