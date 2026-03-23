#!/usr/bin/env python3
"""Production server entrypoint for PhotoIntel.

Boots the new ``app.api.factory`` application and serves it via waitress
(multi-threaded).  Falls back to Flask dev server when waitress is absent.

Legacy ``PHOTOFINDER_*`` env vars are still respected for backwards compat,
but the canonical config source is now ``.env`` / ``PHOTOINTEL_*``.
"""

import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def main() -> None:
    from app.api.factory import create_app
    from app.config.settings import get_settings

    cfg = get_settings()
    application = create_app(cfg)

    host = cfg.host
    port = cfg.port
    threads = cfg.workers

    try:
        from waitress import serve

        print(f"PhotoIntel ready -> http://{host}:{port}")
        print(f"  Server:   waitress ({threads} threads)")
        print(f"  Vault:    {cfg.vault_root}")
        print(f"  Scan:     {cfg.scan_roots}")
        print(f"  Log:      {cfg.log_level}")
        serve(application, host=host, port=port, threads=threads)
    except ImportError:
        print("waitress not installed — falling back to Flask dev server")
        print(f"  http://{host}:{port}")
        application.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
