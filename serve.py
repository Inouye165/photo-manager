#!/usr/bin/env python3
"""Production local server for PhotoFinder using waitress (multi-threaded)."""
import os
import sys

# Ensure print output is visible in log files (no buffering)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from src.runtime_paths import resolve_runtime_directories


def main():
    input_dir = os.environ.get("PHOTOFINDER_INPUT", "working_dir")
    output_dir = os.environ.get("PHOTOFINDER_OUTPUT", "working_dir")
    host = os.environ.get("PHOTOFINDER_HOST", "127.0.0.1")
    port = int(os.environ.get("PHOTOFINDER_PORT", "5000"))
    threads = int(os.environ.get("PHOTOFINDER_THREADS", "8"))

    for d in (input_dir, output_dir):
        if not os.path.isdir(d):
            print(f"Error: directory '{d}' does not exist.")
            sys.exit(1)

    runtime_dirs = resolve_runtime_directories(input_dir, output_dir)

    # web_ui.py reads sys.argv inside route handlers
    sys.argv = ["serve.py", str(runtime_dirs.source_dir), str(runtime_dirs.data_dir)]

    from web_ui import app  # noqa: E402  — import after argv is set

    try:
        from waitress import serve

        print(f"PhotoFinder ready -> http://{host}:{port}")
        print(f"  Server:  waitress ({threads} threads)")
        print(f"  Source:  {runtime_dirs.source_dir}")
        print(f"  Output:  {runtime_dirs.data_dir}")
        if runtime_dirs.legacy_source_mode:
            print("  Mode:    legacy single-folder startup detected; uploads redirected into a separate source vault")
            if runtime_dirs.migrated_files:
                print(f"  Migrated legacy root photos: {len(runtime_dirs.migrated_files)}")
        serve(app, host=host, port=port, threads=threads)
    except ImportError:
        print("waitress not installed -- falling back to Flask dev server (1 thread)")
        print(f"  http://{host}:{port}")
        app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
