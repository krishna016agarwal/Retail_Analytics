#!/usr/bin/env python3
"""Retail Inventory API — Launcher (Step 10).

Starts the standalone FastAPI inventory server on port 8001.
Does NOT import or start the crowd/queue pipeline (src/, main.py).

Usage:
    python run_inventory_api.py [--host HOST] [--port PORT] [--reload]

Examples:
    python run_inventory_api.py
    python run_inventory_api.py --port 8002 --reload
"""

import argparse
import pathlib
import sys

# Ensure project root is on path
ROOT_DIR = pathlib.Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(description="Retail Inventory API Server")
    parser.add_argument("--host",   default="0.0.0.0",     help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8001, type=int, help="Port (default: 8001)")
    parser.add_argument("--reload", action="store_true",    help="Enable hot-reload (development only)")
    parser.add_argument(
        "--workers", default=1, type=int,
        help="Number of workers (default: 1; keep at 1 on CPU to avoid pipeline contention)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Retail Inventory API — Step 10")
    print("=" * 70)
    print(f"  Host:     {args.host}:{args.port}")
    print(f"  Reload:   {args.reload}")
    print(f"  Workers:  {args.workers}")
    print()
    print("  Endpoints:")
    print(f"    GET  http://localhost:{args.port}/inventory/health")
    print(f"    GET  http://localhost:{args.port}/inventory/report")
    print(f"    POST http://localhost:{args.port}/inventory/run")
    print(f"    GET  http://localhost:{args.port}/inventory/report/text")
    print(f"    GET  http://localhost:{args.port}/docs  (Swagger UI)")
    print()
    print("  NOTE: Pipeline is CPU-bound (~5-6 FPS). /inventory/run runs")
    print("        in background; poll /inventory/report for updated data.")
    print("=" * 70)

    import uvicorn
    uvicorn.run(
        "inventory.inventory_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
