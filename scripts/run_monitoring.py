"""Thin entry point: drift report, then latency against --url if given.

    python scripts/run_monitoring.py --url https://<your-app>.vercel.app
"""

import sys

from hotel_cancellation.cli import main

if __name__ == "__main__":
    args = sys.argv[1:]
    main(["monitor", "drift"])
    if "--url" in args:
        main(["monitor", "latency", *args])
