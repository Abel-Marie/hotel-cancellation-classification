"""Thin entry point: same as `hotel-cancellation all` (pass --fast for a smoke run)."""

from hotel_cancellation.cli import main

if __name__ == "__main__":
    import sys
    main(["all", *sys.argv[1:]])
