"""Check a deployed copy of the API from the outside, as a user would.

    python scripts/check_live.py https://<your-app>.vercel.app

Checks the demo page, the documentation, health, the model card, a prediction, a batch,
agreement with the exported model on real validation bookings, the feedback loop and a few
deliberately bad requests. Writes reports/metrics/pipeline/live_check.json and exits with a
non-zero status if any check fails. Needs a finished pipeline run (for the exported model).
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hotel_cancellation.config import load_config  # noqa: E402
from hotel_cancellation.export import service_style_proba  # noqa: E402
from hotel_cancellation.monitoring.latency import to_api_payload  # noqa: E402

EXAMPLE = {
    "hotel": "City Hotel", "lead_time": 120, "arrival_date": "2026-08-14", "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 3, "adults": 2, "children": 0, "babies": 0, "meal": "BB",
    "market_segment": "Online TA", "distribution_channel": "TA/TO", "is_repeated_guest": False,
    "previous_cancellations": 0, "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
    "deposit_type": "No Deposit", "agent": 9, "company": None, "customer_type": "Transient", "adr": 115.0,
}


def call(base: str, path: str, body=None, raw: bytes | None = None, content_type: str = "application/json",
         method: str | None = None) -> tuple[int, object]:
    """Status code and parsed body (JSON when possible, else text). HTTP errors are returned, not raised."""
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data, method=method or ("GET" if data is None else "POST"),
                                 headers={"Content-Type": content_type, "User-Agent": "live-check/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status, payload = resp.status, resp.read()
    except urllib.error.HTTPError as err:
        status, payload = err.code, err.read()
    try:
        return status, json.loads(payload)
    except ValueError:
        return status, payload.decode("utf-8", "replace")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    base = sys.argv[1].rstrip("/")
    cfg = load_config(ROOT / "configs" / "config.yaml")
    export = ROOT / "artifacts" / "export"
    metadata = json.loads((export / "metadata.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    status, page = call(base, "/")
    check("demo page loads", status == 200 and "cancelled" in str(page).lower(), f"GET / -> {status}")
    status, docs = call(base, "/docs")
    check("API documentation loads", status == 200 and "swagger" in str(docs).lower(), f"GET /docs -> {status}")
    status, health = call(base, "/health")
    healthy = status == 200 and isinstance(health, dict) and health.get("status") == "ok"
    check("health", healthy, f"GET /health -> {status} {health if healthy else str(health)[:200]}")
    version = health.get("model_version") if isinstance(health, dict) else None
    check("serves the exported model", version == metadata["model_version"],
          f"live {version}, exported {metadata['model_version']}")
    status, card = call(base, "/api/v1/model")
    check("model card", status == 200 and abs(card["threshold"] - metadata["threshold"]) < 1e-12
          and card["test_metrics"] == metadata["test_metrics"],
          f"threshold {card.get('threshold') if isinstance(card, dict) else '?'}, test metrics match the export")
    status, pred = call(base, "/api/v1/predict", EXAMPLE)
    check("prediction", status == 200 and 0 <= pred["cancellation_probability"] <= 1,
          f"example booking -> {pred.get('cancellation_probability') if isinstance(pred, dict) else pred}")

    rows = pd.read_pickle(export / "parity_rows.pkl").head(100)
    offline = service_style_proba(rows, metadata, export / "model.onnx")
    status, batch = call(base, "/api/v1/predict/batch", {"bookings": [to_api_payload(r) for r in rows.to_dict("records")]})
    live = np.array([p["cancellation_probability"] for p in batch["predictions"]]) if status == 200 else np.array([])
    diff = float(np.max(np.abs(live - offline))) if len(live) == len(offline) else float("nan")
    check("batch of 100 real bookings agrees with the exported model", status == 200 and diff <= 1e-4,
          f"largest difference {diff:.1e} (limit 1e-4)")

    status, ack = call(base, "/api/v1/feedback", {
        "prediction_id": pred.get("prediction_id", "x") if isinstance(pred, dict) else "x", "was_cancelled": False,
        "cancellation_probability": pred.get("cancellation_probability") if isinstance(pred, dict) else None,
        "will_cancel": pred.get("will_cancel") if isinstance(pred, dict) else None})
    check("feedback accepted", status == 200 and ack.get("received") is True, f"stored: {ack.get('stored') if isinstance(ack, dict) else ack}")
    status, summary = call(base, "/api/v1/feedback/summary")
    check("feedback summary", status == 200 and summary.get("records", 0) >= 1,
          f"{summary.get('records') if isinstance(summary, dict) else '?'} record(s), storage: "
          f"{summary.get('storage') if isinstance(summary, dict) else '?'}")

    bad = [
        ("malformed JSON", {"raw": b'{"hotel": "City Hotel",'}, 422),
        ("empty body", {"raw": b""}, 422),
        ("wrong content type", {"raw": b"hotel=City", "content_type": "text/plain"}, 415),
        ("missing fields", {"body": {"hotel": "City Hotel"}}, 422),
        ("negative lead time", {"body": {**EXAMPLE, "lead_time": -5}}, 422),
        ("impossible date", {"body": {**EXAMPLE, "arrival_date": "2026-02-30"}}, 422),
        ("zero guests", {"body": {**EXAMPLE, "adults": 0}}, 422),
        ("excluded column sent", {"body": {**EXAMPLE, "country": "PRT"}}, 422),
        ("text where a number belongs", {"body": {**EXAMPLE, "adr": "cheap"}}, 422),
    ]
    for name, kwargs, expected in bad:
        status, reply = call(base, "/api/v1/predict", **kwargs)
        message = reply.get("message", "") if isinstance(reply, dict) else str(reply)[:80]
        check(f"bad input: {name}", status == expected and bool(message), f"{status}: {message[:110]}")
    status, reply = call(base, "/api/v1/predict/batch", {"bookings": [EXAMPLE] * 101})
    check("bad input: batch of 101", status == 422, f"{status}")
    status, reply = call(base, "/no-such-page")
    check("unknown path", status == 404 and isinstance(reply, dict), f"{status}")
    status, reply = call(base, "/api/v1/predict")
    check("wrong method", status == 405, f"{status}")
    status, reply = call(base, "/api/v1/predict", {**EXAMPLE, "agent": 99999})
    check("unknown agent is scored and reported", status == 200 and reply.get("unseen_categories") == ["agent_id"],
          f"{status}: unseen_categories={reply.get('unseen_categories') if isinstance(reply, dict) else '?'}")

    failed = [c for c in checks if not c["ok"]]
    result = {"url": base, "checked_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "model_version": version, "feedback_storage": health.get("feedback_storage") if isinstance(health, dict) else None,
              "parity_rows": int(len(rows)), "parity_max_abs_diff": diff, "checks": checks,
              "passed": len(checks) - len(failed), "failed": len(failed), "configured_url": cfg.service.url}
    out = ROOT / "reports" / "metrics" / "pipeline" / "live_check.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n{result['passed']} of {len(checks)} checks passed. Wrote {out.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
