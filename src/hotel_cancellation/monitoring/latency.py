"""Replay real bookings against a running API and measure latency and agreement."""

from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pandas as pd

MONTH_INDEX = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
     "November", "December"], start=1)}


def to_api_payload(row: dict) -> dict:
    """Convert one cleaned data row into the JSON body ``/api/v1/predict`` expects."""
    def opt_int(v):
        return None if v is None or (isinstance(v, float) and v != v) else int(v)

    month = MONTH_INDEX[row["arrival_date_month"]]
    return {
        "hotel": row["hotel"], "lead_time": int(row["lead_time"]),
        "arrival_date": f"{int(row['arrival_date_year']):04d}-{month:02d}-{int(row['arrival_date_day_of_month']):02d}",
        "stays_in_weekend_nights": int(row["stays_in_weekend_nights"]),
        "stays_in_week_nights": int(row["stays_in_week_nights"]),
        "adults": int(row["adults"]), "children": int(row["children"]), "babies": int(row["babies"]),
        "meal": row["meal"], "market_segment": row["market_segment"],
        "distribution_channel": row["distribution_channel"],
        "is_repeated_guest": bool(row["is_repeated_guest"]),
        "previous_cancellations": int(row["previous_cancellations"]),
        "previous_bookings_not_canceled": int(row["previous_bookings_not_canceled"]),
        "reserved_room_type": row["reserved_room_type"], "deposit_type": row["deposit_type"],
        "agent": opt_int(row.get("agent")), "company": opt_int(row.get("company")),
        "customer_type": row["customer_type"], "adr": float(row["adr"]),
    }


def post_json(url: str, body: dict, timeout: float = 30.0) -> tuple[dict, float, float | None]:
    """POST ``body``; return the parsed response, client latency (ms) and server time (ms)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
        server = resp.headers.get("X-Process-Time-ms")
    return payload, (time.perf_counter() - start) * 1000, float(server) if server else None


def replay_requests(base_url: str, rows: pd.DataFrame, expected_proba: np.ndarray | None = None) -> dict:
    """Send each row as one request; the first request is reported separately (cold start)."""
    url = base_url.rstrip("/") + "/api/v1/predict"
    client, server, diffs, errors = [], [], [], 0
    for i, row in enumerate(rows.to_dict("records")):
        try:
            payload, ms, srv = post_json(url, to_api_payload(row))
        except Exception:  # noqa: BLE001 - count failures, keep measuring
            errors += 1
            continue
        client.append(ms)
        if srv is not None:
            server.append(srv)
        if expected_proba is not None:
            diffs.append(abs(payload["cancellation_probability"] - float(expected_proba[i])))
    if not client:
        raise RuntimeError(f"no successful requests to {url}")
    warm = np.array(client[1:]) if len(client) > 1 else np.array(client)
    return {
        "url": url, "requests": len(rows), "errors": errors, "error_rate": errors / len(rows),
        "first_request_ms": float(client[0]),
        "warm_p50_ms": float(np.percentile(warm, 50)), "warm_p95_ms": float(np.percentile(warm, 95)),
        "warm_p99_ms": float(np.percentile(warm, 99)),
        "server_p50_ms": float(np.percentile(server, 50)) if server else None,
        "max_abs_diff_vs_offline": float(max(diffs)) if diffs else None,
        "client_latencies_ms": [float(v) for v in client],
    }
