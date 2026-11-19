"""Hotel cancellation risk API (FastAPI). Vercel loads the ``app`` object from this file.

Layout of this file: model and logging, the feedback store, the app and its
request guard, error handling (every error is JSON with the same shape), then
the endpoints.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
import urllib.request
import uuid
from collections import Counter, deque
from pathlib import Path

import numpy as np
from core.inference import CancellationModel
from core.schemas import (
    EXAMPLE_BOOKING,
    EXAMPLE_FEEDBACK,
    MAX_BATCH,
    BatchRequest,
    BatchResponse,
    Booking,
    ErrorResponse,
    Feedback,
    FeedbackAck,
    FeedbackSummary,
    PredictionResponse,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.routing import APIRoute
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException

HERE = Path(__file__).resolve().parent
DEMO_PAGE = HERE / "public" / "index.html"
MODEL = CancellationModel()

log = logging.getLogger("api")
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

STARTED = time.time()
STATS = {"requests": 0, "errors": 0, "rejected": 0, "predictions": 0, "latencies_ms": deque(maxlen=2000)}

MAX_BODY_BYTES = 1_000_000      # a full batch of 100 bookings is about 50 kB
DRAIN_LIMIT_BYTES = 20_000_000  # how much of an oversized body is read (and discarded) before answering
MAX_PROBLEMS_LISTED = 25
HINT = "See /docs for every field, its allowed values and a working example."


# ----------------------------------------------------------------------------------------------
# Feedback store
# ----------------------------------------------------------------------------------------------

STORAGE_NOTES = {
    "durable": "Kept in a Redis database, so feedback survives restarts and is shared by all instances.",
    "this instance only": "Kept in this server instance's memory: it is lost when the instance is recycled and is "
                          "not shared between instances. Set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN "
                          "to keep it durably.",
}


class FeedbackStore:
    """Keeps the most recent feedback records, newest first.

    With a Redis REST endpoint (Upstash, or Vercel's KV variables) records go to one capped
    Redis list. Without one, or whenever Redis cannot be reached, they go to a ring buffer in
    this instance's memory instead. A record holds an ID, the outcome and the prediction that
    was made; it never holds the booking itself.
    """

    KEY = "hotel-cancellation:feedback:v1"
    ENV_NAMES = (("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN"), ("KV_REST_API_URL", "KV_REST_API_TOKEN"))

    def __init__(self, url: str | None = None, token: str | None = None, max_records: int = 2000,
                 timeout_s: float = 2.0, retry_after_s: float = 30.0) -> None:
        self.url, self.token = (url.rstrip("/"), token) if url and token else (None, None)
        self.max_records, self.timeout_s, self.retry_after_s = max_records, timeout_s, retry_after_s
        self._memory: deque[dict] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._skip_redis_until = 0.0

    @classmethod
    def from_env(cls, env=None) -> FeedbackStore:
        env = os.environ if env is None else env
        for url_name, token_name in cls.ENV_NAMES:
            url, token = env.get(url_name, "").strip(), env.get(token_name, "").strip()
            if url.startswith("https://") and token:
                return cls(url=url, token=token)
        return cls()

    @property
    def kind(self) -> str:
        """Where records go when everything works."""
        return "durable" if self.url else "this instance only"

    def _redis(self, *commands: list) -> list:
        """Run Redis commands through the REST pipeline endpoint; raise on any failure."""
        request = urllib.request.Request(
            f"{self.url}/pipeline", data=json.dumps(commands).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310 - https only
            results = json.loads(response.read())
        for item in results:
            if "error" in item:
                raise RuntimeError(f"redis: {item['error']}")
        return [item.get("result") for item in results]

    def _try_redis(self, *commands: list) -> list | None:
        """The results, or None if Redis is not configured, is failing, or failed in the last few seconds."""
        if not self.url or time.monotonic() < self._skip_redis_until:
            return None
        try:
            return self._redis(*commands)
        except Exception as exc:  # noqa: BLE001 - any failure means "use memory"; never fail the request
            self._skip_redis_until = time.monotonic() + self.retry_after_s
            log.warning(json.dumps({"event": "feedback_store_unavailable", "error": type(exc).__name__,
                                    "fallback": "memory", "retry_in_s": self.retry_after_s}))
            return None

    def add(self, record: dict) -> str:
        """Store one record and say where it went: "durable" or "this instance only"."""
        value = json.dumps(record, separators=(",", ":"))
        if self._try_redis(["LPUSH", self.KEY, value], ["LTRIM", self.KEY, 0, self.max_records - 1]) is not None:
            return "durable"
        with self._lock:
            self._memory.appendleft(record)
        return "this instance only"

    def records(self) -> tuple[list[dict], str]:
        """All kept records, newest first, and where they were read from."""
        with self._lock:
            local = list(self._memory)
        results = self._try_redis(["LRANGE", self.KEY, 0, self.max_records - 1])
        if results is None:
            return local, "this instance only"
        durable = []
        for raw in results[0] or []:
            try:
                item = json.loads(raw) if isinstance(raw, str) else raw
            except ValueError:
                continue
            if isinstance(item, dict) and "id" in item and "y" in item:
                durable.append(item)
        merged = sorted(local + durable, key=lambda r: r.get("ts", 0), reverse=True) if local else durable
        return merged, "durable"


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarise_feedback(records: list[dict], threshold: float) -> dict:
    """Live performance from feedback records (newest first). One record per prediction: the latest wins."""
    latest: dict[str, dict] = {}
    for record in records:
        latest.setdefault(str(record["id"]), record)
    unique = list(latest.values())
    cancelled = sum(1 for r in unique if r["y"])
    with_probability = [r for r in unique if r.get("p") is not None]
    with_decision = [r for r in unique if r.get("d") is not None]
    tp = sum(1 for r in with_decision if r["d"] and r["y"])
    fp = sum(1 for r in with_decision if r["d"] and not r["y"])
    tn = sum(1 for r in with_decision if not r["d"] and not r["y"])
    fn = sum(1 for r in with_decision if not r["d"] and r["y"])
    return {
        "records": len(unique), "repeats_ignored": len(records) - len(unique),
        "outcomes": {"cancelled": cancelled, "arrived": len(unique) - cancelled},
        "observed_cancellation_rate": _ratio(cancelled, len(unique)),
        "calibration": {
            "records": len(with_probability),
            "observed_cancellation_rate": _ratio(sum(1 for r in with_probability if r["y"]), len(with_probability)),
            "mean_predicted_probability": _ratio(sum(float(r["p"]) for r in with_probability),
                                                 len(with_probability)),
        },
        "decisions": {
            "records": len(with_decision),
            "confusion": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
            "accuracy": _ratio(tp + tn, len(with_decision)), "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
        },
        "current_threshold": threshold,
        "model_versions": dict(Counter(str(r["v"]) for r in unique if r.get("v"))),
    }


FEEDBACK = FeedbackStore.from_env()


# ----------------------------------------------------------------------------------------------
# The app
# ----------------------------------------------------------------------------------------------

def _self_check() -> dict | None:
    """Score the documented example once at start-up: proves model and schema fit, and feeds the docs."""
    try:
        return {**MODEL.predict([Booking(**EXAMPLE_BOOKING).to_record()])[0],
                "prediction_id": EXAMPLE_FEEDBACK["prediction_id"]}
    except Exception:  # noqa: BLE001 - reported by /health instead of taking the whole service down
        log.exception(json.dumps({"event": "self_check_failed", "model_version": MODEL.version}))
        return None


EXAMPLE_PREDICTION = _self_check()

DESCRIPTION = f"""
Estimates the probability that a hotel booking will be **cancelled before arrival**, using only what the hotel
knows on the day the booking is made.

**Try it without writing code:** the [demo page](/) fills in a booking for you.

### Intended use
A hotel's revenue team scores each new booking, then plans overbooking or asks for a deposit when the estimate
reaches the decision threshold (currently {MODEL.threshold:g}; see `/api/v1/model`).

### Limits
* This is a **course project demo** for an MSc machine-learning assignment, not a production service.
* The model was trained on two hotels in Portugal ({MODEL.metadata.get("data", {}).get("name", "Hotel Booking Demand")}
  dataset, arrivals 2015 to 2017). Other hotels, countries or years may behave differently.
* Columns that are only filled in after booking (room assigned, parking, special requests, nationality confirmed at
  check-in) are deliberately not used; sending them is an error. The list is in `/api/v1/model`.
* Values the model never saw (a new agent ID, for example) are scored as "unknown" and named in
  `unseen_categories`: treat such an estimate with more caution.
* It must not be the only basis for refusing a guest. It scores bookings, not people.

### Errors
Every error is JSON with a `message` for people, an `errors` list of `{{field, problem}}` items and a `hint`.
Invalid input is a 422; nothing a client sends should produce a 500. Requests are limited to
{MAX_BATCH} bookings per batch and {MAX_BODY_BYTES // 1000} kB per body.
"""

TAGS = [
    {"name": "Prediction", "description": "Score one booking or a batch."},
    {"name": "Feedback", "description": "Report what really happened, so live performance can be measured and "
                                        "the model retrained."},
    {"name": "Model", "description": "What the model is, how it scored on unseen test bookings, and what it "
                                     "deliberately ignores."},
    {"name": "Operations", "description": "Health check and request statistics."},
]

app = FastAPI(title="Hotel cancellation risk", version=MODEL.version, description=DESCRIPTION, openapi_tags=TAGS)


def error_response(status: int, code: str, message: str, *, errors: list[dict] | None = None, hint: str = HINT,
                   detail=None, headers: dict | None = None, **extra) -> JSONResponse:
    """The one shape every error has: error, message, errors, hint, detail."""
    body = {"error": code, "message": message, "errors": errors or [], "hint": hint,
            "detail": message if detail is None else detail, **extra}
    return JSONResponse(body, status_code=status, headers=headers)


class RequestGuard:
    """Times and logs every request, refuses oversized bodies, and turns any crash into clean JSON."""

    def __init__(self, inner) -> None:
        self.inner = inner

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return
        start = time.perf_counter()
        seen = {"status": 500, "started": False, "bytes": 0}

        async def counted_receive():
            message = await receive()
            if message["type"] == "http.request":
                seen["bytes"] += len(message.get("body", b""))
                if seen["bytes"] > MAX_BODY_BYTES:
                    # Read the rest without keeping it, so the client gets an answer rather than a reset.
                    while message.get("more_body") and seen["bytes"] < DRAIN_LIMIT_BYTES:
                        message = await receive()
                        if message["type"] != "http.request":
                            break
                        seen["bytes"] += len(message.get("body", b""))
                    # FastAPI's own class: its body reader re-raises that one and wraps anything else in a 400.
                    raise HTTPException(status_code=413)
            return message

        async def stamped_send(message):
            if message["type"] == "http.response.start":
                seen["started"], seen["status"] = True, message["status"]
                MutableHeaders(scope=message)["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
            await send(message)

        try:
            await self.inner(scope, counted_receive, stamped_send)
        except Exception:  # noqa: BLE001 - the catch-all: log the traceback, answer with JSON
            error_id = uuid.uuid4().hex[:12]
            log.exception(json.dumps({"event": "unhandled_error", "error_id": error_id, "path": scope["path"],
                                      "method": scope["method"]}))
            if not seen["started"]:
                await unexpected_error(error_id)(scope, receive, stamped_send)
        finally:
            ms = (time.perf_counter() - start) * 1000
            status = seen["status"]
            STATS["requests"] += 1
            STATS["errors"] += int(status >= 500)
            STATS["rejected"] += int(400 <= status < 500)
            if scope["path"].startswith("/api/"):
                STATS["latencies_ms"].append(ms)
            log.info(json.dumps({"path": scope["path"], "method": scope["method"], "status": status,
                                 "ms": round(ms, 2), "model_version": MODEL.version}))


# The middleware added last is outermost. CORS goes outside the guard so that the guard's own error
# responses get CORS headers too.
app.add_middleware(RequestGuard)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
                   expose_headers=["X-Process-Time-ms"])


# ----------------------------------------------------------------------------------------------
# Errors: every one is JSON with the same keys
# ----------------------------------------------------------------------------------------------

def unexpected_error(error_id: str) -> JSONResponse:
    return error_response(
        500, "internal_error",
        "Something went wrong on the server while handling this request. It has been logged; the request was "
        "not processed.",
        hint=f"Try again. If it keeps happening, quote error_id {error_id}. /health shows whether the service "
             "is up.",
        error_id=error_id)


def _shown(value) -> str:
    """A short, safe rendering of a rejected value ('' when it is not worth echoing)."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if value != value else ("Infinity" if value > 0 else "-Infinity")
    if value is not None and not isinstance(value, (str, int, float, bool)):
        return ""
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 40 else text[:37] + "..."


def _plain(number) -> str:
    """1000000.0 -> '1000000', 0.5 -> '0.5' (no exponent notation in messages)."""
    return str(int(number)) if float(number).is_integer() else str(number)


def _field_name(loc: tuple) -> str:
    parts = list(loc[1:]) if loc and loc[0] == "body" else list(loc)
    name = ""
    for part in parts:
        name += f"[{part}]" if isinstance(part, int) else (f".{part}" if name else str(part))
    return name or "body"


def _excluded_columns() -> set[str]:
    return {column for group in MODEL.metadata.get("excluded_columns", {}).values() for column in group}


def describe_problem(err: dict) -> dict:
    """Turn one pydantic error into ``{field, problem}`` in plain words."""
    kind, ctx, loc, value = err["type"], err.get("ctx") or {}, tuple(err["loc"]), err.get("input")
    echo = True
    not_finite = isinstance(value, float) and not math.isfinite(value)
    if kind == "json_invalid":
        return {"field": "body", "problem": f"is not valid JSON ({ctx.get('error', 'syntax error')} at "
                                            f"character {loc[-1]})"}
    if kind == "no_guests":
        return {"field": _field_name((*loc, "adults")),
                "problem": "plus children plus babies must be at least 1: a booking needs a guest"}
    if kind == "missing":
        text, echo = ("is empty: send a JSON object" if len(loc) == 1 else "is required but missing"), False
    elif kind == "extra_forbidden":
        echo = False
        if loc[-1] in _excluded_columns():
            text = ("is deliberately not used: it is not reliably known when a booking is made (see "
                    "excluded_columns in /api/v1/model). Leave it out")
        else:
            text = "is not a field of this API. Leave it out"
    elif kind in ("model_type", "model_attributes_type", "dict_type"):
        text, echo = 'must be a JSON object, like {"hotel": "City Hotel", ...}', False
    elif kind == "list_type":
        text, echo = "must be a JSON list, like [ {...}, {...} ]", False
    elif kind == "literal_error":
        text = f"must be {ctx.get('expected', 'one of the listed values')}"
    elif not_finite or kind == "finite_number":
        text = "must be a finite number"
    elif kind == "greater_than_equal":
        text = f"must be at least {_plain(ctx['ge'])}"
    elif kind == "less_than_equal":
        text = f"must be at most {_plain(ctx['le'])}"
    elif kind in ("int_parsing", "int_type", "int_from_float", "int_parsing_size"):
        text = "must be a whole number"
    elif kind in ("float_parsing", "float_type"):
        text = "must be a number"
    elif kind in ("bool_parsing", "bool_type"):
        text = "must be true or false"
    elif kind in ("date_type", "date_parsing", "date_format") or kind.startswith("date_from_datetime"):
        text = "must be a real calendar date written as YYYY-MM-DD"
    elif kind in ("too_long", "too_short"):
        limit = f"at most {ctx.get('max_length')}" if kind == "too_long" else f"at least {ctx.get('min_length')}"
        text, echo = f"must contain {limit} item(s), but has {ctx.get('actual_length')}", False
    elif kind == "string_pattern_mismatch":
        text = "may only contain letters, digits and the characters . _ : -"
    else:
        text = err.get("msg", "is not valid").replace("Value error, ", "")
        for old, new in (("Input should be ", "must be "), ("String should ", "must "), ("Input should ", "must ")):
            if text.startswith(old):
                text = new + text[len(old):]
    shown = _shown(value) if echo else ""
    return {"field": _field_name(loc), "problem": f"{text} (got {shown})" if shown else text}


def _sentence(problem: dict) -> str:
    subject = "the request body" if problem["field"] == "body" else problem["field"]
    return f"{subject} {problem['problem']}"


def _is_json_content_type(value: str | None) -> bool:
    media = (value or "").split(";")[0].strip().lower()
    return media.startswith("application/") and (media.endswith("/json") or media.endswith("+json"))


@app.exception_handler(RequestValidationError)
async def on_invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    raw = exc.errors()
    detail = [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in raw[:MAX_PROBLEMS_LISTED]]
    whole_body = bool(raw) and tuple(raw[0]["loc"]) == ("body",)
    content_type = request.headers.get("content-type")
    if whole_body and raw[0]["type"] != "missing" and not _is_json_content_type(content_type):
        return error_response(
            415, "unsupported_media_type",
            "This endpoint reads JSON. Send the body with the header Content-Type: application/json "
            f"(this request had {_shown(content_type) if content_type else 'no Content-Type header'}).",
            errors=[{"field": "Content-Type", "problem": "must be application/json"}], detail=detail)
    problems = [describe_problem(e) for e in raw[:MAX_PROBLEMS_LISTED]]
    if len(raw) == 1:
        message = f"The request was not accepted: {_sentence(problems[0])}."
    else:
        message = f"The request was not accepted: {len(raw)} problems found. The first: {_sentence(problems[0])}."
    if len(raw) > MAX_PROBLEMS_LISTED:
        message += f" Only the first {MAX_PROBLEMS_LISTED} are listed."
    return error_response(422, "validation_error", message, errors=problems, detail=detail)


def endpoint_list() -> list[dict]:
    listed = [{"method": "GET", "path": "/", "summary": "Demo page"},
              {"method": "GET", "path": "/docs", "summary": "Interactive API documentation"}]
    for route in app.routes:
        if isinstance(route, APIRoute) and route.include_in_schema:
            listed += [{"method": method, "path": route.path, "summary": route.summary or route.name}
                       for method in sorted(route.methods)]
    return listed


@app.exception_handler(StarletteHTTPException)
async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    where = f"{request.method} {request.url.path}"
    headers = dict(exc.headers) if exc.headers else None
    if exc.status_code == 404:
        return error_response(404, "not_found", f"There is no {_shown(where)} on this API.",
                              hint="The endpoints that exist are listed under 'endpoints'; /docs describes them.",
                              detail="Not Found", endpoints=endpoint_list())
    if exc.status_code == 405:
        allowed = (exc.headers or {}).get("Allow") or (exc.headers or {}).get("allow") or "another method"
        return error_response(405, "method_not_allowed",
                              f"{request.method} is not allowed on {request.url.path}. Use: {allowed}.",
                              detail="Method Not Allowed", headers=headers)
    if exc.status_code == 413:
        return error_response(413, "payload_too_large",
                              f"The request body is too large: the limit is {MAX_BODY_BYTES // 1000} kB, and a "
                              f"batch may hold at most {MAX_BATCH} bookings. Split it into several requests.",
                              detail="Payload Too Large")
    if exc.status_code == 400:
        return error_response(400, "bad_request",
                              "The request body could not be read as JSON. It must be UTF-8 text and not "
                              "nested unusually deeply.", detail=str(exc.detail))
    return error_response(exc.status_code, "http_error", str(exc.detail), detail=str(exc.detail), headers=headers)


@app.exception_handler(Exception)
async def on_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Second line of defence; RequestGuard normally catches a crash first."""
    error_id = uuid.uuid4().hex[:12]
    log.error(json.dumps({"event": "unhandled_error", "error_id": error_id, "path": request.url.path,
                          "error": type(exc).__name__}))
    return unexpected_error(error_id)


INVALID = {"model": ErrorResponse, "description": "The input was not accepted. `errors` names each field and "
                                                  "what is wrong with it."}
TOO_LARGE = {"model": ErrorResponse, "description": f"The body is larger than {MAX_BODY_BYTES // 1000} kB."}
NOT_JSON = {"model": ErrorResponse, "description": "The body was not sent as `application/json`."}
BODY_ERRORS = {422: INVALID, 413: TOO_LARGE, 415: NOT_JSON}


def _example(value) -> dict:
    return {"content": {"application/json": {"example": value}}}


# ----------------------------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------------------------

@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def index():
    # On Vercel the files in public/ are served by the CDN at /index.html and may not be in the function bundle.
    return FileResponse(DEMO_PAGE) if DEMO_PAGE.is_file() else RedirectResponse("/index.html", status_code=307)


@app.api_route("/index.html", methods=["GET", "HEAD"], include_in_schema=False)
def index_file():
    if not DEMO_PAGE.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(DEMO_PAGE)


def health_report() -> dict:
    healthy = EXAMPLE_PREDICTION is not None
    return {"status": "ok" if healthy else "degraded", "model_version": MODEL.version,
            "model_self_check": "ok" if healthy else "failed", "feedback_storage": FEEDBACK.kind,
            "uptime_s": round(time.time() - STARTED, 1)}


@app.get("/health", tags=["Operations"], summary="Is the service up?",
         responses={200: _example({**health_report(), "uptime_s": 12.3})})
def health() -> dict:
    """Returns `status: ok` when the model is loaded and scored the documented example booking at start-up
    (`model_self_check`). Also says where feedback is kept: `durable` (Redis) or `this instance only`.
    """
    return health_report()


def model_card() -> dict:
    m = MODEL.metadata
    keys = ["model_version", "trained_utc", "task", "estimator", "params", "threshold", "risk_bands", "costs",
            "test_metrics", "excluded_columns", "data", "feature_sources"]
    return {**{k: m[k] for k in keys if k in m}, "features": m["input_order"],
            "calibrated": m["calibration"]["used"], "known_categories": MODEL.known_categories()}


@app.get("/api/v1/model", tags=["Model"], summary="Model card: metrics, threshold, what is excluded",
         responses={200: _example(model_card())})
def model() -> dict:
    """Everything needed to judge the model, read from the metadata file exported with it:

    * `test_metrics`: scores on test bookings the model never saw, at the decision `threshold`
    * `threshold`, `costs`, `risk_bands`: how a probability becomes a decision and a label
    * `excluded_columns`: dataset columns that are deliberately not used, grouped by reason
    * `features`, `known_categories`: the model's inputs and the category values it was trained on
    * `estimator`, `params`, `trained_utc`, `model_version`, `data`: how and on what it was trained
    """
    return model_card()


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"],
          summary="Score one booking",
          responses={200: _example({**(EXAMPLE_PREDICTION or {}), "latency_ms": 1.2}), **BODY_ERRORS})
def predict(booking: Booking) -> dict:
    """Send one booking as it looks on the day it is made; get back the probability that it is cancelled.

    * `cancellation_probability`: 0 to 1. About that share of similar bookings is cancelled.
    * `will_cancel`: true when the probability is at or above `threshold`.
    * `risk_band`: low, medium or high, for display.
    * `unseen_categories`: inputs whose value the model never saw (scored as unknown).
    * `prediction_id`: keep it, and report the real outcome to `/api/v1/feedback` later.

    The example request below is a typical online booking; the example response is what this model returns
    for it.
    """
    start = time.perf_counter()
    result = MODEL.predict([booking.to_record()])[0]
    STATS["predictions"] += 1
    return {**result, "latency_ms": round((time.perf_counter() - start) * 1000, 3)}


@app.post("/api/v1/predict/batch", response_model=BatchResponse, tags=["Prediction"],
          summary=f"Score up to {MAX_BATCH} bookings",
          responses={200: _example({"predictions": [EXAMPLE_PREDICTION] if EXAMPLE_PREDICTION else [],
                                    "model_version": MODEL.version, "latency_ms": 2.5}), **BODY_ERRORS})
def predict_batch(batch: BatchRequest) -> dict:
    """Same as `/api/v1/predict` for a list of bookings. Predictions come back in the order sent.

    The whole batch is validated first: if any booking is invalid nothing is scored, and the error names the
    booking by position (for example `bookings[3].lead_time`).
    """
    start = time.perf_counter()
    results = MODEL.predict([b.to_record() for b in batch.bookings])
    STATS["predictions"] += len(results)
    return {"predictions": results, "model_version": MODEL.version,
            "latency_ms": round((time.perf_counter() - start) * 1000, 3)}


FEEDBACK_CAVEAT = ("Demo service: anyone can post feedback and the prediction is reported by the client, so these "
                   "figures show how monitoring works rather than audited performance.")


@app.post("/api/v1/feedback", response_model=FeedbackAck, tags=["Feedback"],
          summary="Report what really happened to a booking",
          responses={200: _example({"received": True, "stored": FEEDBACK.kind, "usable_for_metrics": True,
                                    "note": "Stored. See /api/v1/feedback/summary."}), **BODY_ERRORS})
def feedback(item: Feedback) -> dict:
    """Once a booking's outcome is known, send it back with the `prediction_id`.

    The API keeps no record of the bookings it scores, so also send the `cancellation_probability` and
    `will_cancel` you received (a booking system stores them next to the reservation anyway). With them the
    record counts towards the live figures in `/api/v1/feedback/summary`; without them only the outcome is
    counted.

    Only the ID, the outcome, the prediction and a timestamp are stored. Booking details are never stored, and
    unknown fields are rejected. Reporting the same `prediction_id` again replaces the earlier report.
    """
    probability = item.cancellation_probability
    decision = item.will_cancel
    if decision is None and probability is not None:
        decision = probability >= MODEL.threshold
    record = {"id": item.prediction_id, "y": int(item.was_cancelled),
              "p": None if probability is None else round(probability, 6),
              "d": None if decision is None else int(decision),
              "v": item.model_version, "ts": int(time.time())}
    stored = FEEDBACK.add(record)
    log.info(json.dumps({"event": "feedback", "prediction_id": item.prediction_id,
                         "was_cancelled": item.was_cancelled, "stored": stored, "model_version": MODEL.version}))
    usable = decision is not None
    note = ("Stored. It counts towards /api/v1/feedback/summary." if usable else
            "Stored as an outcome only. Send cancellation_probability and will_cancel as well to have it count "
            "towards the accuracy figures in /api/v1/feedback/summary.")
    return {"received": True, "stored": stored, "usable_for_metrics": usable, "note": note}


def feedback_report() -> dict:
    records, storage = FEEDBACK.records()
    return {"storage": storage, "storage_note": STORAGE_NOTES[storage], "capacity": FEEDBACK.max_records,
            **summarise_feedback(records, MODEL.threshold), "caveat": FEEDBACK_CAVEAT}


@app.get("/api/v1/feedback/summary", response_model=FeedbackSummary, tags=["Feedback"],
         summary="How the model is doing on the outcomes reported so far",
         responses={200: _example({
             "storage": FEEDBACK.kind, "storage_note": STORAGE_NOTES[FEEDBACK.kind], "capacity": FEEDBACK.max_records,
             **summarise_feedback([{"id": "a", "y": 1, "p": 0.8, "d": 1}, {"id": "b", "y": 0, "p": 0.2, "d": 0},
                                   {"id": "c", "y": 1, "p": 0.4, "d": 0}, {"id": "d", "y": 0}], MODEL.threshold),
             "caveat": FEEDBACK_CAVEAT})})
def feedback_summary() -> dict:
    """Live monitoring from the feedback collected by `/api/v1/feedback`.

    * `calibration`: over records that include the predicted probability, the share that really cancelled next
      to the average predicted probability. A large gap suggests the model has drifted.
    * `decisions`: over records that include the decision, the confusion counts with accuracy, precision and
      recall.
    * `storage`: `durable` when a Redis database is configured, otherwise `this instance only`.

    Ratios are `null` until there is something to divide by.
    """
    return feedback_report()


@app.get("/api/v1/metrics", tags=["Operations"], summary="Request counts and latency of this instance",
         responses={200: _example({"scope": "this instance only (serverless instances are stateless)",
                                   "uptime_s": 12.3, "requests": 42, "errors": 0, "rejected": 3, "predictions": 30,
                                   "api_latency_ms": {"p50": 1.9, "p95": 4.2, "p99": 6.0}})})
def metrics() -> dict:
    """Counters since this instance started: `requests` (all), `errors` (5xx answers), `rejected` (4xx answers,
    such as invalid input), `predictions` (bookings scored) and latency percentiles of the `/api/` requests.
    """
    lat = np.asarray(STATS["latencies_ms"]) if STATS["latencies_ms"] else np.asarray([0.0])
    return {"scope": "this instance only (serverless instances are stateless)",
            "uptime_s": round(time.time() - STARTED, 1), "requests": STATS["requests"], "errors": STATS["errors"],
            "rejected": STATS["rejected"], "predictions": STATS["predictions"],
            "api_latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                               "p95": round(float(np.percentile(lat, 95)), 2),
                               "p99": round(float(np.percentile(lat, 99)), 2)}}
