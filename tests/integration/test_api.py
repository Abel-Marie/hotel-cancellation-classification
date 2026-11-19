"""The FastAPI service, exercised in-process."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(service_app):
    return TestClient(service_app.app)


@pytest.fixture(scope="module")
def booking(service_app):
    from core.schemas import EXAMPLE_BOOKING

    return dict(EXAMPLE_BOOKING)


def test_health_and_model_card(client):
    assert client.get("/health").json()["status"] == "ok"
    card = client.get("/api/v1/model").json()
    assert "country" in card["excluded_columns"]["timing_uncertain"]
    assert 0 < card["threshold"] < 1 and card["test_metrics"]["roc_auc"] > 0.5


def test_predict(client, booking):
    r = client.post("/api/v1/predict", json=booking)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["cancellation_probability"] <= 1
    assert body["will_cancel"] == (body["cancellation_probability"] >= body["threshold"])
    assert body["risk_band"] in {"low", "medium", "high"} and body["unseen_categories"] == []
    assert float(r.headers["X-Process-Time-ms"]) > 0


def test_non_refundable_deposit_raises_risk(client, booking):
    low = client.post("/api/v1/predict", json=booking).json()["cancellation_probability"]
    high = client.post("/api/v1/predict", json={**booking, "deposit_type": "Non Refund"}).json()
    assert high["cancellation_probability"] > low


def test_batch_and_unknown_agent(client, booking):
    r = client.post("/api/v1/predict/batch", json={"bookings": [booking, {**booking, "agent": 99999}]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2 and preds[1]["unseen_categories"] == ["agent_id"]


@pytest.mark.parametrize("change", [{"hotel": "Motel"}, {"lead_time": -1}, {"adults": 0}, {"country": "PRT"},
                                    {"arrival_date": "not-a-date"}])
def test_invalid_input_is_a_422(client, booking, change):
    body = {**booking, **change}
    if change == {"adults": 0}:
        body.update(children=0, babies=0)
    assert client.post("/api/v1/predict", json=body).status_code == 422


def test_feedback_and_metrics(client):
    assert client.post("/api/v1/feedback", json={"prediction_id": "abc", "was_cancelled": True}).json()["received"]
    m = client.get("/api/v1/metrics").json()
    assert m["predictions"] >= 1 and m["api_latency_ms"]["p50"] >= 0


def test_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


# ----------------------------------------------------------------------------------------------
# Odd input: always a clean JSON answer, never a 500
# ----------------------------------------------------------------------------------------------

JSON = {"Content-Type": "application/json"}


def raw_booking(booking, **replace) -> bytes:
    """The example booking as JSON text, with some values replaced by raw tokens such as NaN."""
    import json

    text = json.dumps(booking)
    for key, token in replace.items():
        text = text.replace(f'"{key}": {json.dumps(booking[key])}', f'"{key}": {token}')
    return text.encode()


def assert_friendly_error(response, status: int) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["message"] and body["hint"] and isinstance(body["errors"], list)
    assert "/docs" in body["hint"] or status >= 500  # client errors point to the documentation
    assert "Traceback" not in response.text and "detail" in body
    assert float(response.headers["X-Process-Time-ms"]) >= 0
    return body


def test_validation_error_names_the_field_in_plain_words(client, booking):
    body = assert_friendly_error(client.post("/api/v1/predict", json={**booking, "lead_time": -1}), 422)
    assert body["error"] == "validation_error"
    assert body["errors"] == [{"field": "lead_time", "problem": "must be at least 0 (got -1)"}]
    assert "lead_time must be at least 0" in body["message"]
    assert body["detail"][0]["loc"] == ["body", "lead_time"]  # FastAPI's usual entry is still there


def test_excluded_column_is_explained_not_just_refused(client, booking):
    body = assert_friendly_error(client.post("/api/v1/predict", json={**booking, "country": "PRT"}), 422)
    assert body["errors"][0]["field"] == "country" and "deliberately not used" in body["errors"][0]["problem"]


@pytest.mark.parametrize("change", [
    {"lead_time": "abc"}, {"lead_time": True}, {"lead_time": 1.5}, {"lead_time": 10 ** 30}, {"lead_time": 1001},
    {"adr": -5}, {"adr": 5400}, {"adr": "free"}, {"adults": 99}, {"agent": 0}, {"agent": 10 ** 12},
    {"arrival_date": "2026-02-30"}, {"arrival_date": "1900-01-01"}, {"arrival_date": "2200-01-01"},
    {"arrival_date": 1699920000}, {"arrival_date": "14/08/2026"}, {"hotel": None}, {"hotel": ["City Hotel"]},
    {"meal": "XX"}, {"is_repeated_guest": "maybe"}, {"hotel": "<script>alert(1)</script>"},
])
def test_impossible_values_are_422_with_a_named_field(client, booking, change):
    body = assert_friendly_error(client.post("/api/v1/predict", json={**booking, **change}), 422)
    assert body["errors"][0]["field"] == next(iter(change))


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
@pytest.mark.parametrize("field", ["adr", "lead_time"])
def test_nan_and_infinity_are_rejected_cleanly(client, booking, field, token):
    r = client.post("/api/v1/predict", content=raw_booking(booking, **{field: token}), headers=JSON)
    assert assert_friendly_error(r, 422)["errors"][0]["field"] == field


@pytest.mark.parametrize(("content", "headers", "status", "words"), [
    (b'{"hotel": "City Hotel",', JSON, 422, "not valid JSON"),
    (b"", JSON, 422, "empty"),
    (b"", {}, 422, "empty"),
    (b"[1, 2, 3]", JSON, 422, "JSON object"),
    (b'"hello"', JSON, 422, "JSON object"),
    (b"null", JSON, 422, "JSON object"),
    (b"{}", JSON, 422, "required"),
    (b'{"hotel": "City Hotel"}', {"Content-Type": "text/plain"}, 415, "application/json"),
    (b"hotel=City+Hotel", {"Content-Type": "application/x-www-form-urlencoded"}, 415, "application/json"),
    (b'{"hotel": "\xff\xfe\xfd"}', JSON, 400, "could not be read"),  # not UTF-8
    (b"[" * 100_000 + b"]" * 100_000, JSON, 400, "could not be read"),
])
def test_malformed_bodies(client, content, headers, status, words):
    for path in ("/api/v1/predict", "/api/v1/predict/batch", "/api/v1/feedback"):
        body = assert_friendly_error(client.post(path, content=content, headers=headers), status)
        assert words in body["message"]


def test_zero_nights_and_odd_lead_times_are_scored(client, booking):
    """The data contains day-use bookings, and lead time cannot be checked without a booking date."""
    for change in ({"stays_in_weekend_nights": 0, "stays_in_week_nights": 0}, {"lead_time": 0}, {"lead_time": 1000},
                   {"company": 99999}, {"lead_time": "120"}):
        assert client.post("/api/v1/predict", json={**booking, **change}).status_code == 200


def test_zero_guests_message(client, booking):
    r = client.post("/api/v1/predict", json={**booking, "adults": 0, "children": 0, "babies": 0})
    assert "needs a guest" in assert_friendly_error(r, 422)["message"]


@pytest.mark.parametrize(("size", "status"), [(0, 422), (1, 200), (100, 200), (101, 422)])
def test_batch_size_limits(client, booking, size, status):
    r = client.post("/api/v1/predict/batch", json={"bookings": [booking] * size})
    assert r.status_code == status
    if status == 200:
        assert len(r.json()["predictions"]) == size
    else:
        assert assert_friendly_error(r, 422)["errors"][0]["field"] == "bookings"


def test_batch_error_names_the_booking(client, booking):
    r = client.post("/api/v1/predict/batch", json={"bookings": [booking, booking, {**booking, "adr": -1}]})
    assert assert_friendly_error(r, 422)["errors"][0]["field"] == "bookings[2].adr"


def test_oversized_body_is_a_413(client, booking, service_app):
    r = client.post("/api/v1/predict/batch", json={"bookings": [booking] * 10_000})
    assert len(r.request.content) > service_app.MAX_BODY_BYTES
    assert assert_friendly_error(r, 413)["error"] == "payload_too_large"


def test_unknown_path_lists_the_endpoints(client):
    body = assert_friendly_error(client.get("/api/v2/predict"), 404)
    paths = {e["path"] for e in body["endpoints"]}
    assert {"/api/v1/predict", "/api/v1/predict/batch", "/api/v1/feedback", "/api/v1/feedback/summary",
            "/api/v1/model", "/api/v1/metrics", "/health", "/docs"} <= paths


@pytest.mark.parametrize(("method", "path", "allowed"), [("GET", "/api/v1/predict", "POST"),
                                                         ("POST", "/health", "GET"),
                                                         ("DELETE", "/api/v1/feedback", "POST")])
def test_wrong_method_says_which_is_allowed(client, method, path, allowed):
    r = client.request(method, path)
    assert allowed in assert_friendly_error(r, 405)["message"] and allowed in r.headers["allow"]


def test_a_crash_becomes_clean_json(service_app, booking, monkeypatch):
    def broken(_records):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(service_app.MODEL, "predict", broken)
    crash_client = TestClient(service_app.app, raise_server_exceptions=False)
    body = assert_friendly_error(crash_client.post("/api/v1/predict", json=booking), 500)
    assert body["error"] == "internal_error" and body["error_id"]
    assert "secret internal detail" not in str(body)


# ----------------------------------------------------------------------------------------------
# Categories and limits agree with the data and with the model
# ----------------------------------------------------------------------------------------------

def test_undefined_categories_from_the_raw_data_are_accepted(client, booking, cfg, service_app):
    from core.schemas import Booking

    r = client.post("/api/v1/predict", json={**booking, "market_segment": "Undefined",
                                             "distribution_channel": "Undefined", "meal": "Undefined"})
    assert r.status_code == 200
    # "Undefined" meal is scored exactly as the training pipeline's cleaning rule treats it.
    record = Booking(**{**booking, "meal": "Undefined"}).to_record()
    assert record["meal"] == cfg.cleaning.fill_meal_undefined_as
    known = service_app.MODEL.codes
    expected_unseen = [name for name in ("market_segment", "distribution_channel") if "Undefined" not in known[name]]
    assert r.json()["unseen_categories"] == expected_unseen


def test_schema_accepts_every_category_the_model_knows(service_app, booking):
    """If a regenerated model knows a value the request schema refuses, this fails."""
    from core.schemas import Booking

    for name, codes in service_app.MODEL.codes.items():
        for value in codes:
            change = {name: value} if name != "agent_id" else {"agent": None if value == "none" else int(value)}
            assert Booking(**{**booking, **change})


def test_real_bookings_are_never_rejected(root, cfg, service_app):
    """Cleaned dataset rows, including the extremes and the rare categories, all pass the request schema."""
    import pandas as pd
    from core.schemas import Booking

    from hotel_cancellation.cleaning import clean_bookings
    from hotel_cancellation.monitoring.latency import to_api_payload

    raw = pd.read_csv(root / "data" / "mirror" / "hotels.csv.gz")
    clean, _ = clean_bookings(raw, cfg, drop_duplicates=False)
    numeric = ["lead_time", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
               "previous_cancellations", "previous_bookings_not_canceled", "adr", "agent", "company"]
    extremes = {int(clean[c].idxmax()) for c in numeric} | {int(clean[c].idxmin()) for c in numeric}
    rare = clean.index[(clean[["meal", "market_segment", "distribution_channel"]] == "Undefined").any(axis=1)
                       | clean["reserved_room_type"].isin(["L", "P"])]
    rows = clean.loc[sorted(extremes | set(rare) | set(clean.index[::10]))]
    assert len(rows) > 10_000
    for row in rows.to_dict("records"):
        Booking(**to_api_payload(row))


# ----------------------------------------------------------------------------------------------
# Feedback: stored, summarised, never the booking itself
# ----------------------------------------------------------------------------------------------

class FakeRedis:
    """Stands in for the Upstash REST pipeline endpoint behind ``urllib.request.urlopen``."""

    def __init__(self):
        self.lists, self.calls, self.down = {}, [], False

    def urlopen(self, request, timeout=None):
        import io
        import json
        import urllib.error

        self.calls.append({"url": request.full_url, "auth": request.get_header("Authorization"), "timeout": timeout})
        if self.down:
            raise urllib.error.URLError("connection refused")
        results = []
        for name, key, *args in json.loads(request.data):
            items = self.lists.setdefault(key, [])
            if name == "LPUSH":
                items.insert(0, args[0])
                results.append({"result": len(items)})
            elif name == "LTRIM":
                del items[int(args[1]) + 1:]
                results.append({"result": "OK"})
            elif name == "LRANGE":
                results.append({"result": items[int(args[0]):int(args[1]) + 1]})
            else:
                results.append({"error": f"ERR unknown command {name}"})
        return io.BytesIO(json.dumps(results).encode())


@pytest.fixture
def fake_redis(monkeypatch):
    import urllib.request

    fake = FakeRedis()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    return fake


def record(i, cancelled=True, p=0.8, d=1):
    return {"id": f"id{i}", "y": int(cancelled), "p": p, "d": d, "v": "m1", "ts": i}


def test_feedback_store_in_memory_is_a_ring_buffer(service_app, fake_redis):
    store = service_app.FeedbackStore(max_records=3)
    assert store.kind == "this instance only"
    assert [store.add(record(i)) for i in range(5)] == ["this instance only"] * 5
    records, where = store.records()
    assert where == "this instance only" and [r["id"] for r in records] == ["id4", "id3", "id2"]
    assert fake_redis.calls == []  # no network without configuration


def test_feedback_store_uses_redis_when_configured(service_app, fake_redis):
    store = service_app.FeedbackStore.from_env({"UPSTASH_REDIS_REST_URL": "https://example.upstash.io/",
                                               "UPSTASH_REDIS_REST_TOKEN": "s3cret"})
    store.max_records = 3
    assert store.kind == "durable"
    assert [store.add(record(i)) for i in range(5)] == ["durable"] * 5
    assert len(fake_redis.lists[store.KEY]) == 3  # capped
    call = fake_redis.calls[0]
    assert call["url"] == "https://example.upstash.io/pipeline" and call["auth"] == "Bearer s3cret"
    assert 0 < call["timeout"] <= 5
    # A second instance (another serverless invocation) reads what the first one wrote.
    other = service_app.FeedbackStore(url="https://example.upstash.io", token="s3cret")
    records, where = other.records()
    assert where == "durable" and [r["id"] for r in records] == ["id4", "id3", "id2"]


def test_feedback_store_reads_vercel_kv_names_and_insists_on_https(service_app):
    from_kv = service_app.FeedbackStore.from_env({"KV_REST_API_URL": "https://kv.example", "KV_REST_API_TOKEN": "t"})
    assert from_kv.kind == "durable" and from_kv.url == "https://kv.example"
    for env in ({}, {"UPSTASH_REDIS_REST_URL": "https://x.example"}, {"UPSTASH_REDIS_REST_TOKEN": "t"},
                {"UPSTASH_REDIS_REST_URL": "http://x.example", "UPSTASH_REDIS_REST_TOKEN": "t"},
                {"UPSTASH_REDIS_REST_URL": "file:///etc/passwd", "UPSTASH_REDIS_REST_TOKEN": "t"}):
        assert service_app.FeedbackStore.from_env(env).kind == "this instance only"


def test_feedback_store_falls_back_to_memory_when_redis_fails(service_app, fake_redis):
    store = service_app.FeedbackStore(url="https://example.upstash.io", token="t", retry_after_s=60)
    assert store.add(record(1)) == "durable"
    fake_redis.down = True
    assert store.add(record(2)) == "this instance only"  # no exception reaches the caller
    calls = len(fake_redis.calls)
    assert store.add(record(3)) == "this instance only"
    assert len(fake_redis.calls) == calls  # paused: a failing database is not retried on every request
    records, where = store.records()
    assert where == "this instance only" and [r["id"] for r in records] == ["id3", "id2"]
    fake_redis.down, store._skip_redis_until = False, 0.0
    records, where = store.records()
    assert where == "durable" and [r["id"] for r in records] == ["id3", "id2", "id1"]


def test_feedback_store_treats_a_redis_error_reply_as_failure(service_app, fake_redis):
    store = service_app.FeedbackStore(url="https://example.upstash.io", token="t")
    assert store._try_redis(["FLUSHALL", "x"]) is None


def test_feedback_summary_arithmetic(service_app):
    records = [
        record(1, cancelled=True, p=0.9, d=1),    # true positive
        record(2, cancelled=False, p=0.7, d=1),   # false positive
        record(3, cancelled=False, p=0.1, d=0),   # true negative
        record(4, cancelled=True, p=0.3, d=0),    # false negative
        record(5, cancelled=True, p=0.5, d=0),    # false negative
        {"id": "id6", "y": 1, "p": None, "d": None, "v": None, "ts": 6},  # outcome only
        record(1, cancelled=False, p=0.9, d=1),   # older report for id1: ignored
    ]
    s = service_app.summarise_feedback(records, threshold=0.6)
    assert s["records"] == 6 and s["repeats_ignored"] == 1
    assert s["outcomes"] == {"cancelled": 4, "arrived": 2}
    assert s["calibration"] == {"records": 5, "observed_cancellation_rate": 0.6, "mean_predicted_probability": 0.5}
    assert s["decisions"]["confusion"] == {"true_positive": 1, "false_positive": 1, "true_negative": 1,
                                           "false_negative": 2}
    assert s["decisions"]["accuracy"] == 0.4 and s["decisions"]["precision"] == 0.5
    assert s["decisions"]["recall"] == round(1 / 3, 4)
    assert s["model_versions"] == {"m1": 5}
    empty = service_app.summarise_feedback([], threshold=0.6)
    assert empty["records"] == 0 and empty["observed_cancellation_rate"] is None
    assert empty["decisions"]["precision"] is None and empty["calibration"]["mean_predicted_probability"] is None


def test_feedback_round_trip_through_the_api(client, booking, service_app, monkeypatch):
    monkeypatch.setattr(service_app, "FEEDBACK", service_app.FeedbackStore())
    low = client.post("/api/v1/predict", json=booking).json()
    high = client.post("/api/v1/predict", json={**booking, "deposit_type": "Non Refund"}).json()
    for prediction, outcome in ((low, False), (high, True)):
        ack = client.post("/api/v1/feedback", json={
            "prediction_id": prediction["prediction_id"], "was_cancelled": outcome,
            "cancellation_probability": prediction["cancellation_probability"],
            "will_cancel": prediction["will_cancel"], "model_version": prediction["model_version"]}).json()
        assert ack == {**ack, "received": True, "stored": "this instance only", "usable_for_metrics": True}
    # Probability only: the decision is derived with the current threshold. Outcome only: counted, not scored.
    assert client.post("/api/v1/feedback", json={"prediction_id": "p-only", "was_cancelled": True,
                                                 "cancellation_probability": 0.99}).json()["usable_for_metrics"]
    assert not client.post("/api/v1/feedback", json={"prediction_id": "bare", "was_cancelled": True}
                           ).json()["usable_for_metrics"]
    summary = client.get("/api/v1/feedback/summary").json()
    assert summary["storage"] == "this instance only" and summary["records"] == 4
    assert summary["calibration"]["records"] == 3 and summary["decisions"]["records"] == 3
    confusion = summary["decisions"]["confusion"]
    assert sum(confusion.values()) == 3 and confusion["true_positive"] == int(high["will_cancel"]) + 1
    assert summary["current_threshold"] == low["threshold"]
    assert summary["model_versions"] == {low["model_version"]: 2}
    # What is kept: an ID, the outcome, the prediction, a timestamp. Nothing about the booking.
    stored, _ = service_app.FEEDBACK.records()
    assert all(set(r) == {"id", "y", "p", "d", "v", "ts"} for r in stored)


@pytest.mark.parametrize("body", [
    {"prediction_id": "abc"}, {"was_cancelled": True}, {"prediction_id": "", "was_cancelled": True},
    {"prediction_id": "x" * 65, "was_cancelled": True}, {"prediction_id": "has space", "was_cancelled": True},
    {"prediction_id": "abc", "was_cancelled": "maybe"},
    {"prediction_id": "abc", "was_cancelled": True, "cancellation_probability": 1.5},
    {"prediction_id": "abc", "was_cancelled": True, "guest_name": "Ana"},
    {"prediction_id": "abc", "was_cancelled": True, "adr": 100.0},
])
def test_feedback_rejects_bad_or_extra_fields(client, body):
    assert_friendly_error(client.post("/api/v1/feedback", json=body), 422)


# ----------------------------------------------------------------------------------------------
# Documentation and demo page
# ----------------------------------------------------------------------------------------------

def test_every_endpoint_is_documented(client):
    spec = client.get("/openapi.json").json()
    assert "demo page" in spec["info"]["description"] and "Limits" in spec["info"]["description"]
    assert len(spec["paths"]) == 7
    for path, operations in spec["paths"].items():
        for method, op in operations.items():
            assert op["summary"] and len(op["description"]) > 40 and op["tags"], path
            assert "example" in op["responses"]["200"]["content"]["application/json"], path
            if method == "post":
                schema = op["requestBody"]["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
                assert "example" in spec["components"]["schemas"][schema], path
                assert op["responses"]["422"]["content"]["application/json"]["schema"]["$ref"].endswith(
                    "ErrorResponse")
    assert client.get("/docs").status_code == 200


def test_documented_prediction_example_is_what_the_model_returns(client, booking):
    spec = client.get("/openapi.json").json()
    example = spec["paths"]["/api/v1/predict"]["post"]["responses"]["200"]["content"]["application/json"]["example"]
    live = client.post("/api/v1/predict", json=booking).json()
    for key in ("cancellation_probability", "will_cancel", "risk_band", "threshold", "model_version"):
        assert example[key] == live[key]


def test_health_reports_self_check_and_feedback_storage(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_self_check"] == "ok"
    assert body["feedback_storage"] in {"durable", "this instance only"}


def test_model_card_keeps_its_keys(client, service_app):
    card = client.get("/api/v1/model").json()
    assert {"estimator", "params", "threshold", "test_metrics", "excluded_columns", "model_version",
            "known_categories", "features"} <= set(card)
    assert card["known_categories"]["hotel"] == sorted(service_app.MODEL.codes["hotel"])


def test_demo_page_is_self_contained_and_wired_to_the_api(client, root):
    import re

    html = client.get("/").text
    assert client.get("/index.html").text == html
    assert not re.search(r"<script[^>]+src=", html)  # no external JavaScript
    for needle in ("/api/v1/predict", "/api/v1/feedback", "/api/v1/feedback/summary", "/api/v1/model",
                   "/openapi.json", 'data-api-link="/docs"', 'data-api-link="/health"', "It was cancelled",
                   "The guest arrived", "Waking the server up", "course project", "X-Process-Time-ms"):
        assert needle.lower() in html.lower(), needle
    assert len(re.findall(r'data-preset="', html)) == 3
    # Nothing that lives in metadata.json is typed into the page.
    metadata = (root / "service" / "models" / "metadata.json").read_text(encoding="utf-8")
    version = re.search(r'"model_version": "([^"]+)"', metadata).group(1)
    assert version not in html


def test_demo_presets_give_clearly_different_risks(client, root):
    """The three example buttons, rebuilt from the page's own script, must be rated far apart."""
    import datetime as dt
    import json
    import re

    html = (root / "service" / "public" / "index.html").read_text(encoding="utf-8")

    def js_object(name: str) -> dict:
        body = re.search(rf"const {name} = (\{{.*?\}});\n", html, re.S).group(1)
        body = re.sub(r",(\s*})", r"\1", body)  # JavaScript allows a trailing comma, JSON does not
        return json.loads(re.sub(r"([{,]\s*)([A-Za-z_]+):", r'\1"\2":', body))

    base, presets = js_object("BASE"), js_object("PRESETS")
    assert set(presets) == {"online", "group", "returning"}
    today = dt.date.today()
    seen = {}
    for name, preset in presets.items():
        form = {**base, **preset["values"]}
        arrival = today + dt.timedelta(days=preset["lead"])
        weekend = sum((arrival + dt.timedelta(days=i)).weekday() >= 5 for i in range(preset["nights"]))
        payload = {**form, "lead_time": preset["lead"], "arrival_date": arrival.isoformat(),
                   "stays_in_weekend_nights": weekend, "stays_in_week_nights": preset["nights"] - weekend,
                   "agent": form["agent"] or None, "company": form["company"] or None}
        r = client.post("/api/v1/predict", json=payload)
        assert r.status_code == 200, r.text
        seen[name] = r.json()
    p = {name: body["cancellation_probability"] for name, body in seen.items()}
    assert p["returning"] + 0.2 < p["online"] < p["group"] - 0.2, p
    assert seen["group"]["will_cancel"] and not seen["returning"]["will_cancel"]
