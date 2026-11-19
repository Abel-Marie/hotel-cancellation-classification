"""Request and response models.

Two rules shape the input model:

* Category fields list every value that occurs in the raw dataset, including the
  ``Undefined`` placeholders, so a real booking row is never rejected. Which of
  those values the *model* knows is decided by ``metadata.json``; anything it has
  not seen is scored as "unknown" and named in ``unseen_categories``.
* Numeric limits are sanity limits, set above the largest value in the data
  (noted next to each field), not business rules. The price limit is the same
  one the training pipeline's cleaning step applies.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]

MAX_BATCH = 100
# The arrival year is not a model input (only week, weekday and month are), so these two
# dates only stop typos such as 1900 or 2200.
EARLIEST_ARRIVAL, LATEST_ARRIVAL = date(2000, 1, 1), date(2100, 12, 31)
# The dataset paper gives "Undefined" and "SC" the same meaning (no meal package); the
# training pipeline merges them during cleaning, so the service does the same.
MEAL_UNDEFINED_MEANS = "SC"

EXAMPLE_BOOKING = {
    "hotel": "City Hotel", "lead_time": 120, "arrival_date": "2026-08-14",
    "stays_in_weekend_nights": 1, "stays_in_week_nights": 3, "adults": 2, "children": 0, "babies": 0,
    "meal": "BB", "market_segment": "Online TA", "distribution_channel": "TA/TO",
    "is_repeated_guest": False, "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A", "deposit_type": "No Deposit", "agent": 9, "company": None,
    "customer_type": "Transient", "adr": 115.0,
}
EXAMPLE_FEEDBACK = {
    "prediction_id": "3f1c9a52b7e04d6f8a1b2c3d4e5f6071", "was_cancelled": True,
    "cancellation_probability": 0.42, "will_cancel": False,
}


def _not_a_boolean(value):
    """JSON ``true`` would otherwise be read as the number 1."""
    if isinstance(value, bool):
        raise PydanticCustomError("number_type", "Input should be a number, not true or false")
    return value


def _iso_date_only(value):
    """Accept ``YYYY-MM-DD`` (or a date object); refuse timestamps, which would parse as dates."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        return value.strip()
    raise PydanticCustomError("date_format", "Input should be a calendar date written as YYYY-MM-DD")


def _plausible_arrival(value: date) -> date:
    if not EARLIEST_ARRIVAL <= value <= LATEST_ARRIVAL:
        raise PydanticCustomError("date_out_of_range", "Input should be a date between {earliest} and {latest}",
                                  {"earliest": EARLIEST_ARRIVAL.isoformat(), "latest": LATEST_ARRIVAL.isoformat()})
    return value


Count = Annotated[int, BeforeValidator(_not_a_boolean)]
Amount = Annotated[float, BeforeValidator(_not_a_boolean)]
ArrivalDate = Annotated[date, BeforeValidator(_iso_date_only), AfterValidator(_plausible_arrival)]


class Booking(BaseModel):
    """A booking as it is known on the day it is made."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_BOOKING})

    hotel: Literal["City Hotel", "Resort Hotel"] = Field(description="Which of the two hotels")
    lead_time: Count = Field(ge=0, le=1000, description="Days between booking and arrival (largest in the "
                                                        "data: 737). Not checked against arrival_date, because "
                                                        "the booking date is not part of the request")
    arrival_date: ArrivalDate = Field(description="Arrival day as YYYY-MM-DD. Only its week of the year, weekday "
                                                  "and month are used; the year is deliberately ignored")
    stays_in_weekend_nights: Count = Field(ge=0, le=30, description="Saturday and Sunday nights (largest in the "
                                                                    "data: 19)")
    stays_in_week_nights: Count = Field(ge=0, le=60, description="Monday to Friday nights (largest in the data: "
                                                                 "50). Zero nights in total is allowed: the data "
                                                                 "contains day-use bookings")
    adults: Count = Field(ge=0, le=60, description="Adults (largest in the data: 55). At least one guest in total "
                                                   "is required")
    children: Count = Field(default=0, ge=0, le=10, description="Children (largest in the data: 10)")
    babies: Count = Field(default=0, ge=0, le=10, description="Babies (largest in the data: 10)")
    meal: Literal["BB", "HB", "FB", "SC", "Undefined"] = Field(
        description="BB bed and breakfast, HB half board, FB full board, SC no meal package. 'Undefined' is the "
                    "dataset's other code for no meal package and is scored as SC")
    market_segment: Literal["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary",
                            "Aviation", "Undefined"] = Field(
        description="How the booking reached the hotel (TA: travel agent, TO: tour operator). 'Undefined' occurs "
                    "in the data and is accepted")
    distribution_channel: Literal["TA/TO", "Direct", "Corporate", "GDS", "Undefined"] = Field(
        description="Sales channel. 'Undefined' occurs in the data and is accepted")
    is_repeated_guest: bool = Field(default=False, description="True if the guest has stayed before")
    previous_cancellations: Count = Field(default=0, ge=0, le=100,
                                          description="Earlier bookings this guest cancelled (largest in the data: "
                                                      "26)")
    previous_bookings_not_canceled: Count = Field(default=0, ge=0, le=200,
                                                  description="Earlier bookings this guest kept (largest in the "
                                                              "data: 72)")
    reserved_room_type: Literal["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"] = Field(
        description="Room type code reserved (anonymised letters, as in the dataset)")
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"] = Field(description="Deposit taken at booking")
    agent: Count | None = Field(default=None, ge=1, le=1_000_000,
                                description="Travel agency ID, or null for none. An ID the model has not seen is "
                                            "scored as unknown and listed in unseen_categories")
    company: Count | None = Field(default=None, ge=1, le=1_000_000,
                                  description="Company ID, or null for none")
    customer_type: Literal["Transient", "Transient-Party", "Contract", "Group"] = Field(
        description="Type of booking: Transient is an individual booking not tied to a group or contract")
    adr: Amount = Field(ge=0, le=1000, description="Average daily rate: price per night. The training data was "
                                                   "cleaned with the same 0 to 1000 limit")

    @model_validator(mode="after")
    def at_least_one_guest(self) -> Booking:
        if self.adults + self.children + self.babies < 1:
            raise PydanticCustomError("no_guests", "a booking needs at least one guest (adults, children and "
                                                   "babies are all zero)")
        return self

    def to_record(self) -> dict:
        """The raw-column form the shared feature code expects (same keys as the dataset)."""
        d = self.arrival_date
        return {
            "hotel": self.hotel, "lead_time": self.lead_time, "arrival_date_year": d.year,
            "arrival_date_month": MONTH_NAMES[d.month - 1], "arrival_date_day_of_month": d.day,
            "stays_in_weekend_nights": self.stays_in_weekend_nights,
            "stays_in_week_nights": self.stays_in_week_nights, "adults": self.adults, "children": self.children,
            "babies": self.babies, "meal": MEAL_UNDEFINED_MEANS if self.meal == "Undefined" else self.meal,
            "market_segment": self.market_segment,
            "distribution_channel": self.distribution_channel, "is_repeated_guest": int(self.is_repeated_guest),
            "previous_cancellations": self.previous_cancellations,
            "previous_bookings_not_canceled": self.previous_bookings_not_canceled,
            "reserved_room_type": self.reserved_room_type, "deposit_type": self.deposit_type,
            "agent": self.agent, "company": self.company, "customer_type": self.customer_type, "adr": self.adr,
        }


class Prediction(BaseModel):
    prediction_id: str = Field(description="Random ID for this prediction. Keep it with the reservation and send "
                                           "it back to /api/v1/feedback once the outcome is known")
    cancellation_probability: float = Field(description="Estimated probability, 0 to 1, that the booking is "
                                                        "cancelled before arrival")
    will_cancel: bool = Field(description="True when the probability is at or above the decision threshold")
    risk_band: Literal["low", "medium", "high"] = Field(description="Coarse label for display; the band limits "
                                                                    "are in /api/v1/model")
    threshold: float = Field(description="Decision threshold in use, chosen on validation data")
    unseen_categories: list[str] = Field(description="Inputs whose value the model never saw in training; each "
                                                     "was scored as 'unknown'")
    model_version: str = Field(description="Version of the model that made this prediction")


class PredictionResponse(Prediction):
    latency_ms: float = Field(description="Time spent scoring, in milliseconds (the X-Process-Time-ms response "
                                          "header covers the whole request)")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "example": {"bookings": [EXAMPLE_BOOKING, {**EXAMPLE_BOOKING, "deposit_type": "Non Refund"}]}})

    bookings: list[Booking] = Field(min_length=1, max_length=MAX_BATCH,
                                    description=f"Between 1 and {MAX_BATCH} bookings")


class BatchResponse(BaseModel):
    predictions: list[Prediction] = Field(description="One prediction per booking, in the order they were sent")
    model_version: str
    latency_ms: float = Field(description="Time spent scoring the whole batch, in milliseconds")


class Feedback(BaseModel):
    """The real outcome of a booking that was scored earlier."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_FEEDBACK})

    prediction_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$",
                               description="The prediction_id returned by /api/v1/predict")
    was_cancelled: bool = Field(description="True if the booking was cancelled, false if the guest arrived")
    cancellation_probability: Amount | None = Field(
        default=None, ge=0, le=1, description="The probability the API returned for this booking. Optional, but "
                                              "without it the record cannot be used to check calibration")
    will_cancel: bool | None = Field(
        default=None, description="The decision the API returned. Optional: if only the probability is sent, the "
                                  "decision is derived from it with the current threshold")
    model_version: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$",
                                      description="The model_version the API returned. Optional")


class FeedbackAck(BaseModel):
    received: bool
    stored: Literal["durable", "this instance only"] = Field(description="Where the record was kept")
    usable_for_metrics: bool = Field(description="True when the record carries the prediction, so it counts in "
                                                 "the live accuracy figures")
    note: str


class Confusion(BaseModel):
    true_positive: int = Field(description="Predicted to cancel, and cancelled")
    false_positive: int = Field(description="Predicted to cancel, but the guest arrived")
    true_negative: int = Field(description="Predicted to arrive, and arrived")
    false_negative: int = Field(description="Predicted to arrive, but cancelled")


class CalibrationCheck(BaseModel):
    records: int = Field(description="Feedback records that include the predicted probability")
    observed_cancellation_rate: float | None = Field(description="Share of those bookings that were cancelled")
    mean_predicted_probability: float | None = Field(description="Average probability the model gave them; close "
                                                                 "to the observed rate if the model is calibrated")


class DecisionCheck(BaseModel):
    records: int = Field(description="Feedback records that include the decision (sent, or derived from the "
                                     "probability)")
    confusion: Confusion
    accuracy: float | None
    precision: float | None = Field(description="Of the bookings flagged as likely to cancel, the share that did")
    recall: float | None = Field(description="Of the bookings that cancelled, the share that was flagged")


class FeedbackSummary(BaseModel):
    storage: Literal["durable", "this instance only"]
    storage_note: str
    records: int = Field(description="Feedback records kept, one per prediction_id (the latest report wins)")
    repeats_ignored: int = Field(description="Earlier reports for a prediction_id that was reported again")
    capacity: int = Field(description="Only the most recent reports are kept, up to this many")
    outcomes: dict[str, int] = Field(description="How many bookings were reported as cancelled and as arrived")
    observed_cancellation_rate: float | None
    calibration: CalibrationCheck
    decisions: DecisionCheck
    current_threshold: float
    model_versions: dict[str, int] = Field(description="Records per model version, where the client sent it")
    caveat: str


class Problem(BaseModel):
    field: str = Field(description="The field at fault, e.g. lead_time or bookings[3].adr")
    problem: str = Field(description="What is wrong with it, in plain words")


class ErrorResponse(BaseModel):
    """Every error from this API has this shape."""

    model_config = ConfigDict(extra="allow", json_schema_extra={"example": {
        "error": "validation_error",
        "message": "The request was not accepted: lead_time must be at least 0 (got -1).",
        "errors": [{"field": "lead_time", "problem": "must be at least 0 (got -1)"}],
        "hint": "See /docs for every field, its allowed values and a working example.",
        "detail": [{"type": "greater_than_equal", "loc": ["body", "lead_time"],
                    "msg": "Input should be greater than or equal to 0"}],
    }})

    error: str = Field(description="Short machine-readable code")
    message: str = Field(description="One or two sentences for a person to read")
    errors: list[Problem] = Field(default_factory=list, description="One item per problem found")
    hint: str = Field(description="Where to look next")
    detail: list[dict] | str | None = Field(default=None, description="FastAPI's usual detail entry, kept for "
                                                                      "clients that expect it")
