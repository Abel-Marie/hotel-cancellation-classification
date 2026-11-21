# Model card: hotel booking cancellation risk

- **Version:** `lgbm-99276044514e`
- **Estimator:** LightGBM, tuned by Bayesian search on all training rows (ONNX export)
- **Endpoint:** [https://hotel-cancellation-risk.vercel.app](https://hotel-cancellation-risk.vercel.app)

## Intended use

Estimate, on the day a booking is made, the probability that it will be cancelled before arrival. Intended
users are hotel revenue managers deciding how far to overbook and which bookings to ask for a deposit or a
confirmation. It is a decision aid for groups of bookings, not a judgement about an individual guest.

**Best use.** Adding the probabilities of all bookings for a hotel and week to forecast how many will
cancel (test error 15%, against 25% without the model), and ranking bookings
by risk (the riskiest tenth cancels 74% of the time, the safest tenth 3%).

**Not intended for:** refusing bookings, pricing individual guests differently, hotels or markets unlike
the two in the training data, or periods long after 2017 without retraining.

## Data

Hotel Booking Demand (Antonio, de Almeida and Nunes, 2019, https://doi.org/10.1016/j.dib.2018.11.126),
CC BY 4.0: 119,390 bookings from a city hotel in Lisbon and a resort in the Algarve, arrivals July
2015 to August 2017. 87,228 remain after removing 31,994 duplicates and impossible records.
Split 70 / 15 / 15 (stratified): 61,058 / 13,085 / 13,085.

## Inputs

Only information available when the booking is made (30 features): hotel, lead time, arrival
date, length of stay, party size, meal plan, market segment, channel, deposit type, customer type, room
type, price per night, agent and company (as "has one" and an agent code), and the guest's booking history.

**Deliberately excluded:** reservation status and its date (the outcome), assigned room, booking changes and
days on the waiting list (known only later), parking spaces, special requests and nationality (recorded or
confirmed at check-in), and arrival year.

## Performance (test set)

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| ROC-AUC | 0.807 | 0.799 to 0.814 |
| PR-AUC | 0.636 | 0.620 to 0.651 |
| Brier score | 0.1487 | |

| Operating point | Threshold | Bookings flagged | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| cost-based (deployed) | 0.63 | 6.5% | 0.766 | 0.812 | 0.193 | 0.311 |
| default 0.5 | 0.50 | 15.1% | 0.773 | 0.661 | 0.362 | 0.467 |
| best F1 | 0.29 | 40.5% | 0.731 | 0.507 | 0.746 | 0.604 |

Baselines (test ROC-AUC): logistic regression 0.774, majority class 0.500. Calibration
step: not needed. On a later year it was not trained on, ROC-AUC is 0.764.

The pre-set target, a ROC-AUC of at least 0.85, was **not met**. With the columns recorded after booking
the score would be about 0.920, but that would not hold in use. The test set was scored twice
(first evaluation 0.797, before the tuning was corrected); see the process document, section 8.1.

## Decision threshold

0.63, chosen on validation to minimise cost when a false alarm costs twice a missed cancellation.
At that threshold the model flags few bookings and is usually right about them. Hotels with different
costs should choose their own threshold from the probability; the model does not change. For example:

| Cost of a false alarm | Threshold | Bookings flagged | Precision | Recall | Cost per booking | Without a model | Saving |
|---|---|---|---|---|---|---|---|
| 0.2 | 0.17 | 53.7% | 0.440 | 0.859 | 0.099 | 0.145 | 31.7% |
| 0.5 | 0.31 | 38.9% | 0.516 | 0.728 | 0.169 | 0.275 | 38.6% |
| 1 | 0.50 | 15.1% | 0.661 | 0.362 | 0.227 | 0.275 | 17.6% |
| 2 | 0.63 | 6.5% | 0.812 | 0.193 | 0.247 | 0.275 | 10.3% |
| 3 | 0.74 | 3.9% | 0.906 | 0.128 | 0.251 | 0.275 | 8.8% |

## Limitations and risks

- Two hotels, one country, two years: performance elsewhere is unknown.
- Drift between years is real (see monitoring); the model needs periodic retraining.
- The split is random, not by time, so the headline score is somewhat optimistic for real use.
- Exact duplicate rows were removed before training; many were probably block bookings, which the model
  therefore sees once each.
- Market segment, agent and customer type correlate with guest groups (for example tour-operator or
  corporate guests). The model is meant for aggregate overbooking decisions; using it to treat individual
  guests differently could disadvantage those groups.
- Nationality is excluded for a data-quality reason (its timing), which also avoids a direct
  nationality-based score.
- The "Non Refund" deposit type shows a counter-intuitively high cancellation rate in this data, probably a
  recording practice for block bookings; the model learns it as recorded.
