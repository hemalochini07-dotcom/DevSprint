import pandas as pd
import numpy as np


REQUIRED = [
    "date",
    "campaign_id",
    "spend",
    "clicks",
    "impressions",
    "conversions"
]

ALERT_COLUMNS = [
    "date",
    "campaign_id",
    "metric",
    "observed",
    "baseline",
    "change_pct",
    "severity",
    "category",
    "explanation",
    "recommended_action",
    "ground_truth_incident"
]


def detect_anomalies(df):
    """
    Detect anomalies using each campaign's past observations only.
    Uses NumPy arrays to reduce repeated pandas row operations.
    """

    if df.empty:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    data = df.copy()

    # Validate required columns
    missing = [col for col in REQUIRED if col not in data.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    # Prepare data
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["date", "campaign_id"]
    )

    data = data.sort_values(
        ["campaign_id", "date"]
    )

    results = []

    metrics_to_check = [
        "spend",
        "clicks",
        "conversions"
    ]

    # Process one campaign at a time
    for campaign, group in data.groupby(
        "campaign_id",
        sort=False
    ):

        group = group.reset_index(drop=True)

        if len(group) <= 14:
            continue

        # Convert metrics to numeric arrays once per campaign
        metric_values = {}

        for metric in metrics_to_check:
            metric_values[metric] = pd.to_numeric(
                group[metric],
                errors="coerce"
            ).to_numpy(dtype=float)

        dates = group["date"].to_numpy()

        if "ground_truth_incident" in group.columns:
            incidents = (
                group["ground_truth_incident"]
                .fillna(False)
                .astype(bool)
                .to_numpy()
            )
        else:
            incidents = np.zeros(len(group), dtype=bool)

        # Evaluate each row using past observations only
        for i in range(14, len(group)):

            history_start = max(0, i - 28)

            checks = []

            for metric in metrics_to_check:

                history = metric_values[metric][history_start:i]
                history = history[np.isfinite(history)]

                value = metric_values[metric][i]

                if len(history) < 7 or not np.isfinite(value):
                    continue

                # Historical robust baseline
                baseline = float(np.median(history))

                mad = float(
                    np.median(np.abs(history - baseline))
                )

                threshold = max(
                    3.5 * 1.4826 * mad,
                    abs(baseline) * 0.35,
                    1.0
                )

                deviation = float(value - baseline)

                if abs(deviation) > threshold:

                    if baseline != 0:
                        pct = (
                            (value - baseline)
                            / abs(baseline)
                            * 100
                        )
                    else:
                        pct = None

                    checks.append(
                        (
                            metric,
                            float(value),
                            baseline,
                            pct,
                            deviation
                        )
                    )

            if not checks:
                continue

            # Prioritise conversion collapse, then spend movement
            checks.sort(
                key=lambda item: (
                    0
                    if item[0] == "conversions"
                    and item[2] > 0
                    and item[1] < item[2] * 0.25
                    else 1
                    if item[0] == "spend"
                    else 2
                )
            )

            metric, value, baseline, pct, deviation = checks[0]

            incident = bool(incidents[i])

            # Classify anomaly
            if (
                metric == "conversions"
                and baseline > 0
                and value <= baseline * 0.25
            ):

                category = (
                    "Possible tracking failure / conversion drop"
                )

                severity = (
                    "Critical" if value == 0 else "Warning"
                )

                action = (
                    "Check the conversion tag, event logs, "
                    "and recent tracking changes."
                )

            elif metric == "spend" and value > baseline * 1.5:

                category = "Unexpected spend increase"

                severity = (
                    "Critical"
                    if value > baseline * 2
                    else "Warning"
                )

                action = (
                    "Review daily budget, bid strategy, "
                    "and campaign change history."
                )

            elif metric == "spend" and value < baseline * 0.5:

                category = "Unexpected spend drop"

                severity = "Warning"

                action = (
                    "Check campaign delivery, budget limits, "
                    "and ad approval status."
                )

            else:

                category = f"Unusual {metric} movement"

                severity = "Warning"

                action = (
                    f"Compare {metric} with recent campaign "
                    "settings and related metrics."
                )

            # Explanation
            if pct is not None:
                pct_text = f"{pct:+.1f}%"
            else:
                pct_text = (
                    "not calculable from a zero baseline"
                )

            explanation = (
                f"{campaign}: {metric} is {value:,.1f}, "
                f"compared with a historical median of "
                f"{baseline:,.1f} ({pct_text}). "
                "This is unusual relative to the campaign's "
                "recent history; the underlying cause is not confirmed."
            )

            results.append({
                "date": pd.Timestamp(dates[i]).date().isoformat(),
                "campaign_id": campaign,
                "metric": metric,
                "observed": round(value, 2),
                "baseline": round(baseline, 2),
                "change_pct": (
                    round(pct, 2)
                    if pct is not None
                    else None
                ),
                "severity": severity,
                "category": category,
                "explanation": explanation,
                "recommended_action": action,
                "ground_truth_incident": incident
            })

    # No anomalies found
    if not results:
        return pd.DataFrame(columns=ALERT_COLUMNS)

    alerts = pd.DataFrame(results)

    # Latest alert per campaign and category
    alerts = alerts.sort_values(
        ["date", "severity"],
        ascending=[False, True]
    )

    alerts = alerts.drop_duplicates(
        subset=["campaign_id", "category"],
        keep="first"
    )

    return alerts.reset_index(drop=True)


def evaluate_detection(df, alerts):
    """
    Evaluate detection only when ground-truth labels are available.
    """

    if (
        "ground_truth_incident" not in df.columns
        or df["ground_truth_incident"].isna().all()
    ):
        return {
            "precision": None,
            "recall": None,
            "false_positive_rate": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0
        }

    labelled = df.copy()

    labelled["date"] = pd.to_datetime(
        labelled["date"],
        errors="coerce"
    )

    labelled["ground_truth_incident"] = (
        labelled["ground_truth_incident"]
        .fillna(False)
        .astype(bool)
    )

    # Use campaign and date as the comparison key
    detected_keys = set()

    if not alerts.empty:

        for _, row in alerts.iterrows():

            detected_date = pd.to_datetime(
                row["date"],
                errors="coerce"
            )

            if pd.notna(detected_date):
                detected_keys.add(
                    (
                        str(row["campaign_id"]),
                        detected_date.date().isoformat()
                    )
                )

    valid_labelled = labelled.dropna(
        subset=["date", "campaign_id"]
    )

    actual = set()

    for _, row in valid_labelled[
        valid_labelled["ground_truth_incident"]
    ].iterrows():

        actual.add(
            (
                str(row["campaign_id"]),
                row["date"].date().isoformat()
            )
        )

    # Confusion matrix counts
    tp = len(detected_keys & actual)
    fp = len(detected_keys - actual)
    fn = len(actual - detected_keys)

    total_non_incident = int(
        (~valid_labelled["ground_truth_incident"]).sum()
    )

    tn = max(total_non_incident - fp, 0)

    # Evaluation metrics
    precision = (
        tp / (tp + fp)
        if tp + fp
        else None
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else None
    )

    false_positive_rate = (
        fp / (fp + tn)
        if fp + tn
        else None
    )

    return {
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }