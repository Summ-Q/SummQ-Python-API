
import os
import joblib
import pandas as pd
 
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'retention_model.pkl')
model = joblib.load(MODEL_PATH)

DECISION_THRESHOLD = 0.45
 
# Interval growth multipliers, based on observed patterns in the training dataset (not assumed)

MODERATE_CONFIDENCE_MULTIPLIER = 2.55
HIGH_CONFIDENCE_MULTIPLIER = 3.11

MAX_INTERVAL_DAYS = 60
 
 
def predict_card_retention(
    past_reviews_count: int,
    avg_score: float,
    days_since_last_review: float,
) -> dict:
    
    overdue_ratio = days_since_last_review / (past_reviews_count + 1)
 
    features = pd.DataFrame([{
        "past_reviews_count": past_reviews_count,
        "avg_score": avg_score,
        "days_since_last_review": days_since_last_review,
        "overdue_ratio": overdue_ratio,
    }])
 
    forget_probability = float(model.predict_proba(features)[0][1])
    needs_review_today = forget_probability >= DECISION_THRESHOLD
 
    # Convert the probability into a concrete interval
    # Uses the current gap "days_since_last_review" as the baseline "current interval"
    current_interval = max(days_since_last_review, 1)
 
    if needs_review_today:
        suggested_interval_days = 1
    elif forget_probability < 0.10:
    
        # observed interval growth was 3.11x 
        suggested_interval_days = round(current_interval * HIGH_CONFIDENCE_MULTIPLIER)
    else:
        # Moderate-confidence multiplier (avg_score 2.5-4.0) — median observed growth was 2.55x
        suggested_interval_days = round(current_interval * MODERATE_CONFIDENCE_MULTIPLIER)

 
    # ensure the suggested interval does not exceed the max
    suggested_interval_days = min(suggested_interval_days, MAX_INTERVAL_DAYS)
 
    return {
        "needs_review_today": bool(needs_review_today),
        "suggested_interval_days": int(suggested_interval_days),
    }
 