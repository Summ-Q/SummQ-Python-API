# AI & Data Science
# Data Science - Retention Prediction

A small prediction service that estimates whether a flashcard needs to be reviewed and suggests a suitable next review interval.

## Model

The model uses a Random Forest classifier with:

- `50` trees
    
- `max_depth=6`
    
- `class_weight='balanced'`
    

### Input features

- `past_reviews_count` — number of previous reviews
    
- `avg_score` — average review score, from 1 to 4
    
- `days_since_last_review` — time since the previous review
    
- `overdue_ratio` — derived from:  
    `days_since_last_review / (past_reviews_count + 1)`
    

The model was trained using a sample from the [FSRS dataset](https://huggingface.co/datasets/open-spaced-repetition/fsrs-dataset), with the data split by card to keep reviews from the same card together

A decision threshold of `0.45` is used for the retention prediction. It was selected during validation with more focus on recall.


More details about model experiments, comparisons, and the interval calculation can be found in `ds_model_development/model_experiments.md`.

## API

### `POST /ds/predict-retention`

**Request:**

```json
{
  "past_reviews_count": 5,
  "avg_score": 3.2,
  "days_since_last_review": 8.0
}
```

**Response:**

```json
{
  "status": "success",
  "result": {
    "needs_review_today": false,
    "suggested_interval_days": 12
  }
}
```

### Response fields

- `needs_review_today` — whether the card is predicted to need a review now
    
- `suggested_interval_days` — suggested number of days until the next review
    

## Setup

The required packages are already included in the shared `requirements.txt`.

The main dependencies for this part are:

- `scikit-learn==1.6.1`
    
- `pandas`
    
- `joblib`
    

No extra model setup is required. `predictor.py` loads the trained model from disk when the service starts.