import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

hp_features = pd.read_csv('data/for_threshold_analysis.csv')

le = LabelEncoder()
hp_features['gpu_model_encoded'] = le.fit_transform(hp_features['gpu_model'])

FEATURES = [
    'gpu_model_encoded',
    'cpu_request',
    'gpu_request',
    'worker_num',
    'concurrent_spot_jobs',
    'concurrent_spot_workers',
]

DEVIATION_SWEEP  = [0.2, 0.3, 0.5, 0.7, 1.0]
WORKERS_SWEEP    = [200, 387, 500, 700, 935]
FIXED_WORKERS    = 387
FIXED_DEVIATION  = 0.5

def run_sensitivity(dev_thresh, workers_thresh, hp_features, features):

    temp_label = (
        (hp_features['deviation_capped'] > dev_thresh) &
        (hp_features['concurrent_spot_workers'] > workers_thresh)
    ).astype(int)

    label_pct = temp_label.mean() * 100

    # assigning label range to avoid highly imbalanced dataset
    if label_pct < 5 or label_pct > 40:
        return {
            'deviation_thresh' : dev_thresh,
            'workers_thresh'   : workers_thresh,
            'label_1_pct'      : round(label_pct, 1),
            'f1'               : None,
            'precision'        : None,
            'recall'           : None,
            'top_feature'      : None,
            'note'             : f'SKIPPED - label={label_pct:.1f}% outside valid range'
        }

    X = hp_features[features]
    y = temp_label

    # temporal split — no shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    importances = pd.Series(rf.feature_importances_, index=features)
    top_feature = importances.idxmax()

    return {
        'deviation_thresh' : dev_thresh,
        'workers_thresh'   : workers_thresh,
        'label_1_pct'      : round(label_pct, 1),
        'f1'               : round(f1_score(y_test, y_pred), 4),
        'precision'        : round(precision_score(y_test, y_pred), 4),
        'recall'           : round(recall_score(y_test, y_pred), 4),
        'top_feature'      : top_feature,
        'note'             : 'OK'
    }

# SWEEP 1: Varying deviation with fix workers at 387
print("=" * 70)
print("SWEEP 1: Varying Deviation Threshold (spot workers fixed at 387)")
print("=" * 70)

sweep1_results = []
for dev in DEVIATION_SWEEP:
    print(f"  Running deviation={dev}, workers={FIXED_WORKERS}")
    result = run_sensitivity(dev, FIXED_WORKERS, hp_features, FEATURES)
    sweep1_results.append(result)
    if result['note'] == 'OK':
        print(f"  → label={result['label_1_pct']}%, F1={result['f1']}, top={result['top_feature']}")
    else:
        print(f"  → {result['note']}")

# SWEEP 2: Varying workers with fix deviation at 0.5
print("\n" + "=" * 70)
print("SWEEP 2: Varying Spot Workers Threshold (deviation fixed at 0.5)")
print("=" * 70)

sweep2_results = []
for workers in WORKERS_SWEEP:
    print(f"  Running deviation={FIXED_DEVIATION}, workers={workers}")
    result = run_sensitivity(FIXED_DEVIATION, workers, hp_features, FEATURES)
    sweep2_results.append(result)
    if result['note'] == 'OK':
        print(f"  → label={result['label_1_pct']}%, F1={result['f1']}, top={result['top_feature']}")
    else:
        print(f"  → {result['note']}")

# summary tables
print("\n" + "=" * 100)
print("SUMMARY TABLE 1: Varying Deviation Threshold")
print("=" * 100)
sweep1_df = pd.DataFrame(sweep1_results)
print(sweep1_df[['deviation_thresh', 'label_1_pct',
                  'f1', 'precision', 'recall',
                  'top_feature', 'note']].to_string(index=False))

print("\n" + "=" * 100)
print("SUMMARY TABLE 2: Varying Spot Workers Threshold")
print("=" * 100)
sweep2_df = pd.DataFrame(sweep2_results)
print(sweep2_df[['workers_thresh', 'label_1_pct',
                  'f1', 'precision', 'recall',
                  'top_feature', 'note']].to_string(index=False))