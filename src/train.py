from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,StratifiedKFold,TunedThresholdClassifierCV
import joblib

from src.data import load_data, split_data
from src.preprocessing import build_preprocessor
from src.models import build_gradient_boosting_model, get_gb_param_grid
from src.evaluate import evaluate_on_test
from src.config import *

def main():
    df = load_data(DATA_PATH)

    X_train, y_train, X_test, y_test,numeric_features, categorical_features = split_data(
        df,
        TEST_SIZE,
    )
    
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    model = build_gradient_boosting_model()

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])


    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = GridSearchCV(
        estimator=pipe,
        param_grid=get_gb_param_grid(),
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    tt = TunedThresholdClassifierCV(
        estimator=best_model,
        cv=cv,
        scoring='f1'
    )

    tt.fit(X_train, y_train)

    meta = {
        "best_params": search.best_params_,
        "threshold": tt.best_threshold_,
        "cv_score": search.best_score_,
    }

    joblib.dump(meta, ARTIFACTS_DIR + "meta.joblib")

    metrics = evaluate_on_test(tt, X_test, y_test, label="Gradient Boosting with Tuned Threshold")

    print(metrics)

    joblib.dump(tt, ARTIFACTS_DIR + "model.joblib")


if __name__ == "__main__":
    main()