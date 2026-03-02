from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE
from typing import Tuple
import pandas as pd
from pathlib import Path

LEAKAGE_FEATURES = ['Complain', 'Satisfaction Score', 'Point Earned']
DROP_FEATURES = ['Exited', 'RowNumber', 'CustomerId', 'Surname']


def load_data(path: Path) -> Tuple[pd.DataFrame]:
    df = pd.read_csv(path)
    return df


def split_data(df:pd.DataFrame,test_size=0.25) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    X = df.drop(DROP_FEATURES+LEAKAGE_FEATURES, axis=1)
    y = df["Exited"]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, y_train, X_test, y_test, numeric_features, categorical_features