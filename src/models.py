from sklearn.ensemble import GradientBoostingClassifier
from src.config import RANDOM_STATE
from typing import Dict


def build_gradient_boosting_model() -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    return model

def get_gb_param_grid() -> Dict:
    param_grid = {
        "model__learning_rate":[0.03,0.05,0.1],
        "model__max_depth":[2,3,4,5],
        "model__max_leaf_nodes":[15,31,63],
    }
    return param_grid