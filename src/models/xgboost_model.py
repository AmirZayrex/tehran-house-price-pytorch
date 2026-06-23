
# TODO: Working on it..

import xgboost as xgb

def get_xgboost_model(
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> xgb.XGBRegressor:
    """
    ساخت مدل XGBoost برای پیش‌بینی LogPrice

    Parameters:
    n_estimators : تعداد درخت‌ها
    max_depth    : حداکثر عمق هر درخت
    learning_rate: نرخ یادگیری (shrinkage)
    subsample    : نسبت داده برای هر درخت
    colsample_bytree: نسبت feature برای هر درخت
    random_state : seed برای reproducibility
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        verbosity=0
    )
    return model