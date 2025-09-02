import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm
from OOA import OspreyOptimizer
import joblib
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'

# GPU
OOA_GPU_PARAMS = {
    'device': 'gpu',
    'gpu_device_id': 0,
    'force_col_wise': True,
    'force_row_wise': False
}

# GPU check
def check_gpu_for_ooa():
    try:
        import lightgbm as lgb
        test_model = LGBMRegressor(
            n_estimators=1,
            verbose=-1,
            **OOA_GPU_PARAMS
        )
        X_test = np.random.rand(10, 2)
        y_test = np.random.rand(10)
        test_model.fit(X_test, y_test)
        print("✅ GPUs can be used for OOA")
        return True
    except Exception as e:
        print(f"❌ GPU is unavailable, OOA optimization will use CPU.: {e}")
        return False
gpu_available_for_ooa = check_gpu_for_ooa()
if not gpu_available_for_ooa:
    OOA_GPU_PARAMS = {}

# Read dataset
data = pd.read_csv(r"C:\Yourfiles\XCH4.csv")
data['ws'] = np.sqrt(data['u10'] ** 2 + data['v10'] ** 2)
features = ['elevation', 'blh', 'ssr', 'ws', 't2m', 'tp', 'e', 'swvl', 'evi', 'ntl']
target = 'xch4'
X = data[features]
y = data[target]

# Dataset Partitioning
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.222, random_state=42)

print(f"Dataset partitioning completed：")
print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# LightGBM
def objective_function(params):
    num_leaves, learning_rate, feature_fraction, bagging_fraction, max_depth = params
    model = LGBMRegressor(
        num_leaves=int(num_leaves),
        learning_rate=learning_rate,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        max_depth=int(max_depth),
        objective='regression',
        metric='rmse',
        verbose=-1,
        random_state=42,
        **OOA_GPU_PARAMS
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

# OOA Parameter Settings
pop_size = 20
dim = 5
lb = np.array([32, 0.01, 0.1, 0.5, 5])
ub = np.array([150, 0.1, 1.0, 1.0, 20])
max_iter = 50
choice = 'yes'

if choice.lower() == 'yes':
    print("✅ OOA running")
    if gpu_available_for_ooa:
        print("✅ OOA utilizes GPU acceleration")
        print(f"Estimated number of training models: {pop_size * max_iter} models")
    else:
        print("OOA utilizes CPU")
    param_names = ['num_leaves', 'learning_rate', 'feature_fraction', 'bagging_fraction', 'max_depth']
    param_types = {0: int, 4: int}
    optimizer = OspreyOptimizer(
        pop_size=pop_size,
        dim=dim,
        lb=lb,
        ub=ub,
        max_iter=max_iter,
        objective_function=objective_function,
        param_names=param_names,
        param_types=param_types
    )
    GbestScore, GbestPosition, Curve = optimizer.optimize()
    best_num_leaves, best_learning_rate, best_feature_fraction, best_bagging_fraction, best_max_depth = GbestPosition
    print("\nOptimal parameters：")
    print("num_leaves:", int(best_num_leaves))
    print("learning_rate:", best_learning_rate)
    print("feature_fraction:", best_feature_fraction)
    print("bagging_fraction:", best_bagging_fraction)
    print("max_depth:", int(best_max_depth))
    print("RMSE:", GbestScore)
    best_fitness_curve = [Curve[i] for i in range(max_iter)]

    # Convergence rate curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_iter + 1), best_fitness_curve, marker='o', linestyle='-', markersize=5, color='#1f77b4')
    plt.title("Convergence rate curve", fontsize=16)
    plt.xlabel("Number of iterations", fontsize=12)
    plt.ylabel("Optimal fitness value", fontsize=12)
    plt.grid(True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    np.save(r"C:\Yourfiles\XCH4.npy", best_fitness_curve)
    print("✅ Convergence rate curve is saved")
    plt.show(block=True)

    # Train LightGBM using optimal parameters
    model = LGBMRegressor(
        num_leaves=int(best_num_leaves),
        learning_rate=best_learning_rate,
        feature_fraction=best_feature_fraction,
        bagging_fraction=best_bagging_fraction,
        max_depth=int(best_max_depth),
        objective='regression',
        metric=['mse', 'mae', 'rmse']
    )
    model.fit(X_train, y_train)

    # KF cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scores, rmse_scores, mae_scores = [], [], []
    all_y_test, all_y_pred = [], []
    shap_values_list = []
    print("\nKF cross-validation")
    for fold, (train_index, val_index) in enumerate(tqdm(kf.split(X_temp), total=kf.get_n_splits(), desc="Cross-Validation Progress")):
        X_fold_train, X_fold_val = X_temp.iloc[train_index], X_temp.iloc[val_index]
        y_fold_train, y_fold_val = y_temp.iloc[train_index], y_temp.iloc[val_index]

        model.fit(X_fold_train, y_fold_train)
        preds = model.predict(X_fold_val)

        r2 = r2_score(y_fold_val, preds)
        rmse = np.sqrt(mean_squared_error(y_fold_val, preds))
        mae = mean_absolute_error(y_fold_val, preds)

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)

        all_y_test.extend(y_fold_val)
        all_y_pred.extend(preds)

    print("\nCross-validation results：")
    print(f"Average R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
    print(f"Average MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")

    # Evaluate on the independent test set
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\ntest set results：")
    print(f"R²: {test_r2:.3f}")
    print(f"RMSE: {test_rmse:.3f}")
    print(f"MAE: {test_mae:.3f}")
    joblib.dump(model, r"C:\Yourfiles\OOA-LightGBM.pkl")
    print("✅ OOA-LightGBM saved")

    # Perform global predictions using the trained model
    model = joblib.load(r"C:\Yourfiles\OOA-LightGBM.pkl")
    input_csv = r"C:\Yourfiles\features.csv"
    data = pd.read_csv(input_csv)
    data['ws'] = np.sqrt(data['u10'] ** 2 + data['v10'] ** 2)
    features = ['elevation', 'blh', 'ssr', 'ws', 't2m', 'tp', 'e', 'swvl', 'evi', 'ntl']
    data['xch4'] = model.predict(data[features])
    output_csv = r"C:\Yourfiles\OOA-LightGBM.csv"
    data.to_csv(output_csv, index=False)
    print(f"✅ Nationwide prediction results have been saved as {output_csv}")