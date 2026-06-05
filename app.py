from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import traceback
import io
import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve, RandomizedSearchCV
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

app = Flask(__name__)
_model_state = {}

# ─────────────────────────────────────────────
# Supported formats
# ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    '.csv', '.tsv', '.txt',
    '.xlsx', '.xls', '.xlsm', '.xlsb', '.ods',
    '.json', '.parquet', '.feather',
    '.pkl', '.pickle', '.h5', '.hdf5',
    '.dta', '.sas7bdat', '.sav'
}

EXTENSION_READERS = {
    '.csv':      lambda b, _: pd.read_csv(io.BytesIO(b), encoding='latin1'),
    '.tsv':      lambda b, _: pd.read_csv(io.BytesIO(b), sep='\t', encoding='latin1'),
    '.txt':      lambda b, _: pd.read_csv(io.BytesIO(b), sep=None, engine='python', encoding='latin1'),
    '.xlsx':     lambda b, _: pd.read_excel(io.BytesIO(b), engine='openpyxl'),
    '.xls':      lambda b, _: pd.read_excel(io.BytesIO(b), engine='xlrd'),
    '.xlsm':     lambda b, _: pd.read_excel(io.BytesIO(b), engine='openpyxl'),
    '.xlsb':     lambda b, _: pd.read_excel(io.BytesIO(b), engine='pyxlsb'),
    '.ods':      lambda b, _: pd.read_excel(io.BytesIO(b), engine='odf'),
    '.json':     lambda b, _: pd.read_json(io.BytesIO(b)),
    '.parquet':  lambda b, _: pd.read_parquet(io.BytesIO(b)),
    '.feather':  lambda b, _: pd.read_feather(io.BytesIO(b)),
    '.pkl':      lambda b, _: pd.read_pickle(io.BytesIO(b)),
    '.pickle':   lambda b, _: pd.read_pickle(io.BytesIO(b)),
    '.h5':       lambda b, _: pd.read_hdf(io.BytesIO(b)),
    '.hdf5':     lambda b, _: pd.read_hdf(io.BytesIO(b)),
    '.dta':      lambda b, _: pd.read_stata(io.BytesIO(b)),
    '.sas7bdat': lambda b, _: pd.read_sas(io.BytesIO(b), format='sas7bdat'),
    '.sav':      lambda b, _: pd.read_spss(io.BytesIO(b)),
}

MAX_ROWS_FOR_SLOW_OPS = 5000
MAX_ROWS_TUNING = 3000


def read_file_to_df(raw_bytes, filename):
    ext = os.path.splitext(filename.lower())[1]
    if ext not in EXTENSION_READERS:
        raise ValueError(f"Unsupported file format '{ext}'.")
    df = EXTENSION_READERS[ext](raw_bytes, filename)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("File did not produce a tabular DataFrame.")
    df = df.loc[:, ~df.columns.astype(str).str.match(r'^Unnamed')]
    return df


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def safe_float(v):
    try:
        f = float(v)
        return 0.0 if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return 0.0


def _train_one(args):
    name, model, X_train, X_test, y_train, y_test = args
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 3)
    pred = model.predict(X_test)
    return name, {
        "acc":        safe_float(accuracy_score(y_test, pred)),
        "precision":  safe_float(precision_score(y_test, pred, average='weighted', zero_division=0)),
        "recall":     safe_float(recall_score(y_test, pred, average='weighted', zero_division=0)),
        "f1":         safe_float(f1_score(y_test, pred, average='weighted', zero_division=0)),
        "train_time": train_time,
        "pred":       pred.tolist()
    }


def detect_failure_reasons(result_data):
    reasons, tips = [], []
    acc = result_data.get("test_acc", 0)
    train_acc = result_data.get("train_acc", 0)
    cv = result_data.get("cv_score", 0)
    rows = result_data["dataset_info"]["rows"]
    missing = result_data["dataset_info"]["missing"]
    imbalance = result_data.get("imbalance_ratio", 1.0)
    roc_auc = result_data.get("roc_auc", 0)

    if train_acc - acc > 0.15:
        reasons.append({"icon": "fire", "title": "Overfitting detected", "severity": "high",
            "detail": f"Training acc {train_acc:.1%} vs test acc {acc:.1%} — gap of {(train_acc - acc):.1%}. Model memorised training data."})
        tips.append("Use regularisation (C↓ for LR/SVM), reduce tree depth, or add pruning.")

    if acc < 0.60 and train_acc < 0.65:
        reasons.append({"icon": "trending-down", "title": "Underfitting", "severity": "high",
            "detail": f"Both train ({train_acc:.1%}) and test ({acc:.1%}) accuracy are low. Model is too simple."})
        tips.append("Try a more complex model (Gradient Boosting, Random Forest). Add more features or engineer new ones.")

    if cv < acc - 0.10:
        reasons.append({"icon": "chart-bar", "title": "High variance — unstable", "severity": "medium",
            "detail": f"Cross-val score {cv:.3f} vs test acc {acc:.3f}. Results vary across folds."})
        tips.append("Increase training data, use ensemble methods, or apply stratified k-fold splitting.")

    if imbalance < 0.3:
        reasons.append({"icon": "scale", "title": "Severe class imbalance", "severity": "high",
            "detail": f"Minority/majority ratio = {imbalance:.2f}. Model likely predicts majority class most of the time."})
        tips.append("Use SMOTE oversampling, class_weight='balanced', or try F1/AUC as your primary metric.")
    elif imbalance < 0.7:
        reasons.append({"icon": "alert-triangle", "title": "Moderate class imbalance", "severity": "medium",
            "detail": f"Minority/majority ratio = {imbalance:.2f}. Some bias toward majority class possible."})
        tips.append("Consider class_weight='balanced' or stratified splits.")

    if missing > 0:
        pct = missing / (rows * result_data["dataset_info"]["columns"]) * 100
        reasons.append({"icon": "question-mark", "title": f"Missing values ({pct:.1f}%)",
            "severity": "medium" if pct > 10 else "low",
            "detail": f"{missing} missing cells found. Median imputation was used, which may introduce bias."})
        tips.append("Use KNN or iterative imputation for better missing-value handling.")

    if rows < 200:
        reasons.append({"icon": "database", "title": "Small dataset", "severity": "medium",
            "detail": f"Only {rows} rows. Small datasets lead to noisy, unreliable model performance."})
        tips.append("Collect more data. Use data augmentation or cross-validation for more reliable estimates.")

    if 0 < roc_auc < 0.70:
        reasons.append({"icon": "target", "title": "Poor discrimination (AUC)", "severity": "high",
            "detail": f"ROC-AUC = {roc_auc:.3f}. Model barely outperforms random guessing (0.5)."})
        tips.append("Try feature engineering, hyperparameter tuning, or a completely different algorithm.")

    if not reasons:
        reasons.append({"icon": "circle-check", "title": "No critical failures found", "severity": "none",
            "detail": f"Test accuracy {acc:.1%}, AUC {roc_auc:.3f}, balanced classes. Model appears healthy."})
        tips.append("Consider trying Gradient Boosting or Neural Networks for potentially higher performance.")

    return reasons, tips


def compute_dataset_profile(df_original):
    profile = []
    for col in df_original.columns:
        s = df_original[col]
        dtype = str(s.dtype)
        is_num = pd.api.types.is_numeric_dtype(s)
        entry = {"name": col, "dtype": dtype,
                 "missing": int(s.isnull().sum()), "unique": int(s.nunique())}
        if is_num:
            entry.update({
                "mean": safe_float(s.mean()), "std": safe_float(s.std()),
                "min":  safe_float(s.min()),  "max": safe_float(s.max()),
                "median": safe_float(s.median()),
                "q25": safe_float(s.quantile(0.25)),
                "q75": safe_float(s.quantile(0.75)),
                "skew": safe_float(s.skew()),
                "hist": [safe_float(v) for v in np.histogram(s.dropna(), bins=10)[0].tolist()],
            })
        else:
            top = s.value_counts().head(5).to_dict()
            entry["top_values"] = {str(k): int(v) for k, v in top.items()}
        profile.append(entry)
    return profile


def compute_correlation(df_num):
    if df_num.shape[1] < 2:
        return [], []
    corr = df_num.corr().round(3)
    return corr.columns.tolist(), corr.values.tolist()


def compute_learning_curve(model, X, y):
    try:
        n = len(X)
        if n > MAX_ROWS_FOR_SLOW_OPS:
            idx = np.random.choice(n, MAX_ROWS_FOR_SLOW_OPS, replace=False)
            X, y = X[idx], y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
        sizes = np.linspace(0.15, 1.0, 5)
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=sizes,
            cv=3, scoring='accuracy', n_jobs=-1
        )
        return {
            "sizes": [int(s) for s in train_sizes.tolist()],
            "train": [safe_float(s.mean()) for s in train_scores],
            "val":   [safe_float(s.mean()) for s in val_scores],
            "train_std": [safe_float(s.std()) for s in train_scores],
            "val_std":   [safe_float(s.std()) for s in val_scores],
        }
    except Exception:
        return {"sizes": [], "train": [], "val": [], "train_std": [], "val_std": []}


def compute_per_class_metrics(y_test, pred, class_labels):
    """Compute precision/recall/f1 per class."""
    try:
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        result = []
        for label in class_labels:
            if label in report:
                result.append({
                    "label": label,
                    "precision": safe_float(report[label]["precision"]),
                    "recall":    safe_float(report[label]["recall"]),
                    "f1":        safe_float(report[label]["f1-score"]),
                    "support":   int(report[label]["support"])
                })
        return result
    except Exception:
        return []


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    global _model_state
    file = request.files.get("file")

    if not file or not file.filename:
        return jsonify({"error": "No file uploaded."}), 400

    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format '{ext}'."}), 400

    try:
        t_start = time.time()
        raw_bytes   = file.read()
        df_original = read_file_to_df(raw_bytes, file.filename)

        if df_original.shape[0] < 10:
            return jsonify({"error": "Dataset too small (< 10 rows)."}), 400
        if df_original.shape[1] < 2:
            return jsonify({"error": "Dataset must have at least 2 columns."}), 400

        rows, cols     = df_original.shape
        missing_before = int(df_original.isnull().sum().sum())

        dataset_profile = compute_dataset_profile(df_original)
        corr_labels, corr_matrix = compute_correlation(
            df_original.select_dtypes(include=[np.number])
        )

        target_col = df_original.columns[-1]
        class_dist = {str(k): int(v)
                      for k, v in df_original[target_col].value_counts().items()}

        # ── Preprocessing ──
        df = df_original.copy()
        df = df.dropna(thresh=int(rows * 0.5), axis=1)

        encoders = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            df[col] = df[col].astype(np.int64) // 10**9

        df = df.fillna(df.median(numeric_only=True))

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        feature_names = list(X.columns)

        class_counts    = y.value_counts()
        imbalance_ratio = float(class_counts.min() / class_counts.max())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42,
            stratify=y if len(class_counts) > 1 else None
        )

        scaler   = StandardScaler()
        X_train_ = scaler.fit_transform(X_train)
        X_test_  = scaler.transform(X_test)
        X_all    = scaler.transform(X)

        # ── Train all 7 models in parallel ──
        model_zoo = {
            "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=300, C=1, solver='saga', n_jobs=-1, random_state=42),
            "SVM":                 SVC(kernel='linear', probability=False, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
            "Naive Bayes":         GaussianNB(),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=50, max_depth=4, subsample=0.8, random_state=42),
        }

        tasks = [(n, m, X_train_, X_test_, y_train, y_test) for n, m in model_zoo.items()]
        results = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for name, metrics in pool.map(_train_one, tasks):
                results[name] = metrics

        best_model_name = max(results, key=lambda x: results[x]["acc"])
        best_model_obj  = model_zoo[best_model_name]
        best_model_obj.fit(X_train_, y_train)

        # ── Feature Importance ──
        if hasattr(best_model_obj, "feature_importances_"):
            imp_scores = best_model_obj.feature_importances_.tolist()
        elif hasattr(best_model_obj, "coef_"):
            imp_scores = np.abs(best_model_obj.coef_).mean(axis=0).tolist()
        else:
            n_test = len(X_test_)
            if n_test > 500:
                idx = np.random.choice(n_test, 500, replace=False)
                X_perm = X_test_[idx]
                y_perm = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
            else:
                X_perm, y_perm = X_test_, y_test
            perm = permutation_importance(best_model_obj, X_perm, y_perm,
                                          n_repeats=2, random_state=42, n_jobs=-1)
            imp_scores = np.abs(perm.importances_mean).tolist()

        feat_imp = sorted(
            zip(feature_names, [safe_float(v) for v in imp_scores]),
            key=lambda x: x[1], reverse=True
        )[:15]

        # ── Confusion matrix & ROC ──
        pred_best    = best_model_obj.predict(X_test_)
        cm           = confusion_matrix(y_test, pred_best).tolist()
        class_labels = [str(c) for c in sorted(y.unique())]

        # Per-class metrics
        per_class = compute_per_class_metrics(y_test, pred_best, class_labels)

        roc_fpr, roc_tpr, roc_auc_val = [], [], 0.0
        is_binary = len(class_counts) == 2
        best_for_proba = best_model_obj
        if is_binary and not hasattr(best_model_obj, "predict_proba"):
            best_for_proba = SVC(kernel='linear', probability=True, random_state=42)
            best_for_proba.fit(X_train_, y_train)
        if is_binary and hasattr(best_for_proba, "predict_proba"):
            try:
                probs = best_for_proba.predict_proba(X_test_)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, probs)
                roc_auc_val = safe_float(auc(fpr, tpr))
                step = max(1, len(fpr) // 100)
                roc_fpr = [safe_float(v) for v in fpr[::step]]
                roc_tpr = [safe_float(v) for v in tpr[::step]]
                # Optimal threshold (Youden's J)
                j_scores = tpr - fpr
                best_thresh_idx = int(np.argmax(j_scores))
                optimal_threshold = safe_float(thresholds[best_thresh_idx])
            except Exception:
                optimal_threshold = 0.5
        else:
            optimal_threshold = 0.5

        # ── Cross-validation ──
        n_total = len(X_all)
        X_cv, y_cv = X_all, y
        if n_total > MAX_ROWS_FOR_SLOW_OPS:
            idx = np.random.choice(n_total, MAX_ROWS_FOR_SLOW_OPS, replace=False)
            X_cv = X_all[idx]
            y_cv = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]

        cv_scores = cross_val_score(best_model_obj, X_cv, y_cv, cv=3, scoring="accuracy", n_jobs=-1)
        cv_mean = safe_float(cv_scores.mean())
        cv_std  = safe_float(cv_scores.std())
        cv_each = [safe_float(v) for v in cv_scores]
        while len(cv_each) < 5:
            cv_each.append(cv_each[-1] if cv_each else 0.0)

        train_acc = safe_float(best_model_obj.score(X_train_, y_train))
        test_acc  = safe_float(best_model_obj.score(X_test_, y_test))

        # ── Learning curve ──
        lc = compute_learning_curve(best_model_obj, X_cv, y_cv)

        # ── Feature Engineering suggestions ──
        fe_suggestions = []
        df_num = df.iloc[:, :-1].select_dtypes(include=[np.number])
        if df_num.shape[1] >= 2:
            corr_mat = df_num.corr().abs()
            for i in range(len(corr_mat.columns)):
                for j in range(i + 1, len(corr_mat.columns)):
                    v = safe_float(corr_mat.iloc[i, j])
                    if v > 0.90:
                        fe_suggestions.append({"type": "high_corr", "severity": "high",
                            "msg": f"'{corr_mat.columns[i]}' and '{corr_mat.columns[j]}' are {v:.0%} correlated. Consider dropping one."})
                    elif v > 0.75:
                        fe_suggestions.append({"type": "mod_corr", "severity": "medium",
                            "msg": f"'{corr_mat.columns[i]}' and '{corr_mat.columns[j]}' have {v:.0%} correlation."})
        for col in df_num.columns:
            var = safe_float(df_num[col].var())
            if var < 0.01:
                fe_suggestions.append({"type": "low_var", "severity": "high",
                    "msg": f"'{col}' has near-zero variance ({var:.4f}). Consider dropping it."})
        # Check skewness
        for col in df_num.columns:
            sk = safe_float(df_num[col].skew())
            if abs(sk) > 2.0:
                fe_suggestions.append({"type": "skew", "severity": "low",
                    "msg": f"'{col}' is highly skewed (skew={sk:.2f}). Try log or Box-Cox transform."})
        if not fe_suggestions:
            fe_suggestions.append({"type": "ok", "severity": "none",
                "msg": "No obvious feature engineering issues found. Features look clean."})

        # ── Radar data ──
        radar_data = {name: {
            "acc": m["acc"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"]
        } for name, m in results.items()}

        # ── Hyperparameter tuning ──
        default_acc   = results[best_model_name]["acc"]
        tuning_report = None

        n_tune = len(X_train_)
        if n_tune > MAX_ROWS_TUNING:
            idx = np.random.choice(n_tune, MAX_ROWS_TUNING, replace=False)
            Xt = X_train_[idx]
            yt = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
        else:
            Xt, yt = X_train_, y_train

        if best_model_name == "Logistic Regression":
            param_dist = {"C": [0.01, 0.1, 1, 10, 100]}
            search = RandomizedSearchCV(
                LogisticRegression(max_iter=300, solver='saga'),
                param_dist, n_iter=5, cv=3, n_jobs=-1, random_state=42
            )
            search.fit(Xt, yt)
            tuned_acc = safe_float(search.best_estimator_.score(X_test_, y_test))
            tuning_report = {
                "model": best_model_name, "default_acc": default_acc,
                "tuned_acc": tuned_acc, "best_params": search.best_params_,
                "improvement": round((tuned_acc - default_acc) * 100, 2),
                "cv_results": [{"param": str(p["C"]), "score": safe_float(s)}
                    for p, s in zip(search.cv_results_["params"],
                                    search.cv_results_["mean_test_score"])]
            }
        elif best_model_name == "Random Forest":
            param_dist = {"n_estimators": [30, 50, 100], "max_depth": [None, 5, 10]}
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_dist, n_iter=6, cv=3, n_jobs=-1, random_state=42
            )
            search.fit(Xt, yt)
            tuned_acc = safe_float(search.best_estimator_.score(X_test_, y_test))
            tuning_report = {
                "model": best_model_name, "default_acc": default_acc,
                "tuned_acc": tuned_acc,
                "best_params": {str(k): str(v) for k, v in search.best_params_.items()},
                "improvement": round((tuned_acc - default_acc) * 100, 2),
                "cv_results": [{"param": str(p), "score": safe_float(s)}
                    for p, s in zip(search.cv_results_["params"],
                                    search.cv_results_["mean_test_score"])]
            }

        # ── Dataset quality score ──
        if rows < 50:     dataset_quality, quality_score = "LOW", 30
        elif rows < 300:  dataset_quality, quality_score = "MODERATE", 65
        elif rows < 1000: dataset_quality, quality_score = "GOOD", 80
        else:             dataset_quality, quality_score = "HIGH", 100

        confidence  = results[best_model_name]["acc"] * 100
        final_score = round((confidence * 0.6) + (quality_score * 0.4), 2)
        final_status = ("STRONG MODEL" if final_score > 85
                        else "MODERATE MODEL" if final_score > 70
                        else "WEAK MODEL")

        total_time = round(time.time() - t_start, 2)

        result = {
            "final_score": final_score, "final_status": final_status,
            "best_model": best_model_name, "confidence": round(confidence, 2),
            "dataset_quality": dataset_quality, "dataset_quality_score": quality_score,
            "imbalance_ratio": round(imbalance_ratio, 3),
            "imbalance_status": (
                "Highly imbalanced" if imbalance_ratio < 0.3
                else "Moderately balanced" if imbalance_ratio < 0.7
                else "Well balanced"
            ),
            "dataset_info": {"rows": rows, "columns": cols, "missing": missing_before,
                             "filename": file.filename},
            "models": results,
            "feature_importance": [{"name": n, "value": v} for n, v in feat_imp],
            "confusion_matrix": cm, "class_labels": class_labels,
            "per_class_metrics": per_class,
            "roc_fpr": roc_fpr, "roc_tpr": roc_tpr, "roc_auc": roc_auc_val,
            "is_binary": is_binary, "optimal_threshold": optimal_threshold,
            "cv_score": cv_mean, "cv_std": cv_std, "cv_each": cv_each,
            "train_acc": train_acc, "test_acc": test_acc,
            "learning_curve":     lc,
            "fe_suggestions":     fe_suggestions,
            "class_distribution": class_dist,
            "dataset_profile":    dataset_profile,
            "corr_labels":        corr_labels,
            "corr_matrix":        [[safe_float(v) for v in row] for row in corr_matrix],
            "radar_data":         radar_data,
            "tuning_report":      tuning_report,
            "feature_names":      feature_names,
            "total_time":         total_time,
        }

        failure_reasons, tips = detect_failure_reasons(result)
        result["failure_reasons"] = failure_reasons
        result["tips"] = tips

        _model_state = {
            "model":         best_model_obj,
            "scaler":        scaler,
            "feature_names": feature_names,
            "encoders":      encoders,
            "class_labels":  class_labels,
        }

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    global _model_state
    if not _model_state:
        return jsonify({"error": "No model trained yet."}), 400
    try:
        data   = request.json
        vals   = data.get("values", {})
        fnames = _model_state["feature_names"]
        row    = [float(vals.get(f, 0)) for f in fnames]
        X_in   = _model_state["scaler"].transform([row])
        pred   = _model_state["model"].predict(X_in)[0]
        label  = (_model_state["class_labels"][int(pred)]
                  if int(pred) < len(_model_state["class_labels"]) else str(pred))
        conf   = 0.0
        if hasattr(_model_state["model"], "predict_proba"):
            proba = _model_state["model"].predict_proba(X_in)[0]
            conf  = safe_float(max(proba) * 100)
            proba_dict = {_model_state["class_labels"][i]: safe_float(proba[i] * 100)
                          for i in range(min(len(proba), len(_model_state["class_labels"])))}
        else:
            proba_dict = {label: 100.0}
        return jsonify({"prediction": label, "confidence": conf, "probabilities": proba_dict})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    try:
        d   = request.json
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
            leftMargin=15*mm, rightMargin=15*mm,
            topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()

        GOLD = colors.HexColor('#f7c948')
        DARK = colors.HexColor('#0d1117')
        TEXT = colors.HexColor('#e6edf3')
        MUTE = colors.HexColor('#8b949e')

        Y    = ParagraphStyle('Y',    parent=styles['Normal'],  textColor=GOLD,             fontSize=11, fontName='Helvetica-Bold')
        H1   = ParagraphStyle('H1',   parent=styles['Title'],   textColor=GOLD,             fontSize=20, spaceAfter=4)
        H2   = ParagraphStyle('H2',   parent=styles['Heading2'],textColor=TEXT,             fontSize=13, spaceBefore=14, spaceAfter=6)
        BODY = ParagraphStyle('BODY', parent=styles['Normal'],  textColor=MUTE,             fontSize=9,  leading=14)

        tbl_style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), GOLD),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.black),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 9),
            ('BACKGROUND', (0,1), (-1,-1), DARK),
            ('TEXTCOLOR',  (0,1), (-1,-1), TEXT),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [DARK, colors.HexColor('#161b22')]),
            ('GRID',       (0,0), (-1,-1), 0.3, colors.HexColor('#21262d')),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ])

        story = []
        story.append(Paragraph("Why Did My Model Fail? — Analysis Report", H1))
        story.append(Paragraph(
            f"Best Model: {d.get('best_model','?')}  |  "
            f"Final Score: {d.get('final_score','?')}%  |  "
            f"Status: {d.get('final_status','?')}  |  "
            f"Analysis Time: {d.get('total_time','?')}s", BODY))
        story.append(HRFlowable(width="100%", thickness=1, color=GOLD, spaceAfter=12))

        story.append(Paragraph("Dataset Info", H2))
        di = d.get("dataset_info", {})
        story.append(Paragraph(
            f"File: {di.get('filename','?')}  |  "
            f"Rows: {di.get('rows','?')}  |  Columns: {di.get('columns','?')}  |  "
            f"Missing Values: {di.get('missing','?')}", BODY))
        story.append(Paragraph(
            f"Quality: {d.get('dataset_quality','?')}  |  "
            f"Class Balance: {d.get('imbalance_status','?')}", BODY))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Performance Summary", H2))
        met = [["Metric", "Value"],
               ["Test Accuracy",   f"{d.get('test_acc',0)*100:.1f}%"],
               ["Train Accuracy",  f"{d.get('train_acc',0)*100:.1f}%"],
               ["CV Score",        f"{d.get('cv_score',0)*100:.1f}% ± {d.get('cv_std',0)*100:.1f}%"],
               ["ROC AUC",         str(d.get('roc_auc','N/A'))],
               ["Optimal Threshold", str(d.get('optimal_threshold','0.5'))]]
        t = Table(met, hAlign='LEFT', colWidths=[80*mm, 60*mm])
        t.setStyle(tbl_style)
        story.append(t)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Model Leaderboard", H2))
        rows_data = [["Model", "Accuracy", "Precision", "Recall", "F1", "Train Time"]]
        for name, m in sorted(d.get("models", {}).items(), key=lambda x: -x[1]['acc']):
            rows_data.append([name,
                f"{m['acc']*100:.1f}%", f"{m['precision']*100:.1f}%",
                f"{m['recall']*100:.1f}%", f"{m['f1']*100:.1f}%",
                f"{m.get('train_time',0):.3f}s"])
        t2 = Table(rows_data, hAlign='LEFT')
        t2.setStyle(tbl_style)
        story.append(t2)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Failure Diagnosis", H2))
        for r in d.get("failure_reasons", []):
            story.append(Paragraph(f"[{r['severity'].upper()}] {r['title']}", Y))
            story.append(Paragraph(r['detail'], BODY))
            story.append(Spacer(1, 4))

        story.append(Paragraph("Recommendations", H2))
        for tip in d.get("tips", []):
            story.append(Paragraph(f"→ {tip}", BODY))
            story.append(Spacer(1, 3))

        story.append(Paragraph("Top Feature Importance", H2))
        fi_rows = [["Feature", "Importance"]] + [
            [f['name'], f"{f['value']:.4f}"]
            for f in d.get("feature_importance", [])[:10]
        ]
        t3 = Table(fi_rows, hAlign='LEFT', colWidths=[100*mm, 50*mm])
        t3.setStyle(tbl_style)
        story.append(t3)

        if d.get("per_class_metrics"):
            story.append(Paragraph("Per-Class Metrics", H2))
            pc_rows = [["Class", "Precision", "Recall", "F1", "Support"]] + [
                [m["label"], f"{m['precision']*100:.1f}%", f"{m['recall']*100:.1f}%",
                 f"{m['f1']*100:.1f}%", str(m['support'])]
                for m in d["per_class_metrics"]
            ]
            t4 = Table(pc_rows, hAlign='LEFT')
            t4.setStyle(tbl_style)
            story.append(t4)

        doc.build(story)
        return jsonify({"pdf": base64.b64encode(buf.getvalue()).decode()})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)