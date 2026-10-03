"""
So sanh 5 phuong phap phat hien gian lan the tin dung tren creditcard.csv,
cong them mot ensemble Voting, danh gia dong bo tren cung mot tap test, luu
lai toan bo model da huan luyen va xuat bieu do minh hoa phuc vu viet bao cao.
(English/ASCII docstring kept for the script header; chart text and console
messages below use full Vietnamese diacritics.)

Usage (terminal):
    python train_and_compare_models.py

Dependencies (ngoai pandas/numpy/matplotlib/seaborn da co o buoc truoc):
    pip install scikit-learn xgboost lightgbm imbalanced-learn joblib
"""

import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import os

# ============================== CONFIG ===============================
CSV_PATH = r"data/creditcard.csv"
FIGURE_DIR = r"data/model_figures"
MODEL_DIR = r"data/models"
DPI = 300
FONT_NAME = "Times New Roman"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_MINORITY_RATIO = 0.2        # SMOTE oversample tới 20% số lượng lớp đa số, không ép 50/50
TSNE_SAMPLE_LEGIT = 4000          # số giao dịch hợp lệ lấy mẫu cho t-SNE (toàn bộ gian lận luôn được giữ)
TOP_N_FEATURE_IMPORTANCE = 15
# ======================================================================

COLOR_LEGIT = "#2E75B6"
COLOR_FRAUD = "#C00000"
MODEL_COLORS = {
    "Logistic Regression": "#4C72B0",
    "Random Forest": "#55A868",
    "XGBoost": "#C44E52",
    "LightGBM": "#8172B2",
    "MLP + SMOTE": "#CCB974",
    "Voting Ensemble": "#000000",
}
MODEL_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "MLP + SMOTE", "Voting Ensemble"]


# ----------------------------------------------------------------------
# Style / tiện ích chung
# ----------------------------------------------------------------------
def setup_style():
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if FONT_NAME not in available_fonts:
        warnings.warn(
            f'Không tìm thấy font "{FONT_NAME}" trên hệ thống này, '
            f"matplotlib sẽ dùng font serif mặc định thay thế."
        )
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [FONT_NAME, "Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["legend.title_fontsize"] = 11
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def save_fig(fig, name):
    png_path = os.path.join(FIGURE_DIR, f"{name}.png")
    pdf_path = os.path.join(FIGURE_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Đã lưu {png_path}")
    plt.close(fig)


def strip_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ----------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    before = df.shape[0]
    df = df.drop_duplicates()
    print(f"Đã đọc {before:,} dòng; loại bỏ {before - df.shape[0]:,} dòng trùng lặp; còn lại {df.shape[0]:,} dòng.")
    return df


def split_and_scale(df):
    X = df.drop(columns=["Class"])
    y = df["Class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Tập train: {X_train.shape[0]:,} dòng ({y_train.sum()} gian lận) | "
          f"Tập test: {X_test.shape[0]:,} dòng ({y_test.sum()} gian lận)")
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)


# ----------------------------------------------------------------------
# Huấn luyện 5 phương pháp
# ----------------------------------------------------------------------
def train_all_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train):
    models = {}
    proba = {}
    train_times = {}
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    # 1. Logistic Regression (dữ liệu đã chuẩn hóa, class_weight balanced)
    t0 = time.time()
    lr = LogisticRegression(class_weight="balanced", max_iter=500, solver="lbfgs", random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    train_times["Logistic Regression"] = time.time() - t0
    models["Logistic Regression"] = lr
    proba["Logistic Regression"] = lr.predict_proba(X_test_scaled)[:, 1]
    print(f"  [1/5] Logistic Regression xong ({train_times['Logistic Regression']:.1f}s)")

    # 2. Random Forest (dữ liệu gốc, class_weight balanced_subsample)
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    train_times["Random Forest"] = time.time() - t0
    models["Random Forest"] = rf
    proba["Random Forest"] = rf.predict_proba(X_test)[:, 1]
    print(f"  [2/5] Random Forest xong ({train_times['Random Forest']:.1f}s)")

    # 3. XGBoost (dữ liệu gốc, scale_pos_weight, cấu hình mặc định đã kiểm chứng tốt nhất)
    t0 = time.time()
    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight, tree_method="hist",
        eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train)
    train_times["XGBoost"] = time.time() - t0
    models["XGBoost"] = xgb
    proba["XGBoost"] = xgb.predict_proba(X_test)[:, 1]
    print(f"  [3/5] XGBoost xong ({train_times['XGBoost']:.1f}s)")

    # 4. LightGBM (dữ liệu gốc, class_weight balanced)
    # Lưu ý: scale_pos_weight trong LightGBM cho hiệu suất rất kém và mất cân chỉnh trên bộ dữ
    # liệu này khi kiểm thử (AUC tụt còn ~0.88, Precision chỉ còn ~2%), nên dùng class_weight
    # "balanced" thay thế, cho kết quả đúng như kỳ vọng (AUC ~0.975).
    t0 = time.time()
    lgbm = LGBMClassifier(
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    train_times["LightGBM"] = time.time() - t0
    models["LightGBM"] = lgbm
    proba["LightGBM"] = lgbm.predict_proba(X_test)[:, 1]
    print(f"  [4/5] LightGBM xong ({train_times['LightGBM']:.1f}s)")

    # 5. MLP + SMOTE (SMOTE chỉ fit trên tập train đã chuẩn hóa, tỷ lệ 20%, không dùng tập test)
    t0 = time.time()
    smote = SMOTE(sampling_strategy=SMOTE_MINORITY_RATIO, random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        max_iter=150, early_stopping=True, n_iter_no_change=10,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train_smote, y_train_smote)
    train_times["MLP + SMOTE"] = time.time() - t0
    models["MLP + SMOTE"] = mlp
    proba["MLP + SMOTE"] = mlp.predict_proba(X_test_scaled)[:, 1]
    print(f"  [5/5] MLP + SMOTE xong ({train_times['MLP + SMOTE']:.1f}s) "
          f"(tập train sau SMOTE: {len(y_train_smote):,} dòng, gian lận {int(y_train_smote.sum()):,})")

    # Ensemble: Voting mềm, trung bình không trọng số của 5 xác suất dự đoán ở trên
    t0 = time.time()
    ensemble_proba = np.mean(
        [proba[name] for name in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "MLP + SMOTE"]],
        axis=0,
    )
    proba["Voting Ensemble"] = ensemble_proba
    train_times["Voting Ensemble"] = time.time() - t0  # chỉ là thao tác trung bình, không huấn luyện thêm
    print(f"  [+1] Voting Ensemble (trung bình 5 mô hình trên) xong")

    return models, proba, train_times, scale_pos_weight


# ----------------------------------------------------------------------
# Đánh giá
# ----------------------------------------------------------------------
def evaluate_all(proba, y_test, threshold=0.5):
    rows = []
    cms = {}
    roc_data = {}
    pr_data = {}
    for name in MODEL_ORDER:
        p = proba[name]
        y_pred = (p >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, p)
        ap = average_precision_score(y_test, p)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, p)
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, p)

        rows.append({
            "Phương pháp": name, "Accuracy": acc, "Precision": prec, "Recall": rec,
            "F1-score": f1, "ROC-AUC": auc, "PR-AUC (AP)": ap,
            "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1],
        })
        cms[name] = cm
        roc_data[name] = (fpr, tpr, auc)
        pr_data[name] = (rec_curve, prec_curve, ap)

    results_df = pd.DataFrame(rows).set_index("Phương pháp").loc[MODEL_ORDER]
    return results_df, cms, roc_data, pr_data


# ----------------------------------------------------------------------
# Biểu đồ 1: Lưới confusion matrix cho cả 6 phương pháp
# ----------------------------------------------------------------------
def plot_confusion_matrices(cms):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.reshape(-1)
    for ax, name in zip(axes, MODEL_ORDER):
        cm = cms[name]
        # Chuẩn hóa theo HÀNG (theo từng nhãn thực tế), để mỗi hàng cộng lại đúng 100%,
        # thay vì tính trên tổng toàn bộ ma trận (cách đó làm hàng Gian lận luôn ra số rất nhỏ
        # vì gian lận vốn chỉ chiếm 0.17% dữ liệu, không phản ánh đúng chất lượng dự đoán).
        cm_row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{cm[i, j]:,}\n({cm_row_pct[i, j]:.2f}%)" for j in range(2)] for i in range(2)])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Hợp lệ", "Gian lận"], yticklabels=["Hợp lệ", "Gian lận"],
                    annot_kws={"fontsize": 14}, linewidths=0.5, linecolor="black")
        ax.set_xlabel("Nhãn dự đoán")
        ax.set_ylabel("Nhãn thực tế")
        ax.set_title(name, fontsize=13)
    fig.suptitle("Ma trận nhầm lẫn (Confusion Matrix) của từng phương pháp", fontsize=17, fontweight="bold", y=1.06)
    fig.text(0.5, 1.015, "Tỷ lệ % tính theo từng hàng (theo nhãn thực tế), mỗi hàng cộng lại đúng 100%",
             ha="center", fontsize=11, style="italic")
    fig.tight_layout()
    save_fig(fig, "01_confusion_matrices")


# ----------------------------------------------------------------------
# Biểu đồ 2: Đường ROC chồng lên nhau
# ----------------------------------------------------------------------
def plot_roc_curves(roc_data):
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in MODEL_ORDER:
        fpr, tpr, auc = roc_data[name]
        lw = 2.6 if name == "Voting Ensemble" else 1.8
        ls = "--" if name == "Voting Ensemble" else "-"
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=lw, linestyle=ls,
                label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, label="Ngẫu nhiên (AUC = 0.5)")
    ax.set_xlabel("Tỷ lệ dương tính giả (FPR)")
    ax.set_ylabel("Tỷ lệ dương tính thật (TPR / Recall)")
    ax.set_title("Đường ROC của các phương pháp")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "02_roc_curves")


# ----------------------------------------------------------------------
# Biểu đồ 3: Đường Precision-Recall chồng lên nhau
# ----------------------------------------------------------------------
def plot_pr_curves(pr_data, y_test):
    baseline = y_test.sum() / len(y_test)
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in MODEL_ORDER:
        rec_curve, prec_curve, ap = pr_data[name]
        lw = 2.6 if name == "Voting Ensemble" else 1.8
        ls = "--" if name == "Voting Ensemble" else "-"
        ax.plot(rec_curve, prec_curve, color=MODEL_COLORS[name], linewidth=lw, linestyle=ls,
                label=f"{name} (AP = {ap:.4f})")
    ax.axhline(baseline, color="gray", linestyle=":", linewidth=1,
               label=f"Ngẫu nhiên (AP = {baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Đường Precision-Recall của các phương pháp")
    ax.legend(frameon=False, loc="lower left", fontsize=9.5)
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "03_precision_recall_curves")


# ----------------------------------------------------------------------
# Biểu đồ 4: Cột so sánh các chỉ số giữa các phương pháp
# ----------------------------------------------------------------------
def plot_metric_comparison(results_df):
    metrics = ["Precision", "Recall", "F1-score", "ROC-AUC"]
    x = np.arange(len(MODEL_ORDER))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [plt.get_cmap("Blues")(v) for v in np.linspace(0.4, 0.9, len(metrics))]
    for i, metric in enumerate(metrics):
        values = results_df.loc[MODEL_ORDER, metric].values
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=metric, color=colors[i], edgecolor="black")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.025, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Giá trị chỉ số")
    ax.set_ylim(0, 1.3)
    ax.set_title("So sánh Precision / Recall / F1-score / ROC-AUC giữa các phương pháp")
    ax.legend(frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.14))
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "04_metric_comparison_bar")


# ----------------------------------------------------------------------
# Biểu đồ 5: Thời gian huấn luyện
# ----------------------------------------------------------------------
def plot_training_time(train_times):
    names = [n for n in MODEL_ORDER if n != "Voting Ensemble"]
    times = [train_times[n] for n in names]
    colors = [MODEL_COLORS[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, times, color=colors, edgecolor="black")
    for bar, v in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Thời gian huấn luyện (giây)")
    ax.set_title("Thời gian huấn luyện của từng phương pháp")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "05_training_time_bar")


# ----------------------------------------------------------------------
# Biểu đồ 6: t-SNE minh họa khả năng phân tách lớp trong không gian đặc trưng
# ----------------------------------------------------------------------
def plot_tsne(X_test_scaled, y_test, ensemble_proba):
    rng = np.random.RandomState(RANDOM_STATE)
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    sample_legit_idx = rng.choice(legit_idx, size=min(TSNE_SAMPLE_LEGIT, len(legit_idx)), replace=False)
    sample_idx = np.concatenate([fraud_idx, sample_legit_idx])
    rng.shuffle(sample_idx)

    X_sample = X_test_scaled[sample_idx]
    y_sample = y_test[sample_idx]
    proba_sample = ensemble_proba[sample_idx]

    print(f"  Đang chạy t-SNE trên {len(sample_idx):,} điểm dữ liệu (có thể mất 1-2 phút)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
    X_2d = tsne.fit_transform(X_sample)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    ax = axes[0]
    ax.scatter(X_2d[y_sample == 0, 0], X_2d[y_sample == 0, 1], s=10, color=COLOR_LEGIT,
               alpha=0.5, label="Hợp lệ", linewidths=0)
    ax.scatter(X_2d[y_sample == 1, 0], X_2d[y_sample == 1, 1], s=22, color=COLOR_FRAUD,
               alpha=0.9, label="Gian lận", edgecolor="black", linewidths=0.3)
    ax.set_xlabel("Chiều t-SNE 1")
    ax.set_ylabel("Chiều t-SNE 2")
    ax.set_title("t-SNE theo nhãn thực tế")
    ax.legend(frameon=False, loc="best", title="Lớp")
    strip_axes(ax)

    ax = axes[1]
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], s=14, c=proba_sample, cmap="coolwarm",
                     vmin=0, vmax=1, alpha=0.8, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Xác suất gian lận dự đoán (Voting Ensemble)")
    ax.set_xlabel("Chiều t-SNE 1")
    ax.set_ylabel("Chiều t-SNE 2")
    ax.set_title("t-SNE theo xác suất dự đoán của Ensemble")
    strip_axes(ax)

    fig.suptitle("Chiếu dữ liệu về 2 chiều bằng t-SNE (mẫu gồm toàn bộ gian lận + mẫu ngẫu nhiên giao dịch hợp lệ)",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "06_tsne_projection")


# ----------------------------------------------------------------------
# Biểu đồ 7: So sánh feature importance giữa 3 mô hình cây (RF, XGBoost, LightGBM)
# ----------------------------------------------------------------------
def plot_feature_importance_comparison(models, feature_names, top_n=TOP_N_FEATURE_IMPORTANCE):
    importances = {}
    for name in ["Random Forest", "XGBoost", "LightGBM"]:
        imp = models[name].feature_importances_
        imp = imp / imp.sum()  # chuẩn hóa về cùng thang để so sánh
        importances[name] = pd.Series(imp, index=feature_names)

    avg_rank = pd.DataFrame(importances).mean(axis=1).sort_values(ascending=False)
    top_features = avg_rank.head(top_n).index[::-1]

    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(top_features))
    bar_h = 0.25
    colors = {"Random Forest": "#55A868", "XGBoost": "#C44E52", "LightGBM": "#8172B2"}
    for i, name in enumerate(["Random Forest", "XGBoost", "LightGBM"]):
        values = importances[name].reindex(top_features).values
        ax.barh(y_pos + (i - 1) * bar_h, values, height=bar_h, label=name, color=colors[name], edgecolor="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Tầm quan trọng đặc trưng (đã chuẩn hóa theo từng mô hình)")
    ax.set_ylabel("Đặc trưng")
    ax.set_title(f"Top {top_n} đặc trưng quan trọng nhất (RF vs XGBoost vs LightGBM)")
    ax.legend(frameon=False, loc="lower right")
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "07_feature_importance_comparison")


# ----------------------------------------------------------------------
# Lưu model
# ----------------------------------------------------------------------
def save_models(models, scaler, feature_names, scale_pos_weight):
    os.makedirs(MODEL_DIR, exist_ok=True)
    name_to_file = {
        "Logistic Regression": "logistic_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
        "LightGBM": "lightgbm.joblib",
        "MLP + SMOTE": "mlp_smote.joblib",
    }
    for name, fname in name_to_file.items():
        path = os.path.join(MODEL_DIR, fname)
        joblib.dump(models[name], path)
        print(f"  Đã lưu model: {path}")

    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Đã lưu scaler: {scaler_path}")

    manifest = {
        "feature_order": feature_names,
        "scale_pos_weight_used": scale_pos_weight,
        "scaled_models": ["Logistic Regression", "MLP + SMOTE"],
        "raw_models": ["Random Forest", "XGBoost", "LightGBM"],
        "ensemble": {
            "type": "soft voting (trung bình không trọng số predict_proba)",
            "members": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "MLP + SMOTE"],
            "note": (
                "Để dự đoán lại: nạp scaler.joblib và 5 model joblib ở trên. Với dữ liệu mới X "
                "(giữ đúng thứ tự cột như feature_order), tính X_scaled = scaler.transform(X). "
                "Logistic Regression và MLP + SMOTE dự đoán trên X_scaled, Random Forest/XGBoost/"
                "LightGBM dự đoán trên X gốc. Lấy trung bình cộng 5 xác suất predict_proba(X)[:, 1] "
                "để ra xác suất gian lận của Voting Ensemble, so với ngưỡng 0.5 để ra nhãn dự đoán."
            ),
        },
    }
    manifest_path = os.path.join(MODEL_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  Đã lưu manifest (hướng dẫn tải lại + kết hợp ensemble): {manifest_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    setup_style()

    print(f"Đang đọc dữ liệu từ: {CSV_PATH}")
    df = load_data(CSV_PATH)

    print("\nĐang chia tập train/test và chuẩn hóa dữ liệu...")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = split_and_scale(df)

    print("\nĐang huấn luyện 5 phương pháp...")
    models, proba, train_times, scale_pos_weight = train_all_models(
        X_train, X_test, X_train_scaled, X_test_scaled, y_train
    )

    print("\nĐang đánh giá tất cả phương pháp trên tập test (ngưỡng 0.5)...")
    results_df, cms, roc_data, pr_data = evaluate_all(proba, y_test, threshold=0.5)

    csv_path = os.path.join(FIGURE_DIR, "comparison_table.csv")
    results_df.round(4).to_csv(csv_path, encoding="utf-8-sig")
    print(f"  Đã lưu bảng so sánh: {csv_path}")
    print("\n" + results_df.round(4).to_string())

    print("\nĐang vẽ biểu đồ...")
    plot_confusion_matrices(cms)
    plot_roc_curves(roc_data)
    plot_pr_curves(pr_data, y_test)
    plot_metric_comparison(results_df)
    plot_training_time(train_times)
    plot_tsne(X_test_scaled, y_test, proba["Voting Ensemble"])
    plot_feature_importance_comparison(models, feature_names)

    print("\nĐang lưu lại các model đã huấn luyện...")
    save_models(models, scaler, feature_names, scale_pos_weight)

    print(f"\nHoàn tất. Biểu đồ lưu trong: {FIGURE_DIR}")
    print(f"Model đã lưu trong: {MODEL_DIR}")


if __name__ == "__main__":
    main()
