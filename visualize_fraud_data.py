"""
Script truc quan hoa du lieu cho bo du lieu Credit Card Fraud Detection (creditcard.csv).
(English docstring kept for the script header; chart text below is in Vietnamese.)

Usage (terminal):
    python visualize_fraud_data.py

Edit CONFIG below first (CSV_PATH, OUTPUT_DIR).
CPU only (pandas/matplotlib/seaborn); a GPU is not needed for plotting.

Dependencies:
    pip install pandas numpy matplotlib seaborn
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import seaborn as sns

# ============================== CONFIG ===============================
CSV_PATH = r"C:\Users\hung\Desktop\Hoa Project\creditcard.csv"
OUTPUT_DIR = r"C:\Users\hung\Desktop\Hoa Project\figures"
DPI = 300
FONT_NAME = "Times New Roman"
TOP_FEATURES_FOR_KDE = ("V14", "V4", "V10", "V12", "V17", "V11")
SCATTER_FEATURE_PAIR = ("V14", "V4")  # hai đặc trưng mạnh và độc lập nhất với gian lận
AMOUNT_BINS = [-0.01, 10, 50, 200, 1000, float("inf")]
AMOUNT_LABELS = ["< $10", "$10-50", "$50-200", "$200-1.000", "> $1.000"]
TIME_PERIOD_BINS = [-0.01, 6, 12, 18, 24]
TIME_PERIOD_LABELS = ["Đêm (0-6h)", "Sáng (6-12h)", "Chiều (12-18h)", "Tối (18-24h)"]
# ======================================================================

COLOR_LEGIT = "#2E75B6"
COLOR_FRAUD = "#C00000"


def setup_style():
    """Cấu hình style chung cho matplotlib: font, kích thước, trục, tick, legend."""
    available_fonts = {f.name for f in fm.fontManager.ttflist}
    if FONT_NAME not in available_fonts:
        warnings.warn(
            f'Không tìm thấy font "{FONT_NAME}" trên hệ thống này. '
            f"Matplotlib sẽ dùng font serif mặc định thay thế. "
            f'Trên Windows, Times New Roman có sẵn theo hệ điều hành; trên Linux/Mac '
            f'cần cài thêm (ví dụ gói "msttcorefonts") để khớp chính xác.'
        )

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [FONT_NAME, "Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = FONT_NAME

    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.labelweight"] = "normal"
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["legend.title_fontsize"] = 12

    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0
    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["ytick.major.size"] = 5
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["pdf.fonttype"] = 42  # nhúng chữ thật (không phải đường nét) vào PDF
    plt.rcParams["ps.fonttype"] = 42


def save_fig(fig, name):
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Đã lưu {png_path}")
    print(f"  Đã lưu {pdf_path}")
    plt.close(fig)


def load_data(path):
    df = pd.read_csv(path)
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Đã đọc {before:,} dòng; loại bỏ {before - after:,} dòng trùng lặp; còn lại {after:,} dòng.")
    df["Hour"] = (df["Time"] / 3600) % 24
    df["AmountBucket"] = pd.cut(df["Amount"], bins=AMOUNT_BINS, labels=AMOUNT_LABELS)
    df["TimePeriod"] = pd.cut(df["Hour"], bins=TIME_PERIOD_BINS, labels=TIME_PERIOD_LABELS, include_lowest=True)
    return df


def strip_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def sequential_colors(n, cmap_name="Blues", lo=0.35, hi=0.85):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(x) for x in np.linspace(lo, hi, n)]


# ----------------------------------------------------------------------
# 01. Phân bố lớp (cột, thang log) - số liệu chính xác, góc nhìn đáng tin cậy
# ----------------------------------------------------------------------
def plot_class_distribution(df):
    counts = df["Class"].value_counts().sort_index()
    labels = ["Hợp lệ (0)", "Gian lận (1)"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=[COLOR_LEGIT, COLOR_FRAUD],
                  edgecolor="black", linewidth=1.0, width=0.5)
    ax.set_yscale("log")
    ax.set_ylim(top=counts.values.max() * 8)  # chừa khoảng trống để nhãn không đè tiêu đề
    ax.set_xlabel("Loại giao dịch")
    ax.set_ylabel("Số lượng giao dịch (thang log)")
    ax.set_title("Phân bố lớp: giao dịch hợp lệ so với gian lận", pad=16)

    for bar, v in zip(bars, counts.values):
        pct = v / counts.sum() * 100
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.6,
                f"{v:,}\n({pct:.3f}%)", ha="center", va="bottom", fontsize=11)

    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "01_class_distribution_bar")


# ----------------------------------------------------------------------
# 02. Tỷ lệ lớp (biểu đồ donut)
# ----------------------------------------------------------------------
def plot_class_proportion_pie(df):
    counts = df["Class"].value_counts().sort_index()
    labels = ["Hợp lệ", "Gian lận"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, _, autotexts = ax.pie(
        counts.values,
        colors=[COLOR_LEGIT, COLOR_FRAUD],
        explode=(0.0, 0.12),
        startangle=90,
        autopct=lambda p: f"{p:.2f}%" if p > 1 else f"{p:.3f}%",
        pctdistance=0.8,
        wedgeprops={"edgecolor": "black", "linewidth": 1.0, "width": 0.45},
    )
    for t in autotexts:
        t.set_fontsize(12)
    ax.legend(wedges, labels, title="Lớp", loc="center", frameon=False)
    ax.set_title("Tỷ lệ giao dịch gian lận (biểu đồ Donut)", pad=16)
    fig.tight_layout()
    save_fig(fig, "02_class_proportion_donut")


# ----------------------------------------------------------------------
# 03. Phân bố số tiền giao dịch (thang log), tách theo lớp
# ----------------------------------------------------------------------
def plot_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    titles = ["Giao dịch hợp lệ", "Giao dịch gian lận"]
    colors = [COLOR_LEGIT, COLOR_FRAUD]

    for ax, cls, color, title in zip(axes, [0, 1], colors, titles):
        amounts = df.loc[df["Class"] == cls, "Amount"]
        amounts = amounts[amounts > 0]  # thang log cần giá trị dương
        ax.hist(amounts, bins=50, color=color, edgecolor="black", alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("Số tiền giao dịch, USD (thang log)")
        ax.set_ylabel("Tần suất")
        ax.set_title(title)
        strip_axes(ax)

    fig.suptitle("Phân bố số tiền giao dịch theo lớp", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "03_amount_distribution_hist")


# ----------------------------------------------------------------------
# 04. Cơ cấu khoảng số tiền, Hợp lệ vs Gian lận (hai biểu đồ tròn cạnh nhau)
# ----------------------------------------------------------------------
def plot_amount_category_pie(df):
    colors = sequential_colors(len(AMOUNT_LABELS), cmap_name="Blues")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, cls, title in zip(axes, [0, 1], ["Giao dịch hợp lệ", "Giao dịch gian lận"]):
        counts = df.loc[df["Class"] == cls, "AmountBucket"].value_counts().reindex(AMOUNT_LABELS)
        ax.pie(
            counts.values, labels=counts.index, colors=colors, startangle=90,
            autopct="%.1f%%", pctdistance=0.75,
            wedgeprops={"edgecolor": "black", "linewidth": 0.8},
            textprops={"fontsize": 11},
        )
        ax.set_title(title)

    fig.suptitle("Cơ cấu số tiền giao dịch theo lớp", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "04_amount_category_pie")


# ----------------------------------------------------------------------
# 05. Boxplot số tiền giao dịch theo lớp
# ----------------------------------------------------------------------
def plot_amount_boxplot(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="Class", y="Amount", hue="Class",
                palette=[COLOR_LEGIT, COLOR_FRAUD], legend=False, ax=ax)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Hợp lệ (0)", "Gian lận (1)"])
    ax.set_xlabel("Loại giao dịch")
    ax.set_ylabel("Số tiền giao dịch, USD (thang log)")
    ax.set_title("Số tiền giao dịch theo lớp (Boxplot)")
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "05_amount_boxplot")


# ----------------------------------------------------------------------
# 06. Violin plot số tiền giao dịch theo lớp - bổ sung hình dạng phân bố ngoài boxplot
# ----------------------------------------------------------------------
def plot_amount_violin(df):
    plot_df = df.loc[df["Amount"] > 0, ["Class", "Amount"]].copy()
    plot_df["LogAmount"] = np.log10(plot_df["Amount"])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=plot_df, x="Class", y="LogAmount", hue="Class",
                    palette=[COLOR_LEGIT, COLOR_FRAUD], legend=False, ax=ax, inner="quartile", cut=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Hợp lệ (0)", "Gian lận (1)"])
    ax.set_xlabel("Loại giao dịch")
    ax.set_ylabel(r"Số tiền giao dịch, USD (thang $\log_{10}$)")
    ax.set_title("Hình dạng phân bố số tiền theo lớp (Violin Plot)")
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "06_amount_violin")


# ----------------------------------------------------------------------
# 07. ECDF số tiền giao dịch theo lớp - góc nhìn tích lũy, không phụ thuộc cách chia bin
# ----------------------------------------------------------------------
def plot_amount_ecdf(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, color, label in [(0, COLOR_LEGIT, "Hợp lệ"), (1, COLOR_FRAUD, "Gian lận")]:
        values = np.sort(df.loc[df["Class"] == cls, "Amount"].values)
        values = values[values > 0]
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, color=color, linewidth=2, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Số tiền giao dịch, USD (thang log)")
    ax.set_ylabel("Tỷ lệ tích lũy giao dịch")
    ax.set_title("Hàm phân phối tích lũy thực nghiệm (ECDF) của số tiền theo lớp")
    ax.legend(frameon=False, loc="lower right", title="Lớp")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "07_amount_ecdf")


# ----------------------------------------------------------------------
# 08. Thời điểm giao dịch theo giờ trong ngày (mật độ)
# ----------------------------------------------------------------------
def plot_hourly_pattern(df):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, 25, 1)

    ax.hist(df.loc[df["Class"] == 0, "Hour"], bins=bins, density=True, alpha=0.6,
            label="Hợp lệ", color=COLOR_LEGIT, edgecolor="black")
    ax.hist(df.loc[df["Class"] == 1, "Hour"], bins=bins, density=True, alpha=0.6,
            label="Gian lận", color=COLOR_FRAUD, edgecolor="black")

    ax.set_xlabel("Giờ trong ngày (0-24, suy ra từ Time)")
    ax.set_ylabel("Tần suất tương đối (mật độ)")
    ax.set_title("Thời điểm giao dịch theo giờ trong ngày")
    ax.set_xticks(np.arange(0, 25, 2))
    ax.legend(frameon=False, loc="upper right", title="Lớp")
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "08_hourly_pattern_density")


# ----------------------------------------------------------------------
# 09. Khối lượng giao dịch và tỷ lệ gian lận theo khung giờ (cột + đường, hai trục)
# ----------------------------------------------------------------------
def plot_time_period_bar(df):
    grouped = df.groupby("TimePeriod", observed=True).agg(
        total=("Class", "size"), fraud=("Class", "sum")
    ).reindex(TIME_PERIOD_LABELS)
    grouped["fraud_rate"] = grouped["fraud"] / grouped["total"] * 100

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    bar_colors = sequential_colors(len(TIME_PERIOD_LABELS), cmap_name="Blues")
    bars = ax1.bar(grouped.index, grouped["total"], color=bar_colors, edgecolor="black")
    ax1.set_ylabel("Số lượng giao dịch")
    ax1.set_xlabel("Khung giờ trong ngày")
    for bar, v in zip(bars, grouped["total"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(grouped.index, grouped["fraud_rate"], color=COLOR_FRAUD, marker="o",
             markersize=8, linewidth=2, label="Tỷ lệ gian lận")
    ax2.set_ylabel("Tỷ lệ gian lận (%)", color=COLOR_FRAUD)
    ax2.tick_params(axis="y", labelcolor=COLOR_FRAUD)
    ax2.set_ylim(bottom=0)

    ax1.set_title("Khối lượng giao dịch và tỷ lệ gian lận theo khung giờ trong ngày", pad=16)
    strip_axes(ax1)
    ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "09_time_period_volume_fraud_rate")


# ----------------------------------------------------------------------
# 10. Tỷ lệ gian lận theo khoảng số tiền giao dịch (cột)
# ----------------------------------------------------------------------
def plot_fraud_rate_by_amount_category(df):
    grouped = df.groupby("AmountBucket", observed=True).agg(
        total=("Class", "size"), fraud=("Class", "sum")
    ).reindex(AMOUNT_LABELS)
    grouped["fraud_rate"] = grouped["fraud"] / grouped["total"] * 100

    colors = sequential_colors(len(AMOUNT_LABELS), cmap_name="Reds")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(grouped.index, grouped["fraud_rate"], color=colors, edgecolor="black")
    for bar, v, n in zip(bars, grouped["fraud_rate"], grouped["total"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}%\n(n={n:,})",
                ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Khoảng số tiền giao dịch")
    ax.set_ylabel("Tỷ lệ gian lận (%)")
    ax.set_title("Tỷ lệ gian lận theo khoảng số tiền giao dịch")
    ax.set_ylim(top=grouped["fraud_rate"].max() * 1.35)
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "10_fraud_rate_by_amount_bucket")


# ----------------------------------------------------------------------
# 11. Scatter Giờ vs Số tiền, gian lận nổi bật trên nền hợp lệ mờ nhạt
# ----------------------------------------------------------------------
def plot_time_vs_amount_scatter(df):
    legit = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(legit["Hour"], legit["Amount"] + 1, s=6, color=COLOR_LEGIT, alpha=0.08,
               label="Hợp lệ", rasterized=True, linewidths=0)
    ax.scatter(fraud["Hour"], fraud["Amount"] + 1, s=28, color=COLOR_FRAUD, alpha=0.85,
               label="Gian lận", edgecolor="black", linewidths=0.4)

    ax.set_yscale("log")
    ax.set_xlabel("Giờ trong ngày (0-24, suy ra từ Time)")
    ax.set_ylabel("Số tiền giao dịch + 1, USD (thang log)")
    ax.set_title("Số tiền giao dịch theo giờ trong ngày")
    ax.set_xticks(np.arange(0, 25, 2))
    legend = ax.legend(frameon=False, loc="upper right", title="Lớp", markerscale=2)
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "11_time_vs_amount_scatter")


# ----------------------------------------------------------------------
# 12. Ma trận tương quan đầy đủ
# ----------------------------------------------------------------------
def plot_correlation_heatmap(df):
    corr = df.drop(columns=["Hour"]).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0, linewidths=0.3,
                square=True, cbar_kws={"shrink": 0.8, "label": "Hệ số tương quan"}, ax=ax)
    ax.set_title("Ma trận tương quan của tất cả đặc trưng", fontsize=16, fontweight="bold", pad=15)
    ax.tick_params(axis="x", labelsize=9, rotation=90)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    fig.tight_layout()
    save_fig(fig, "12_correlation_heatmap")


# ----------------------------------------------------------------------
# 13. Các đặc trưng tương quan mạnh nhất với Class (cột ngang)
# ----------------------------------------------------------------------
def plot_top_correlation_with_class(df, top_n=15):
    corr = df.drop(columns=["Hour"]).corr(numeric_only=True)["Class"].drop("Class")
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(top_n)
    colors = [COLOR_FRAUD if v > 0 else COLOR_LEGIT for v in top.values]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hệ số tương quan với Class")
    ax.set_ylabel("Đặc trưng")
    ax.set_title(f"Top {top_n} đặc trưng tương quan mạnh nhất với lớp gian lận")
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, "13_top_correlation_with_class")


# ----------------------------------------------------------------------
# 14. KDE các đặc trưng phân biệt mạnh nhất, tách theo lớp
# ----------------------------------------------------------------------
def plot_feature_kde(df, features=TOP_FEATURES_FOR_KDE):
    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, feat in enumerate(features):
        ax = axes[i]
        sns.kdeplot(df.loc[df["Class"] == 0, feat], ax=ax, color=COLOR_LEGIT,
                    fill=True, alpha=0.4, label="Hợp lệ")
        sns.kdeplot(df.loc[df["Class"] == 1, feat], ax=ax, color=COLOR_FRAUD,
                    fill=True, alpha=0.4, label="Gian lận")
        ax.set_title(f"Phân bố của {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Mật độ")
        strip_axes(ax)

    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Phân bố đặc trưng theo lớp (các đặc trưng phân biệt mạnh nhất)",
                 fontsize=16, fontweight="bold", y=1.07)
    fig.tight_layout()
    save_fig(fig, "14_feature_kde_by_class")


# ----------------------------------------------------------------------
# 15. Scatter hai đặc trưng mạnh nhất - khả năng phân tách lớp trực quan
# ----------------------------------------------------------------------
def plot_feature_scatter(df, feat_pair=SCATTER_FEATURE_PAIR):
    fx, fy = feat_pair
    legit = df[df["Class"] == 0]
    fraud = df[df["Class"] == 1]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(legit[fx], legit[fy], s=8, color=COLOR_LEGIT, alpha=0.10,
               label="Hợp lệ", rasterized=True, linewidths=0)
    ax.scatter(fraud[fx], fraud[fy], s=24, color=COLOR_FRAUD, alpha=0.85,
               label="Gian lận", edgecolor="black", linewidths=0.4)

    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.set_title(f"Khả năng phân tách lớp: {fx} so với {fy}")
    legend = ax.legend(frameon=False, loc="best", title="Lớp", markerscale=2)
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    strip_axes(ax)
    fig.tight_layout()
    save_fig(fig, f"15_feature_scatter_{fx}_{fy}")


# ----------------------------------------------------------------------
# 16. Bảng thống kê mô tả (Amount), thể hiện dưới dạng hình
# ----------------------------------------------------------------------
def plot_summary_stats_table(df):
    stats_overall = df[["Amount"]].describe().round(2)
    stats_legit = df.loc[df["Class"] == 0, ["Amount"]].describe().round(2)
    stats_fraud = df.loc[df["Class"] == 1, ["Amount"]].describe().round(2)

    row_name_map = {
        "count": "Số lượng",
        "mean": "Trung bình",
        "std": "Độ lệch chuẩn",
        "min": "Nhỏ nhất",
        "25%": "Phân vị 25%",
        "50%": "Trung vị (50%)",
        "75%": "Phân vị 75%",
        "max": "Lớn nhất",
    }

    table_data = pd.DataFrame({
        "Toàn bộ": stats_overall["Amount"],
        "Hợp lệ": stats_legit["Amount"],
        "Gian lận": stats_fraud["Amount"],
    })
    table_data.index = [row_name_map.get(i, i) for i in table_data.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    tbl = ax.table(cellText=table_data.values, rowLabels=table_data.index,
                   colLabels=table_data.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.7)
    ax.set_title("Thống kê mô tả số tiền giao dịch (USD)",
                 fontsize=15, fontweight="bold", pad=20)
    fig.tight_layout()
    save_fig(fig, "16_summary_statistics_table")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_style()

    print(f"Đang đọc dữ liệu từ: {CSV_PATH}")
    df = load_data(CSV_PATH)

    print("\nĐang tạo các biểu đồ...")
    plot_class_distribution(df)
    plot_class_proportion_pie(df)
    plot_amount_distribution(df)
    plot_amount_category_pie(df)
    plot_amount_boxplot(df)
    plot_amount_violin(df)
    plot_amount_ecdf(df)
    plot_hourly_pattern(df)
    plot_time_period_bar(df)
    plot_fraud_rate_by_amount_category(df)
    plot_time_vs_amount_scatter(df)
    plot_correlation_heatmap(df)
    plot_top_correlation_with_class(df)
    plot_feature_kde(df)
    plot_feature_scatter(df)
    plot_summary_stats_table(df)

    print(f"\nHoàn tất. Tất cả biểu đồ (PNG + PDF) đã lưu vào: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
