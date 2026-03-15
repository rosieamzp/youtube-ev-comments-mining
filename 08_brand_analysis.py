"""
08_brand_analysis.py - 品牌差異分析

研究題目：台灣消費者購買電動汽車考量因素之研究
目標：分析不同品牌/價位帶消費者的購買考量重點差異

分析內容：
1. 不同品牌的主題分布差異（卡方檢定）
2. 豪華品牌 vs 平價品牌的情緒差異
3. 各品牌最關注的購買考量因素

品牌分類：
- 平價品牌：LUXGEN
- 中高價品牌：Tesla, BMW
- 豪華品牌：Porsche, Mercedes-Benz, Volvo

輸入：output/comments_with_sentiment.csv
輸出：
  - output/brand_topic_distribution.csv
  - output/fig_brand_topic_heatmap.png
  - output/fig_brand_sentiment.png

執行：python 08_brand_analysis.py

作者：陳莘惠
更新：2025/12
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')
matplotlib.rc('font', family=['Microsoft JhengHei', 'Arial Unicode MS'])
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 設定參數
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

COMMENTS_FILE = os.path.join(OUTPUT_DIR, "comments_with_sentiment.csv")
BRAND_TOPIC_DIST_FILE = os.path.join(OUTPUT_DIR, "brand_topic_distribution.csv")
BRAND_HEATMAP_FILE = os.path.join(OUTPUT_DIR, "fig_brand_topic_heatmap.png")
BRAND_SENTIMENT_FILE = os.path.join(OUTPUT_DIR, "fig_brand_sentiment.png")

BRAND_KEYWORDS = {
    'Tesla': ['tesla', 'model y', 'model 3', 'model x', 'model s'],
    'LUXGEN': ['luxgen', 'n7', '納智捷'],
    'BMW': ['bmw', 'ix1', 'ix2', 'ix', 'i4'],
    'Mercedes': ['mercedes', 'benz', '賓士', 'eqe', 'eqa'],
    'Porsche': ['porsche', '保時捷', 'macan'],
    'Volvo': ['volvo', 'ex30', 'xc40', 'c40']
}

def identify_brand(text, brand_keywords):
    """根據評論內容識別品牌"""
    if pd.isna(text):
        return '其他'
    text_lower = str(text).lower()
    for brand, keywords in brand_keywords.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return brand
    return '其他'

def analyze_brand_topic_distribution(df):
    """分析各品牌的主題分布"""
    print(f"\n[品牌主題分布分析]")
    df_valid = df[(df['topic'] != -1) & (df['brand'] != '其他')].copy()
    crosstab = pd.crosstab(df_valid['brand'], df_valid['topic'], normalize='index') * 100
    contingency_table = pd.crosstab(df_valid['brand'], df_valid['topic'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"  卡方檢定: χ² = {chi2:.2f}, p值 = {p_value:.4f}")
    if p_value < 0.05:
        print(f"  結論: 不同品牌的主題分布有顯著差異")
    return crosstab, p_value

def plot_brand_topic_heatmap(crosstab, output_file):
    """繪製品牌-主題熱度圖"""
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(crosstab, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': '百分比 (%)'}, linewidths=0.5, ax=ax)
    ax.set_xlabel('主題編號', fontsize=12)
    ax.set_ylabel('品牌', fontsize=12)
    ax.set_title('各品牌主題分布熱度圖', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已儲存: {output_file}")
    plt.close()

def plot_brand_sentiment(df, output_file):
    """繪製品牌情緒比較圖"""
    df_valid = df[(df['topic'] != -1) & (df['brand'] != '其他')].copy()
    brand_sentiment = df_valid.groupby('brand').agg({
        'sentiment': ['count', lambda x: (x == 0).sum()]
    }).reset_index()
    brand_sentiment.columns = ['brand', 'total', 'negative']
    brand_sentiment['negative_rate'] = brand_sentiment['negative'] / brand_sentiment['total'] * 100
    brand_sentiment = brand_sentiment.sort_values('negative_rate', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#e74c3c' if rate > 60 else '#3498db' for rate in brand_sentiment['negative_rate']]
    bars = ax.barh(brand_sentiment['brand'], brand_sentiment['negative_rate'], color=colors, alpha=0.8)
    for bar, rate in zip(bars, brand_sentiment['negative_rate']):
        ax.text(rate + 1, bar.get_y() + bar.get_height() / 2, f'{rate:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('負面評論比例 (%)', fontsize=12)
    ax.set_ylabel('品牌', fontsize=12)
    ax.set_title('各品牌負面情緒比較', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已儲存: {output_file}")
    plt.close()

def main():
    print("=" * 70)
    print("08_brand_analysis.py - 品牌差異分析")
    print("=" * 70)
    if not os.path.exists(COMMENTS_FILE):
        print(f"\n[錯誤] 找不到 {COMMENTS_FILE}，請先執行 05_sentiment_analysis.py")
        return
    df = pd.read_csv(COMMENTS_FILE, encoding='utf-8-sig')
    print(f"\n[Step 1] 讀取 {len(df):,} 則評論")
    df['brand'] = df['text_cleaned'].apply(lambda x: identify_brand(x, BRAND_KEYWORDS))
    brand_dist = df['brand'].value_counts()
    print(f"\n[品牌分布]")
    for brand, count in brand_dist.items():
        print(f"    {brand}: {count:,} 則 ({count/len(df)*100:.1f}%)")
    crosstab, p_value = analyze_brand_topic_distribution(df)
    crosstab.to_csv(BRAND_TOPIC_DIST_FILE, encoding='utf-8-sig')
    print(f"  ✓ 已儲存: {BRAND_TOPIC_DIST_FILE}")
    plot_brand_topic_heatmap(crosstab, BRAND_HEATMAP_FILE)
    plot_brand_sentiment(df, BRAND_SENTIMENT_FILE)
    print("\n" + "=" * 70)
    print("品牌差異分析完成")
    print("=" * 70)
    print(f"\n各品牌最關注的購買考量因素:")
    for brand in crosstab.index:
        top_topic = crosstab.loc[brand].idxmax()
        top_pct = crosstab.loc[brand].max()
        print(f"    {brand}: Topic {top_topic} ({top_pct:.1f}%)")
    print(f"\n{'=' * 70}")

if __name__ == "__main__":
    main()
