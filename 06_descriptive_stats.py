"""
06_descriptive_stats.py - 描述性統計與趨勢分析

研究題目:台灣消費者購買電動汽車考量因素之研究
目標：分析購買考量因素的重要性、情緒趨勢與時間變化

分析內容：
1. 主題重要性排名（聲量）
2. 各主題情緒分析
3. 時間趨勢分析（月度）
4. 顯著趨勢識別（線性迴歸）

輸入：output/comments_with_sentiment.csv
輸出：
  - output/monthly_topic_stats.csv（月度統計）
  - output/topic_trend_analysis.csv（趨勢分析結果）
  - 多張視覺化圖表

執行：python 06_descriptive_stats.py

作者：陳莘惠
更新：2025/12
"""

import pandas as pd
import numpy as np
from scipy import stats
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
sns.set_style('whitegrid')

# ============================================================
# 設定參數
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

COMMENTS_FILE = os.path.join(OUTPUT_DIR, "comments_with_sentiment.csv")
MONTHLY_STATS_FILE = os.path.join(OUTPUT_DIR, "monthly_topic_stats.csv")
TREND_ANALYSIS_FILE = os.path.join(OUTPUT_DIR, "topic_trend_analysis.csv")
TOPIC_KEYWORDS_PLOT = os.path.join(OUTPUT_DIR, "fig_topic_keywords.png")
SENTIMENT_DIST_PLOT = os.path.join(OUTPUT_DIR, "fig_sentiment_distribution.png")
VOLUME_TREND_PLOT = os.path.join(OUTPUT_DIR, "fig_volume_trend.png")
SENTIMENT_TREND_PLOT = os.path.join(OUTPUT_DIR, "fig_sentiment_trend.png")

# ============================================================
# 1. 計算月度統計
# ============================================================

def calculate_monthly_stats(df):
    """計算各主題的月度統計"""
    print(f"\n[計算月度統計]")
    
    df['month'] = pd.to_datetime(df['published_at']).dt.to_period('M')
    df_valid = df[df['topic'] != -1].copy()
    
    monthly_stats = df_valid.groupby(['month', 'topic']).agg({
        'comment_id': 'count',
        'sentiment': ['mean', lambda x: (x == 0).sum()],
        'sentiment_score': 'mean'
    }).reset_index()
    
    monthly_stats.columns = [
        'month', 'topic', 'volume',
        'sentiment_avg', 'negative_count', 'sentiment_score_avg'
    ]
    monthly_stats['negative_rate'] = (
        monthly_stats['negative_count'] / monthly_stats['volume']
    )
    monthly_stats['month_str'] = monthly_stats['month'].astype(str)
    
    print(f"  ✓ 完成月度統計")
    print(f"  時間範圍: {monthly_stats['month_str'].min()} 至 {monthly_stats['month_str'].max()}")
    print(f"  月份數: {monthly_stats['month'].nunique()}")
    
    return monthly_stats

# ============================================================
# 2. 趨勢分析（線性迴歸）
# ============================================================

def analyze_trends(monthly_stats):
    """分析各主題的趨勢（聲量、負面率）"""
    print(f"\n[趨勢分析]")
    
    results = []
    
    for topic_id in sorted(monthly_stats['topic'].unique()):
        df_topic = monthly_stats[monthly_stats['topic'] == topic_id].copy()
        if len(df_topic) < 3:
            continue
        
        df_topic = df_topic.sort_values('month')
        df_topic['month_num'] = range(len(df_topic))
        
        X = df_topic['month_num'].values
        y_volume = df_topic['volume'].values
        slope_vol, intercept_vol, r_vol, p_vol, se_vol = stats.linregress(X, y_volume)
        
        y_neg = df_topic['negative_rate'].values
        slope_neg, intercept_neg, r_neg, p_neg, se_neg = stats.linregress(X, y_neg)
        
        trend_vol = '上升' if slope_vol > 0 else '下降'
        trend_neg = '上升' if slope_neg > 0 else '下降'
        sig_vol = '顯著' if p_vol < 0.05 else '不顯著'
        sig_neg = '顯著' if p_neg < 0.05 else '不顯著'
        
        results.append({
            '主題': topic_id,
            '聲量斜率': slope_vol,
            '聲量R²': r_vol**2,
            '聲量p值': p_vol,
            '聲量趨勢': trend_vol,
            '聲量顯著性': sig_vol,
            '負面率斜率': slope_neg,
            '負面率R²': r_neg**2,
            '負面率p值': p_neg,
            '負面率趨勢': trend_neg,
            '負面率顯著性': sig_neg
        })
    
    df_trends = pd.DataFrame(results)
    
    print(f"\n  [顯著趨勢發現]")
    sig_vol_trends = df_trends[df_trends['聲量顯著性'] == '顯著']
    if len(sig_vol_trends) > 0:
        print(f"\n  聲量顯著變化的主題:")
        for _, row in sig_vol_trends.iterrows():
            print(f"    Topic {row['主題']}: {row['聲量趨勢']} (p={row['聲量p值']:.4f})")
    sig_neg_trends = df_trends[df_trends['負面率顯著性'] == '顯著']
    if len(sig_neg_trends) > 0:
        print(f"\n  負面率顯著變化的主題:")
        for _, row in sig_neg_trends.iterrows():
            print(f"    Topic {row['主題']}: {row['負面率趨勢']} (p={row['負面率p值']:.4f})")
    
    return df_trends

# ============================================================
# 3. 視覺化
# ============================================================

def plot_volume_trend(monthly_stats, output_file):
    """繪製聲量趨勢圖"""
    print(f"\n[繪製聲量趨勢圖]")
    top_topics = (
        monthly_stats.groupby('topic')['volume'].sum()
        .nlargest(5).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    for topic_id in top_topics:
        df_topic = monthly_stats[monthly_stats['topic'] == topic_id].sort_values('month')
        ax.plot(df_topic['month_str'], df_topic['volume'], marker='o', label=f'Topic {topic_id}', linewidth=2)
    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('評論數', fontsize=12)
    ax.set_title('主要主題聲量趨勢', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已儲存: {output_file}")
    plt.close()

def plot_sentiment_trend(monthly_stats, output_file):
    """繪製情緒趨勢圖"""
    print(f"\n[繪製情緒趨勢圖]")
    top_topics = (
        monthly_stats.groupby('topic')['volume'].sum()
        .nlargest(5).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    for topic_id in top_topics:
        df_topic = monthly_stats[monthly_stats['topic'] == topic_id].sort_values('month')
        ax.plot(df_topic['month_str'], df_topic['negative_rate'] * 100, marker='o', label=f'Topic {topic_id}', linewidth=2)
    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('負面比例 (%)', fontsize=12)
    ax.set_title('主要主題負面情緒趨勢', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已儲存: {output_file}")
    plt.close()

# ============================================================
# 主程式
# ============================================================

def main():
    print("=" * 70)
    print("06_descriptive_stats.py - 描述性統計與趨勢分析")
    print("=" * 70)
    
    print(f"\n[Step 1] 讀取資料")
    if not os.path.exists(COMMENTS_FILE):
        print(f"\n[錯誤] 找不到 {COMMENTS_FILE}")
        print("請先執行 05_sentiment_analysis.py")
        return
    
    df = pd.read_csv(COMMENTS_FILE, encoding='utf-8-sig')
    print(f"  ✓ 讀取 {len(df):,} 則評論")
    
    print(f"\n[Step 2] 基本統計")
    df_valid = df[df['topic'] != -1]
    print(f"\n  [主題分布]")
    topic_dist = df_valid['topic'].value_counts().sort_index()
    for topic_id, count in topic_dist.items():
        pct = count / len(df_valid) * 100
        negative_rate = (df_valid[df_valid['topic'] == topic_id]['sentiment'] == 0).mean() * 100
        print(f"    Topic {topic_id:2d}: {count:5,} 則 ({pct:5.1f}%) | 負面率: {negative_rate:.1f}%")
    
    print(f"\n[Step 3] 月度統計")
    monthly_stats = calculate_monthly_stats(df)
    monthly_stats.to_csv(MONTHLY_STATS_FILE, index=False, encoding='utf-8-sig')
    print(f"  ✓ 已儲存: {MONTHLY_STATS_FILE}")
    
    print(f"\n[Step 4] 趨勢分析")
    df_trends = analyze_trends(monthly_stats)
    df_trends.to_csv(TREND_ANALYSIS_FILE, index=False, encoding='utf-8-sig')
    print(f"  ✓ 已儲存: {TREND_ANALYSIS_FILE}")
    
    print(f"\n[Step 5] 視覺化")
    plot_volume_trend(monthly_stats, VOLUME_TREND_PLOT)
    plot_sentiment_trend(monthly_stats, SENTIMENT_TREND_PLOT)
    
    print("\n" + "=" * 70)
    print("描述性統計完成")
    print("=" * 70)
    print(f"\n[研究期間] {monthly_stats['month_str'].min()} 至 {monthly_stats['month_str'].max()}")
    print(f"\n[下一步] 執行 07_regression_analysis.py")
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
