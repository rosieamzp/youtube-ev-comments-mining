#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
論文第四章圖表生成程式
用途：根據 topic_naming_map.csv 更新所有圖表為中文標籤

使用說明：
1. 先執行 06～09 產出 output 內所需 CSV
2. 執行：python 10_update_figures.py
3. 圖表輸出至 output/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# ============================================================================
# 設定區
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'PingFang TC', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

TOPIC_MAP_FILE = os.path.join(OUTPUT_DIR, 'topic_naming_map.csv')
MONTHLY_STATS_FILE = os.path.join(OUTPUT_DIR, 'monthly_topic_stats.csv')
TREND_ANALYSIS_FILE = os.path.join(OUTPUT_DIR, 'topic_trend_analysis.csv')

# ============================================================================
# 資料讀取
# ============================================================================

print("正在讀取資料檔案...")

try:
    topic_map = pd.read_csv(TOPIC_MAP_FILE, encoding='utf-8-sig')
    monthly_stats = pd.read_csv(MONTHLY_STATS_FILE, encoding='utf-8-sig')
    trend_analysis = pd.read_csv(TREND_ANALYSIS_FILE, encoding='utf-8-sig')
    print("[OK] 資料檔案讀取成功")
except FileNotFoundError as e:
    print(f"[ERROR] 錯誤：找不到檔案 {e.filename}")
    print("請先執行 06_descriptive_stats.py、09_topic_naming_guide.py 產出所需 CSV")
    exit(1)

# ============================================================================
# 資料處理
# ============================================================================

topic_summary = monthly_stats.groupby('topic').agg({
    'volume': 'sum',
    'negative_count': 'sum'
}).reset_index()
topic_summary['negative_rate'] = topic_summary['negative_count'] / topic_summary['volume']
result = topic_map.merge(topic_summary, left_on='topic_id', right_on='topic')

print(f"\n主題總數: {len(result)}")
print(f"評論總數: {result['volume'].sum()}")

# ============================================================================
# 圖4-1：主題分布圓餅圖
# ============================================================================

print("\n正在生成圖4-1：主題分布圖...")
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Set3(range(len(result)))
wedges, texts, autotexts = ax.pie(result['volume'], autopct='%1.1f%%', colors=colors, startangle=90, pctdistance=0.75)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_weight('bold')
legend_labels = [f"Topic {row['topic_id']}: {row['topic_name']}" for _, row in result.iterrows()]
ax.legend(wedges, legend_labels, title="主題", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
ax.set_title('圖4-1 主題分布圖', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, '圖4-1_主題分布圖.png')
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"[OK] 圖4-1 已儲存至: {output_path}")

# ============================================================================
# 圖4-2：各主題情緒分布圖
# ============================================================================

print("正在生成圖4-2：各主題情緒分布圖...")
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(result))
positive = result['volume'] - result['negative_count']
negative = result['negative_count']
ax.bar(x_pos, positive, label='正面', color='#4CAF50', alpha=0.8)
ax.bar(x_pos, negative, bottom=positive, label='負面', color='#F44336', alpha=0.8)
for i, (idx, row) in enumerate(result.iterrows()):
    neg_rate = row['negative_rate'] * 100
    total_height = row['volume']
    ax.text(i, total_height + 50, f"{neg_rate:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f"Topic {row['topic_id']}\n{row['topic_name']}" for _, row in result.iterrows()], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('評論數量', fontsize=11)
ax.set_title('圖4-2 各主題情緒分布圖', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, result['volume'].max() * 1.15)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, '圖4-2_各主題情緒分布圖.png')
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"[OK] 圖4-2 已儲存至: {output_path}")

# ============================================================================
# 圖4-3：顯著趨勢主題聲量變化圖
# ============================================================================

print("正在生成圖4-3：主題聲量趨勢圖...")
significant_topics = trend_analysis[trend_analysis['聲量顯著性'] == '顯著']['主題'].tolist()
if len(significant_topics) == 0:
    significant_topics = result['topic_id'].tolist()
fig, ax = plt.subplots(figsize=(14, 7))
for topic_id in significant_topics:
    topic_data = monthly_stats[monthly_stats['topic'] == topic_id].copy().sort_values('month')
    topic_name = result[result['topic_id'] == topic_id]['topic_name'].values[0]
    ax.plot(range(len(topic_data)), topic_data['volume'], marker='o', linewidth=2, markersize=5, label=f'Topic {topic_id}: {topic_name}')
months = monthly_stats[monthly_stats['topic'] == 0].sort_values('month')['month_str'].values
ax.set_xticks(range(0, len(months), 3))
ax.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, ha='right')
ax.set_xlabel('時間', fontsize=11)
ax.set_ylabel('月度評論數量', fontsize=11)
ax.set_title('圖4-3 顯著趨勢主題聲量變化圖', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, frameon=True)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, '圖4-3_主題聲量趨勢圖.png')
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"[OK] 圖4-3 已儲存至: {output_path}")

# ============================================================================
# 圖4-4：迴歸係數圖
# ============================================================================

print("正在生成圖4-4：迴歸係數圖...")
fig, ax = plt.subplots(figsize=(12, 6))
topics = ['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6']
coefficients = [-0.230, -0.351, -0.072, -0.125, 0.546, 0.154]
p_values = [0.000, 0.000, 0.079, 0.045, 0.000, 0.062]
colors = ['#F44336' if coef < 0 and p < 0.05 else '#4CAF50' if coef > 0 and p < 0.05 else '#9E9E9E' for coef, p in zip(coefficients, p_values)]
ax.barh(topics, coefficients, color=colors, alpha=0.8, height=0.6)
for i, (coef, p) in enumerate(zip(coefficients, p_values)):
    significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ' n.s.'
    x_pos = 0.03 if coef < 0 else coef + 0.03
    ax.text(x_pos, i, f'{coef:.3f}{significance}', ha='left', va='center', fontsize=10, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlim(-0.5, 0.7)
ax.set_xlabel('係數（相對於Topic 0）', fontsize=11)
ax.set_title('圖4-4 主題對按讚數的影響係數圖', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor='#4CAF50', alpha=0.8, label='顯著正向'),
    Patch(facecolor='#F44336', alpha=0.8, label='顯著負向'),
    Patch(facecolor='#9E9E9E', alpha=0.8, label='不顯著')
], loc='lower right', fontsize=9)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, '圖4-4_迴歸係數圖.png')
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"[OK] 圖4-4 已儲存至: {output_path}")

print("\n" + "="*60)
print("所有圖表生成完成")
print("="*60)
print(f"圖表已儲存至: {os.path.abspath(OUTPUT_DIR)}")
