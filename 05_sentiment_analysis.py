"""
05_sentiment_analysis.py - 情緒分析

研究題目：台灣消費者購買電動汽車考量因素之研究
目標：分析消費者對各購買考量因素的情緒態度

方法：RoBERTa 中文情緒分析模型
模型：uer/roberta-base-finetuned-dianping-chinese

輸入：output/comments_with_topics.csv
輸出：
  - output/comments_with_sentiment.csv（評論 + 情緒標記）
  - output/sentiment_by_topic.png（各主題情緒分布圖）

執行：python 05_sentiment_analysis.py

作者：陳莘惠
更新：2025/12
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 設定 stdout 編碼為 UTF-8（解決 Windows 終端編碼問題）
sys.stdout.reconfigure(encoding='utf-8')

# 設定中文字型
matplotlib.rc('font', family=['Microsoft JhengHei', 'Arial Unicode MS'])
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 設定參數
# ============================================================

# 檔案路徑（依腳本所在目錄）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 輸入檔案
COMMENTS_FILE = os.path.join(OUTPUT_DIR, "comments_with_topics.csv")

# 輸出檔案
COMMENTS_WITH_SENTIMENT_FILE = os.path.join(OUTPUT_DIR, "comments_with_sentiment.csv")
SENTIMENT_PLOT_FILE = os.path.join(OUTPUT_DIR, "sentiment_by_topic.png")

# RoBERTa 模型
MODEL_NAME = "uer/roberta-base-finetuned-dianping-chinese"
BATCH_SIZE = 32

# ============================================================
# 1. 載入 RoBERTa 模型
# ============================================================

def load_sentiment_model():
    """載入 RoBERTa 情緒分析模型"""
    print(f"\n[載入 RoBERTa 模型]")
    print(f"  模型: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # 移到 GPU（如果有的話）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"  ✓ 模型載入完成")
    print(f"  設備: {device}")
    
    return tokenizer, model, device

# ============================================================
# 2. 批次情緒分析
# ============================================================

def analyze_sentiment_batch(texts, tokenizer, model, device, batch_size=32):
    """
    批次進行情緒分析
    
    參數：
        texts: 文本列表
        tokenizer: 分詞器
        model: 模型
        device: 設備（CPU/GPU）
        batch_size: 批次大小
    
    返回：
        sentiments: 情緒標籤列表（0=負面, 1=正面）
        scores: 信心分數列表
    """
    sentiments = []
    scores = []
    
    # 批次處理
    for i in tqdm(range(0, len(texts), batch_size), desc="情緒分析"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # 移到設備
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 預測
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # 提取結果
        for pred in predictions:
            # 0=負面, 1=正面
            sentiment = pred.argmax().item()
            score = pred.max().item()
            
            sentiments.append(sentiment)
            scores.append(score)
    
    return sentiments, scores

# ============================================================
# 3. 視覺化
# ============================================================

def plot_sentiment_by_topic(df, output_file):
    """繪製各主題情緒分布圖"""
    print(f"\n[繪製情緒分布圖]")
    
    # 排除噪音主題
    df_valid = df[df['topic'] != -1].copy()
    
    # 計算各主題的負面比例
    topic_sentiment = df_valid.groupby('topic').agg({
        'sentiment': ['count', 'sum']
    }).reset_index()
    
    topic_sentiment.columns = ['topic', 'total', 'positive']
    topic_sentiment['negative'] = topic_sentiment['total'] - topic_sentiment['positive']
    topic_sentiment['negative_rate'] = topic_sentiment['negative'] / topic_sentiment['total'] * 100
    
    # 排序
    topic_sentiment = topic_sentiment.sort_values('topic')
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(topic_sentiment))
    width = 0.35
    
    # 負面
    bars1 = ax.bar(
        x - width/2,
        topic_sentiment['negative'],
        width,
        label='負面',
        color='#e74c3c',
        alpha=0.8
    )
    
    # 正面
    bars2 = ax.bar(
        x + width/2,
        topic_sentiment['positive'],
        width,
        label='正面',
        color='#3498db',
        alpha=0.8
    )
    
    # 標註負面比例
    for i, row in topic_sentiment.iterrows():
        ax.text(
            i,
            row['total'] + 20,
            f"{row['negative_rate']:.1f}%",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    ax.set_xlabel('主題編號', fontsize=12)
    ax.set_ylabel('評論數', fontsize=12)
    ax.set_title('各主題情緒分布圖', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Topic {t}' for t in topic_sentiment['topic']])
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已儲存: {output_file}")
    plt.close()

# ============================================================
# 主程式
# ============================================================

def main():
    print("=" * 70)
    print("05_sentiment_analysis.py - 情緒分析")
    print("=" * 70)
    
    # 1. 讀取資料
    print(f"\n[Step 1] 讀取資料")
    print(f"  檔案: {COMMENTS_FILE}")
    
    if not os.path.exists(COMMENTS_FILE):
        print(f"\n[錯誤] 找不到 {COMMENTS_FILE}")
        print("請先執行 04_topic_modeling.py")
        return
    
    df = pd.read_csv(COMMENTS_FILE, encoding='utf-8-sig')
    print(f"  ✓ 讀取 {len(df):,} 則評論")
    
    # 2. 載入模型
    print(f"\n[Step 2] 載入模型")
    tokenizer, model, device = load_sentiment_model()
    
    # 3. 情緒分析
    print(f"\n[Step 3] 執行情緒分析")
    print(f"  批次大小: {BATCH_SIZE}")
    print(f"  預計時間: 約 {len(df) // (BATCH_SIZE * 60)}-{len(df) // (BATCH_SIZE * 30)} 分鐘")
    
    texts = df['text_cleaned'].tolist()
    sentiments, scores = analyze_sentiment_batch(
        texts,
        tokenizer,
        model,
        device,
        batch_size=BATCH_SIZE
    )
    
    # 加入結果
    df['sentiment'] = sentiments
    df['sentiment_score'] = scores
    
    # 情緒標籤（文字）
    df['sentiment_label'] = df['sentiment'].map({0: '負面', 1: '正面'})
    
    print(f"\n  ✓ 情緒分析完成")
    
    # 4. 統計摘要
    print(f"\n[Step 4] 統計摘要")
    
    print(f"\n  [整體情緒分布]")
    sentiment_dist = df['sentiment_label'].value_counts()
    for label, count in sentiment_dist.items():
        pct = count / len(df) * 100
        print(f"    {label}: {count:,} 則 ({pct:.1f}%)")
    
    print(f"\n  [各主題情緒分布]")
    for topic_id in sorted(df[df['topic'] != -1]['topic'].unique()):
        df_topic = df[df['topic'] == topic_id]
        negative_count = (df_topic['sentiment'] == 0).sum()
        total_count = len(df_topic)
        negative_rate = negative_count / total_count * 100
        
        print(f"    Topic {topic_id:2d}: {negative_rate:5.1f}% 負面 ({negative_count:,}/{total_count:,})")
    
    # 5. 儲存結果
    print(f"\n[Step 5] 儲存結果")
    
    df.to_csv(COMMENTS_WITH_SENTIMENT_FILE, index=False, encoding='utf-8-sig')
    print(f"  ✓ 已儲存: {COMMENTS_WITH_SENTIMENT_FILE}")
    
    # 6. 視覺化
    print(f"\n[Step 6] 視覺化")
    plot_sentiment_by_topic(df, SENTIMENT_PLOT_FILE)
    
    # 7. 完成摘要
    print("\n" + "=" * 70)
    print("情緒分析完成")
    print("=" * 70)
    
    print(f"\n[資料摘要]")
    print(f"  總評論數: {len(df):,}")
    print(f"  正面評論: {(df['sentiment'] == 1).sum():,} 則")
    print(f"  負面評論: {(df['sentiment'] == 0).sum():,} 則")
    print(f"  平均信心分數: {df['sentiment_score'].mean():.3f}")
    
    print(f"\n[下一步]")
    print(f"  執行 06_descriptive_stats.py（描述性統計與趨勢分析）")
    
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
