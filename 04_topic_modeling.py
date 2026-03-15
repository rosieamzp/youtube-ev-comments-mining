"""
04_topic_modeling.py - 主題建模分析

研究題目：台灣消費者購買電動汽車考量因素之研究
目標：從 YouTube 評論中識別購買考量因素

方法：BERTopic 主題建模
理論：顧客價值理論（分類框架）

輸入：data/comments_cleaned.csv
輸出：
  - output/comments_with_topics.csv（評論 + 主題標記）
  - output/topic_info.csv（主題關鍵字）
  - output/topic_distribution.png（主題分布圖）

執行：python 04_topic_modeling.py

作者：陳莘惠
更新：2025/12
"""

import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 檢查 GPU 可用性
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[系統] 使用裝置: {DEVICE}")
if DEVICE == "cuda":
    print(f"[系統] GPU: {torch.cuda.get_device_name(0)}")

# 設定 Windows 終端編碼
if os.name == 'nt':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 設定中文字型
matplotlib.rc('font', family=['Microsoft JhengHei', 'Arial Unicode MS'])
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 設定參數
# ============================================================

# 檔案路徑（依腳本所在目錄）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 輸入檔案
COMMENTS_FILE = os.path.join(DATA_DIR, "comments_cleaned.csv")
EV_DICT_FILE = os.path.join(BASE_DIR, "ev_dict.txt")
STOPWORDS_FILE = os.path.join(BASE_DIR, "stopwords.txt")

# 輸出檔案
COMMENTS_WITH_TOPICS_FILE = os.path.join(OUTPUT_DIR, "comments_with_topics.csv")
TOPIC_INFO_FILE = os.path.join(OUTPUT_DIR, "topic_info.csv")
TOPIC_DIST_PLOT = os.path.join(OUTPUT_DIR, "topic_distribution.png")
MODEL_DIR = os.path.join(BASE_DIR, "models", "bertopic_model")

# BERTopic 參數
MIN_TOPIC_SIZE = 150        # 每個主題至少 150 則評論
NR_TOPICS = 8               # 目標主題數量

# ============================================================
# 1. 載入電動車詞典
# ============================================================

def load_ev_dict(dict_file):
    """載入電動車專業詞典"""
    print(f"\n[載入詞典] {dict_file}")
    
    if not os.path.exists(dict_file):
        print(f"  [警告] 找不到詞典檔案")
        return
    
    # 載入到 jieba
    jieba.load_userdict(dict_file)
    
    # 統計詞數
    with open(dict_file, 'r', encoding='utf-8') as f:
        word_count = sum(1 for line in f if line.strip() and not line.startswith('#'))
    
    print(f"  [OK] 已載入 {word_count} 個專業詞彙")

# ============================================================
# 2. 載入停用詞
# ============================================================

def load_stopwords(stopwords_file):
    """載入停用詞"""
    print(f"\n[載入停用詞] {stopwords_file}")
    
    if not os.path.exists(stopwords_file):
        print(f"  [警告] 找不到停用詞檔案")
        return set()
    
    stopwords = set()
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith('#'):
                stopwords.add(word)
    
    print(f"  [OK] 已載入 {len(stopwords)} 個停用詞")
    return stopwords

# ============================================================
# 3. 斷詞處理
# ============================================================

def segment_text(text, stopwords):
    """
    斷詞處理
    
    策略：
    1. 使用 jieba 斷詞（已載入電動車詞典）
    2. 保留名詞、動詞、形容詞
    3. 過濾停用詞
    4. 過濾單字
    """
    if pd.isna(text) or len(str(text).strip()) == 0:
        return ""
    
    # jieba 詞性標註
    words = pseg.cut(str(text))
    
    # 保留的詞性
    keep_flags = {'n', 'v', 'a', 'vn', 'an'}  # 名詞、動詞、形容詞
    
    # 過濾
    result = []
    for word, flag in words:
        # 詞性過濾
        if flag[0] not in keep_flags:
            continue
        
        # 長度過濾
        if len(word) < 2:
            continue
        
        # 停用詞過濾
        if word in stopwords:
            continue
        
        result.append(word)
    
    return ' '.join(result)

# ============================================================
# 4. BERTopic 主題建模
# ============================================================

def run_bertopic(documents, min_topic_size=100, nr_topics=12):
    """
    執行 BERTopic 主題建模（GPU 加速版）

    參數：
        documents: 文本列表
        min_topic_size: 最小主題大小
        nr_topics: 主題數量

    返回：
        topic_model: BERTopic 模型
        topics: 主題標籤列表
    """
    print(f"\n[BERTopic 主題建模]")
    print(f"  文本數量: {len(documents):,}")
    print(f"  最小主題大小: {min_topic_size}")
    print(f"  目標主題數: {nr_topics}")
    print(f"  使用裝置: {DEVICE}")

    # 載入 embedding 模型並指定裝置（GPU/CPU）
    print(f"\n  載入 embedding 模型...")
    embedding_model = SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2',
        device=DEVICE
    )

    # 設定 CountVectorizer（中文斷詞）
    vectorizer = CountVectorizer(
        min_df=3,           # 詞頻至少 3 次
        max_df=0.95,        # 最多出現在 95% 文本
        ngram_range=(1, 2)  # 1-2 字詞
    )

    # 初始化 BERTopic（使用 GPU embedding）
    topic_model = BERTopic(
        embedding_model=embedding_model,  # 使用 GPU 加速的 embedding
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        vectorizer_model=vectorizer,
        calculate_probabilities=False,  # 加速計算
        verbose=True
    )

    # 訓練模型
    print(f"\n  開始訓練...")
    topics, probabilities = topic_model.fit_transform(documents)

    # 輸出主題資訊
    topic_info = topic_model.get_topic_info()
    print(f"\n  [OK] 訓練完成")
    print(f"  產生主題數: {len(topic_info) - 1}")  # -1 是噪音主題

    return topic_model, topics

# ============================================================
# 5. 主題資訊整理
# ============================================================

def get_topic_info_table(topic_model):
    """整理主題資訊表格"""
    topic_info = topic_model.get_topic_info()
    
    # 移除噪音主題（-1）
    topic_info = topic_info[topic_info['Topic'] != -1]
    
    # 提取關鍵字（前 10 個）
    topic_info['Keywords'] = topic_info['Topic'].apply(
        lambda x: ', '.join([word for word, _ in topic_model.get_topic(x)[:10]])
    )
    
    # 重新命名欄位
    topic_info = topic_info.rename(columns={
        'Topic': '主題編號',
        'Count': '評論數',
        'Name': 'BERTopic_Name',
        'Keywords': '關鍵字'
    })
    
    # 選擇欄位
    topic_info = topic_info[['主題編號', '評論數', '關鍵字']]
    
    return topic_info

# ============================================================
# 6. 視覺化
# ============================================================

def plot_topic_distribution(df_topics, output_file):
    """繪製主題分布圖"""
    print(f"\n[繪製主題分布圖]")
    
    # 統計各主題評論數（排除噪音主題 -1）
    topic_counts = df_topics[df_topics['topic'] != -1]['topic'].value_counts().sort_index()
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(
        range(len(topic_counts)),
        topic_counts.values,
        color='#1565C0',  # 深藍色
        alpha=0.9
    )
    
    # 標註數值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height):,}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    ax.set_xlabel('主題編號', fontsize=12)
    ax.set_ylabel('評論數', fontsize=12)
    ax.set_title('主題分布圖', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(topic_counts)))
    ax.set_xticklabels([f'Topic {i}' for i in topic_counts.index])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  [OK] 已儲存: {output_file}")
    plt.close()

# ============================================================
# 主程式
# ============================================================

def main():
    print("=" * 70)
    print("04_topic_modeling.py - 主題建模分析")
    print("=" * 70)
    
    # 確保輸出資料夾存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    
    # 1. 讀取資料
    print(f"\n[Step 1] 讀取資料")
    print(f"  檔案: {COMMENTS_FILE}")
    
    if not os.path.exists(COMMENTS_FILE):
        print(f"\n[錯誤] 找不到 {COMMENTS_FILE}")
        print("請先執行 03_clean_comments.py")
        return
    
    df = pd.read_csv(COMMENTS_FILE, encoding='utf-8-sig')
    print(f"  [OK] 讀取 {len(df):,} 則評論")
    
    # 2. 載入詞典與停用詞
    print(f"\n[Step 2] 載入詞典與停用詞")
    load_ev_dict(EV_DICT_FILE)
    stopwords = load_stopwords(STOPWORDS_FILE)
    
    # 3. 斷詞處理
    print(f"\n[Step 3] 斷詞處理")
    print(f"  處理中...")
    
    df['text_segmented'] = df['text_cleaned'].apply(
        lambda x: segment_text(x, stopwords)
    )
    
    # 過濾空白文本
    df = df[df['text_segmented'].str.len() > 0]
    print(f"  [OK] 完成斷詞，保留 {len(df):,} 則評論")
    
    # 顯示樣本
    print(f"\n  [斷詞樣本]")
    for idx in range(min(3, len(df))):
        print(f"  原文: {df.iloc[idx]['text_cleaned'][:50]}...")
        print(f"  斷詞: {df.iloc[idx]['text_segmented'][:50]}...")
        print()
    
    # 4. BERTopic 主題建模
    print(f"\n[Step 4] BERTopic 主題建模")
    
    documents = df['text_segmented'].tolist()
    topic_model, topics = run_bertopic(
        documents,
        min_topic_size=MIN_TOPIC_SIZE,
        nr_topics=NR_TOPICS
    )
    
    # 將主題加入 DataFrame
    df['topic'] = topics
    
    # 5. 整理主題資訊
    print(f"\n[Step 5] 整理主題資訊")
    
    topic_info = get_topic_info_table(topic_model)
    print(f"\n{topic_info.to_string(index=False)}")
    
    # 儲存主題資訊
    topic_info.to_csv(TOPIC_INFO_FILE, index=False, encoding='utf-8-sig')
    print(f"\n  [OK] 已儲存主題資訊: {TOPIC_INFO_FILE}")
    
    # 6. 儲存結果
    print(f"\n[Step 6] 儲存結果")
    
    # 選擇輸出欄位
    output_columns = [
        'comment_id',
        'video_id',
        'text_cleaned',
        'text_segmented',
        'topic',
        'char_count',
        'like_count',
        'reply_count',
        'published_at',
        'video_view_count',
        'video_like_count',
        'channel_id',
        'channel_title'
    ]
    
    # 只保留存在的欄位
    output_columns = [col for col in output_columns if col in df.columns]
    
    df_output = df[output_columns].copy()
    df_output.to_csv(COMMENTS_WITH_TOPICS_FILE, index=False, encoding='utf-8-sig')
    print(f"  [OK] 已儲存評論資料: {COMMENTS_WITH_TOPICS_FILE}")
    
    # 7. 儲存模型
    print(f"\n[Step 7] 儲存模型")
    topic_model.save(MODEL_DIR)
    print(f"  [OK] 已儲存模型: {MODEL_DIR}")
    
    # 8. 視覺化
    print(f"\n[Step 8] 視覺化")
    plot_topic_distribution(df_output, TOPIC_DIST_PLOT)
    
    # 9. 摘要統計
    print("\n" + "=" * 70)
    print("主題建模完成")
    print("=" * 70)
    
    # 主題統計
    topic_stats = df_output['topic'].value_counts().sort_index()
    print(f"\n[主題統計]")
    print(f"  產生主題數: {len(topic_stats[topic_stats.index != -1])}")
    print(f"  噪音評論數: {topic_stats.get(-1, 0):,} 則")
    print(f"  有效評論數: {len(df_output[df_output['topic'] != -1]):,} 則")
    
    print(f"\n[各主題評論數]")
    for topic_id in sorted(topic_stats.index):
        if topic_id == -1:
            continue
        count = topic_stats[topic_id]
        pct = count / len(df_output) * 100
        print(f"  Topic {topic_id:2d}: {count:5,} 則 ({pct:5.1f}%)")
    
    # 提示下一步
    print(f"\n[下一步]")
    print(f"  1. 人工檢視 {TOPIC_INFO_FILE}")
    print(f"  2. 為每個主題命名（例如：保值性與折舊）")
    print(f"  3. 將主題對應到顧客價值理論構面")
    print(f"  4. 執行 05_sentiment_analysis.py")
    
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
