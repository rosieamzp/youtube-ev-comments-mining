# 臺灣消費者購買電動汽車考量因素之研究

**Research on the Factors Influencing Electric Vehicle Purchase Decisions Among Taiwanese Consumers: A Text Mining Analysis of YouTube User Comments**

碩士論文研究專案：以 YouTube 台灣電動車相關影片評論進行文字探勘，結合 BERTopic 主題建模、RoBERTa 情緒分析與負二項迴歸，探討消費者購買電動車的考量因素。

---

## 專案結構（整理版）

```
EV_analysis_clean/
├── config.example.py      # 複製成 00_config.py 後，在該檔或環境變數填 YouTube API 金鑰
├── 01_search_videos.py    # Step 1：搜尋影片
├── 02_collect_comments.py # Step 2：蒐集評論
├── 03_clean_comments.py   # Step 3：清理評論
├── 04_topic_modeling.py   # Step 4：BERTopic 主題建模
├── 05_sentiment_analysis.py # Step 5：RoBERTa 情緒分析
├── 06_descriptive_stats.py   # 描述性統計與趨勢分析
├── 07_regression_analysis.py # 負二項迴歸分析
├── 08_brand_analysis.py   # 品牌差異分析
├── 09_topic_naming_guide.py  # 主題命名輔助
├── 10_update_figures.py   # 論文圖表生成
├── ev_dict.txt            # 電動車專業詞典（jieba）
├── stopwords.txt          # 停用詞
├── data/                  # 輸入資料
├── output/                # 分析結果輸出
└── .gitignore
```

---

## 快速開始

1. **設定**
   - 把 `config.example.py` 複製成 `00_config.py`，在裡面填 YouTube Data API 金鑰，或先設定環境變數 `YOUTUBE_API_KEY` 再執行

2. **環境**
   - Python 3.x，建議用虛擬環境。本專案沒有附 requirements.txt，請依各腳本開頭的 import 自行安裝（例如 pandas、requests、bertopic、transformers、statsmodels 等）

3. **執行流程（依序）**
   ```bash
   python 01_search_videos.py
   # 手動篩選 data/video_list.csv 後
   python 02_collect_comments.py
   python 03_clean_comments.py
   python 04_topic_modeling.py   # 建議 GPU
   python 05_sentiment_analysis.py # 建議 GPU
   python 06_descriptive_stats.py
   python 07_regression_analysis.py
   python 08_brand_analysis.py   # 可選
   python 09_topic_naming_guide.py
   python 10_update_figures.py
   ```

---

## 研究方法摘要

| 階段     | 方法／工具 |
|----------|------------|
| 資料蒐集 | YouTube Data API、繁體中文過濾 |
| 主題建模 | BERTopic（paraphrase-multilingual-MiniLM-L12-v2） |
| 情緒分析 | RoBERTa（uer/roberta-base-finetuned-dianping-chinese） |
| 統計分析 | 負二項迴歸（依變數：like_count、reply_count） |

---

## 說明

- 路徑都依腳本所在目錄來算，換到別台電腦或 clone 到別的位置也能跑。
- 00_config.py 裡會放 API 金鑰，請只留在本機使用，不要 push 到 GitHub。
