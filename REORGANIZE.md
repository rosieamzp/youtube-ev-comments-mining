# 整理說明（EV_analysis_clean）

本資料夾是從 **EV_analysis** 複製並整理的版本，僅在複本內做以下變更，**原始專案 EV_analysis 未被修改**。

---

## 已完成的調整

1. **路徑與設定**
   - 新增 `config.example.py`：不含 API 金鑰，使用 `os.path.dirname(os.path.abspath(__file__))` 作為專案根目錄。
   - 01～05 與 04 已改為依腳本所在目錄計算路徑，可在任意位置執行。

2. **檔名**
   - `02_collect comments.py` → `02_collect_comments.py`（無空格，方便指令列執行）。

3. **不納入複本的項目**
   - `.venv`、`.git`、`.claude`、`__pycache__` 未複製。
   - 未複製 `_copy_to_clean.py`、`_ul`、`check.py` 等除錯／暫存檔。
   - 未複製內部文件：`CLAUDE-Rosie-Win11.md`、`thesis_complete_progress_report.md`、`sentiment_analysis_summary.md`。

4. **新增**
   - `.gitignore`：排除 `.venv`、`__pycache__`、機敏設定等。
   - `README.md`：專案說明與執行步驟。
   - 本檔案 `REORGANIZE.md`。

---

## 已納入的腳本

本目錄已包含 `06_descriptive_stats.py`～`10_update_figures.py`，路徑皆改為依腳本所在目錄計算，可直接執行。

## 可選：從上層 EV_analysis 複製的內容

若需要進階分析或既有輸出，可從 **EV_analysis** 手動複製到本資料夾：

| 項目 | 說明 |
|------|------|
| `11_overdispersion_validation.py`～`14_sentiment_trend_analysis.py` | 進階分析。複製後將腳本內 `BASE_DIR` 改為 `os.path.dirname(os.path.abspath(__file__))`。 |
| `scripts/` | 例如 `plot_topic_keywords_cleaned.py`，若有硬編碼路徑請改為依 `__file__` 計算。 |
| `data/` | 若需沿用既有資料，可複製 `video_list.csv`、`comments_raw.csv`、`comments_cleaned.csv`（勿將含個資或機敏的檔案上傳至公開 repo）。 |
| `output/`、`results/` | 既有輸出與圖表，可選複製。 |

---

## 使用 00_config.py 的注意事項

- 01、02、03 透過 `00_config.py` 讀取設定（API 金鑰、路徑等）。
- 請將 `config.example.py` 複製為 `00_config.py`，並填入您的 YouTube API 金鑰。
- 勿將含有真實 API 金鑰的 `00_config.py` 提交至版本控制；`.gitignore` 已建議排除機敏設定檔。
