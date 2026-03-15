"""
09_topic_naming_guide.py - 主題命名輔助工具（加強版）

研究題目：台灣消費者購買電動汽車考量因素之研究
目標：基於代表性評論內容與關鍵詞，產出精確的主題命名建議

方法（三重驗證）：
1. BERTopic 關鍵詞分析（模型視角）
2. 代表性評論詞頻分析（實際內容）
3. 人工檢視 50 則代表性評論（質性驗證）

輸入：
  - output/comments_with_topics.csv
  - output/topic_info.csv

輸出：
  - output/topic_naming_guide.xlsx（完整報告，多工作表）
  - output/topic_naming_summary.txt（摘要，口試用）
  - output/topic_naming_map.csv（主題名稱對照表）

執行：python 09_topic_naming_guide.py

作者：陳莘惠
更新：2025/12
"""

import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 設定參數
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 輸入檔案
COMMENTS_FILE = os.path.join(OUTPUT_DIR, "comments_with_topics.csv")
TOPIC_INFO_FILE = os.path.join(OUTPUT_DIR, "topic_info.csv")
EV_DICT_FILE = os.path.join(BASE_DIR, "ev_dict.txt")
STOPWORDS_FILE = os.path.join(BASE_DIR, "stopwords.txt")

# 輸出檔案
NAMING_GUIDE_FILE = os.path.join(OUTPUT_DIR, "topic_naming_guide.xlsx")
NAMING_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "topic_naming_summary.txt")
NAMING_MAP_FILE = os.path.join(OUTPUT_DIR, "topic_naming_map.csv")

# 抽樣參數（方案 A：均衡型 10-30-10）
N_TOP = 10          # 前 10 則（最具代表性）
N_MIDDLE = 30       # 中間 30 則（隨機抽樣）
N_BOTTOM = 10       # 後 10 則（一致性檢查）

# ============================================================
# 1. 載入詞典與停用詞
# ============================================================

def load_resources():
    """載入電動車詞典與停用詞"""
    print(f"\n[載入資源]")
    
    # 載入詞典
    if os.path.exists(EV_DICT_FILE):
        jieba.load_userdict(EV_DICT_FILE)
        print(f"  [OK] 已載入電動車詞典")
    
    # 載入停用詞
    stopwords = set()
    if os.path.exists(STOPWORDS_FILE):
        with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):
                    stopwords.add(word)
        print(f"  [OK] 已載入 {len(stopwords)} 個停用詞")
    
    return stopwords

# ============================================================
# 2. 分層抽樣代表性評論
# ============================================================

def sample_representative_comments(df_topic, n_top=10, n_middle=30, n_bottom=10):
    """
    分層抽樣代表性評論（方案 A：10-30-10）
    
    策略：
    - 前 10 則：BERTopic 認為最具代表性
    - 中間 30 則：隨機抽樣，避免極端偏誤
    - 後 10 則：檢查主題邊界一致性
    """
    n_total = len(df_topic)
    
    if n_total <= (n_top + n_middle + n_bottom):
        # 如果評論數不足 50 則，全部取用
        result = df_topic.copy()
        result['sample_source'] = '全部'
        return result
    
    # 前 N 則
    top_comments = df_topic.head(n_top)
    
    # 中間 M 則（隨機）
    start_idx = n_top
    end_idx = n_total - n_bottom
    middle_pool = df_topic.iloc[start_idx:end_idx]
    
    if len(middle_pool) >= n_middle:
        middle_comments = middle_pool.sample(n_middle, random_state=42)
    else:
        middle_comments = middle_pool
    
    # 後 N 則
    bottom_comments = df_topic.tail(n_bottom)
    
    # 合併
    result = pd.concat([top_comments, middle_comments, bottom_comments])
    
    # 標記來源
    result = result.copy()
    sources = (
        ['前10則'] * len(top_comments) +
        ['中間30則'] * len(middle_comments) +
        ['後10則'] * len(bottom_comments)
    )
    result['sample_source'] = sources
    
    return result

# ============================================================
# 3. 分析代表性評論的詞頻
# ============================================================

def analyze_comment_wordfreq(comments, stopwords, top_n=30):
    """
    分析評論的詞頻
    
    步驟：
    1. jieba 斷詞
    2. 過濾停用詞
    3. 保留名詞、動詞、形容詞
    4. 統計詞頻
    """
    word_counter = Counter()
    
    for text in comments:
        if pd.isna(text):
            continue
        
        # jieba 詞性標註
        words = pseg.cut(str(text))
        
        # 保留的詞性
        keep_flags = {'n', 'v', 'a', 'vn', 'an'}
        
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
            
            word_counter[word] += 1
    
    # 返回前 N 個高頻詞
    return word_counter.most_common(top_n)

# ============================================================
# 4. 整合分析：BERTopic 關鍵詞 + 評論詞頻
# ============================================================

def integrate_keywords(bertopic_keywords, comment_wordfreq):
    """
    整合 BERTopic 關鍵詞與評論詞頻
    
    策略：
    - 優先使用 BERTopic 關鍵詞（模型認為重要）
    - 補充評論高頻詞（實際內容驗證）
    - 去重合併
    """
    # BERTopic 關鍵詞（字串）
    bert_words = [w.strip() for w in bertopic_keywords.split(',')]
    
    # 評論詞頻（tuple list）
    comment_words = [word for word, freq in comment_wordfreq]
    
    # 合併去重（保持順序）
    integrated = []
    seen = set()
    
    # 先加入 BERTopic 關鍵詞
    for word in bert_words:
        if word not in seen and len(word) >= 2:
            integrated.append(word)
            seen.add(word)
    
    # 補充評論高頻詞
    for word in comment_words:
        if word not in seen:
            integrated.append(word)
            seen.add(word)
        if len(integrated) >= 30:
            break
    
    return integrated

# ============================================================
# 5. 智慧命名：基於整合關鍵詞
# ============================================================

def generate_smart_topic_name(integrated_keywords, topic_id, comment_count):
    """
    基於整合關鍵詞，智慧生成主題名稱
    
    策略：
    1. 識別核心詞彙（前 3-5 個關鍵詞）
    2. 判斷主題類型（品牌、功能、成本等）
    3. 生成 3 個候選名稱
    """
    
    # 取前 5 個關鍵詞作為核心
    core_keywords = integrated_keywords[:5]
    
    # 品牌偵測
    brands = ['特斯拉', 'tesla', '納智捷', 'luxgen', 'n7', 'bmw', 'benz', '賓士', 
              'porsche', '保時捷', 'volvo']
    has_brand = any(b in ''.join(core_keywords).lower() for b in brands)
    
    # 功能類偵測
    functions = ['電池', '充電', '續航', '里程', '駕駛', '操控', '內裝', '音響', 
                 '隔音', '座椅', '螢幕', '語音', '安全']
    has_function = any(f in ''.join(core_keywords) for f in functions)
    
    # 成本類偵測
    costs = ['價格', '保險', '保費', '關稅', '稅金', '補助', '保值', '折舊']
    has_cost = any(c in ''.join(core_keywords) for c in costs)
    
    # 根據關鍵詞模式生成名稱
    candidates = generate_candidates_by_pattern(
        core_keywords, has_brand, has_function, has_cost, topic_id
    )
    
    return candidates

def generate_candidates_by_pattern(keywords, has_brand, has_function, has_cost, topic_id):
    """根據關鍵詞模式生成候選名稱"""
    
    kw_str = ''.join(keywords).lower()
    
    # === 特斯拉相關 ===
    if 'tesla' in kw_str or '特斯拉' in kw_str:
        return {
            'candidate_1': '特斯拉品牌生態系統與服務網絡',
            'candidate_2': '特斯拉品牌與生態系統',
            'candidate_3': '特斯拉超充與軟體體驗',
            'theory': '情感價值',
            'reason': '關鍵詞聚焦特斯拉品牌及其獨特生態（超充、軟體更新）'
        }
    
    # === 納智捷相關 ===
    if 'n7' in kw_str or '納智捷' in kw_str or 'luxgen' in kw_str:
        return {
            'candidate_1': '納智捷N7國產品牌認同與評價',
            'candidate_2': '納智捷N7國產品牌',
            'candidate_3': 'LUXGEN N7使用者評價',
            'theory': '情感價值 + 社會價值',
            'reason': '關鍵詞聚焦國產品牌，反映消費者對本土品牌的支持與評價'
        }
    
    # === 電池相關 ===
    if '電池' in kw_str and any(w in kw_str for w in ['保固', '壽命', '衰退', '里程']):
        return {
            'candidate_1': '電池系統壽命與保固政策',
            'candidate_2': '電池壽命與保固',
            'candidate_3': '電池衰退與續航焦慮',
            'theory': '功能價值',
            'reason': '關鍵詞聚焦電池壽命、保固、衰退等核心技術議題'
        }
    
    # === 內裝相關 ===
    if '內裝' in kw_str or ('音響' in kw_str and '隔音' in kw_str):
        return {
            'candidate_1': '車輛內裝品質與車載娛樂系統',
            'candidate_2': '內裝設計與車載娛樂',
            'candidate_3': '內裝質感與音響隔音',
            'theory': '功能價值 + 情感價值',
            'reason': '關鍵詞涵蓋內裝質感、音響、隔音等功能性與感官體驗'
        }
    
    # === 駕駛相關 ===
    if '駕駛' in kw_str or '開車' in kw_str or '操控' in kw_str:
        return {
            'candidate_1': '駕駛性能體驗與燃料經濟性比較',
            'candidate_2': '駕駛體驗與油電比較',
            'candidate_3': '駕駛操控與能源效率',
            'theory': '功能價值 + 犧牲成本',
            'reason': '關鍵詞涵蓋駕駛體驗與油電比較，反映性能與成本考量'
        }
    
    # === 保險/政策相關 ===
    if ('保險' in kw_str or '保費' in kw_str) and ('關稅' in kw_str or '政府' in kw_str):
        return {
            'candidate_1': '保險成本與政府政策關稅',
            'candidate_2': '保險與政策關稅',
            'candidate_3': '保費成本與稅制補助',
            'theory': '犧牲成本',
            'reason': '關鍵詞涵蓋保險、關稅、補助等購買與持有成本'
        }
    
    # === 國籍/產地相關 ===
    if any(c in kw_str for c in ['中國', '韓國', '日本', '美國']):
        return {
            'candidate_1': '品牌國籍認同與產地競爭',
            'candidate_2': '品牌國籍與競爭',
            'candidate_3': '品牌產地與集團歸屬',
            'theory': '社會價值',
            'reason': '關鍵詞涉及品牌國籍、產地，反映消費者的國籍認同與價值觀'
        }
    
    # === 預設（無法分類） ===
    return {
        'candidate_1': f'主題 {topic_id}：{keywords[0]}與{keywords[1]}',
        'candidate_2': f'{keywords[0]}相關議題',
        'candidate_3': f'{keywords[0]}、{keywords[1]}討論',
        'theory': '待分類',
        'reason': '請根據代表性評論內容判斷主題性質'
    }

# ============================================================
# 6. 生成完整命名報告
# ============================================================

def generate_naming_report(df, topic_info, stopwords):
    """生成完整的主題命名報告"""
    
    print(f"\n[生成主題命名報告]")
    
    reports = []
    all_representative_comments = []
    
    # 排除噪音主題
    valid_topics = sorted(df[df['topic'] != -1]['topic'].unique())
    
    for topic_id in valid_topics:
        print(f"\n  處理 Topic {topic_id}...")
        
        # 該主題的所有評論
        df_topic = df[df['topic'] == topic_id].copy()
        
        # === 步驟 1：抽樣代表性評論 ===
        representative = sample_representative_comments(
            df_topic,
            n_top=N_TOP,
            n_middle=N_MIDDLE,
            n_bottom=N_BOTTOM
        )
        
        representative['topic'] = topic_id
        all_representative_comments.append(representative)
        
        print(f"    抽樣 {len(representative)} 則評論（{len(representative)/len(df_topic)*100:.1f}%）")
        
        # === 步驟 2：獲取 BERTopic 關鍵詞 ===
        topic_row = topic_info[topic_info['主題編號'] == topic_id]
        if len(topic_row) > 0:
            bertopic_keywords = topic_row.iloc[0]['關鍵字']
        else:
            bertopic_keywords = ""
        
        # === 步驟 3：分析評論詞頻 ===
        comment_wordfreq = analyze_comment_wordfreq(
            representative['text_cleaned'].tolist(),
            stopwords,
            top_n=30
        )
        
        print(f"    評論詞頻分析：識別 {len(comment_wordfreq)} 個高頻詞")
        
        # === 步驟 4：整合關鍵詞 ===
        integrated_keywords = integrate_keywords(bertopic_keywords, comment_wordfreq)
        
        print(f"    整合關鍵詞：{len(integrated_keywords)} 個")
        
        # === 步驟 5：智慧命名 ===
        naming = generate_smart_topic_name(integrated_keywords, topic_id, len(df_topic))
        
        # === 整理報告 ===
        report = {
            'topic_id': topic_id,
            'count': len(df_topic),
            'percentage': len(df_topic) / len(df[df['topic'] != -1]) * 100,
            'bertopic_keywords': bertopic_keywords,
            'comment_top_words': ', '.join([w for w, f in comment_wordfreq[:10]]),
            'integrated_keywords': ', '.join(integrated_keywords[:15]),
            'candidate_1': naming['candidate_1'],
            'candidate_2': naming['candidate_2'],
            'candidate_3': naming['candidate_3'],
            'theory': naming['theory'],
            'reason': naming['reason'],
            'n_sampled': len(representative)
        }
        
        reports.append(report)
        
        print(f"    [OK] 建議命名：{naming['candidate_2']}")
    
    df_reports = pd.DataFrame(reports)
    df_all_comments = pd.concat(all_representative_comments, ignore_index=True)
    
    print(f"\n  [OK] 完成 {len(valid_topics)} 個主題的命名分析")
    
    return df_reports, df_all_comments

# ============================================================
# 7. 儲存為 Excel（多工作表）
# ============================================================

def save_to_excel(df_reports, df_comments, output_file):
    """儲存為 Excel，包含多個工作表"""
    
    print(f"\n[儲存 Excel 報告]")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # === 工作表 1：主題命名摘要 ===
        df_summary = df_reports[[
            'topic_id', 'count', 'percentage',
            'candidate_1', 'candidate_2', 'candidate_3',
            'theory', 'reason'
        ]].copy()
        
        df_summary.columns = [
            '主題編號', '評論數', '占比(%)',
            '候選1（學術化）', '候選2（平衡）[推薦]', '候選3（具體）',
            '理論構面', '命名理由'
        ]
        
        df_summary.to_excel(writer, sheet_name='1_主題命名摘要', index=False)
        
        # === 工作表 2：關鍵詞分析 ===
        df_keywords = df_reports[[
            'topic_id',
            'bertopic_keywords',
            'comment_top_words',
            'integrated_keywords'
        ]].copy()
        
        df_keywords.columns = [
            '主題編號',
            'BERTopic關鍵詞',
            '評論高頻詞（前10）',
            '整合關鍵詞（前15）'
        ]
        
        df_keywords.to_excel(writer, sheet_name='2_關鍵詞分析', index=False)
        
        # === 工作表 3-9：各主題代表性評論 ===
        for topic_id in sorted(df_comments['topic'].unique()):
            df_topic_comments = df_comments[df_comments['topic'] == topic_id].copy()
            
            # 選擇要顯示的欄位
            display_cols = ['sample_source', 'text_cleaned', 'like_count', 'reply_count']
            df_display = df_topic_comments[display_cols].copy()
            df_display.columns = ['來源', '評論內容', '按讚數', '回覆數']
            
            sheet_name = f'Topic{topic_id}_評論50則'
            df_display.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"  [OK] 已儲存: {output_file}")
    print(f"  包含 {2 + len(df_comments['topic'].unique())} 個工作表")

# ============================================================
# 8. 生成文字摘要（口試準備）
# ============================================================

def generate_text_summary(df_reports, output_file):
    """生成文字版摘要"""
    
    print(f"\n[生成文字摘要]")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("主題命名指南（加強版）\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【命名方法說明】（口試準備）\n\n")
        f.write("本研究採用三重驗證的命名方法：\n\n")
        
        f.write("第一步：BERTopic 關鍵詞分析（模型視角）\n")
        f.write("  - 分析 BERTopic 模型輸出的主題關鍵詞\n")
        f.write("  - 反映模型認為最能區分該主題的詞彙\n\n")
        
        f.write("第二步：代表性評論詞頻分析（實際內容）\n")
        f.write("  - 抽取 50 則代表性評論（分層抽樣：前10 + 中間30 + 後10）\n")
        f.write("  - 分析這 50 則評論的實際詞頻\n")
        f.write("  - 驗證模型關鍵詞是否與實際討論一致\n\n")
        
        f.write("第三步：整合分析與人工判斷\n")
        f.write("  - 整合 BERTopic 關鍵詞與評論詞頻\n")
        f.write("  - 檢視代表性評論內容\n")
        f.write("  - 對應至顧客價值理論構面\n")
        f.write("  - 產出最終主題名稱\n\n")
        
        f.write("此方法確保主題命名同時具有：\n")
        f.write("  1. 模型支持（BERTopic 關鍵詞）\n")
        f.write("  2. 內容驗證（實際評論詞頻）\n")
        f.write("  3. 理論對應（顧客價值理論）\n\n")
        
        f.write("=" * 80 + "\n\n")
        
        for idx, row in df_reports.iterrows():
            f.write(f"Topic {row['topic_id']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"評論數：{row['count']:,} 則（{row['percentage']:.1f}%）\n")
            f.write(f"抽樣數：{row['n_sampled']} 則\n\n")
            
            f.write(f"【關鍵詞分析】\n")
            f.write(f"  BERTopic 關鍵詞：{row['bertopic_keywords']}\n")
            f.write(f"  評論高頻詞：{row['comment_top_words']}\n")
            f.write(f"  整合關鍵詞：{row['integrated_keywords']}\n\n")
            
            f.write(f"【建議命名】\n")
            f.write(f"  候選 1（學術化）：{row['candidate_1']}\n")
            f.write(f"  候選 2（平衡）[推薦]：{row['candidate_2']}\n")
            f.write(f"  候選 3（具體）：{row['candidate_3']}\n\n")
            
            f.write(f"【理論構面】\n")
            f.write(f"  {row['theory']}\n\n")
            
            f.write(f"【命名理由】\n")
            f.write(f"  {row['reason']}\n\n")
            
            f.write("=" * 80 + "\n\n")
        
        f.write("\n【推薦命名】（候選2，平衡風格）\n\n")
        
        for idx, row in df_reports.iterrows():
            f.write(f"Topic {row['topic_id']}: {row['candidate_2']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("\n【下一步】\n")
        f.write("1. 開啟 topic_naming_guide.xlsx\n")
        f.write("2. 檢視「2_關鍵詞分析」工作表，比較三種關鍵詞來源\n")
        f.write("3. 檢視各主題的 50 則代表性評論\n")
        f.write("4. 從 3 個候選名稱中選擇，或根據評論內容自行調整\n")
        f.write("5. 記住命名理由（口試準備）\n")
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"  [OK] 已儲存: {output_file}")

# ============================================================
# 9. 生成主題名稱對照表
# ============================================================

def generate_naming_map(df_reports, output_file):
    """生成主題名稱對照表（用於後續分析）"""
    
    print(f"\n[生成主題名稱對照表]")
    
    # 使用候選 2（平衡風格）作為預設
    df_map = df_reports[['topic_id', 'candidate_2', 'theory']].copy()
    df_map.columns = ['topic_id', 'topic_name', 'theory_dimension']
    
    df_map.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"  [OK] 已儲存: {output_file}")
    print(f"\n  [預設命名]（使用候選2）")
    for _, row in df_map.iterrows():
        print(f"    Topic {row['topic_id']}: {row['topic_name']}")

# ============================================================
# 主程式
# ============================================================

def main():
    print("=" * 80)
    print("09_topic_naming_guide.py - 主題命名輔助工具（加強版）")
    print("=" * 80)
    
    # 0. 載入資源
    stopwords = load_resources()
    
    # 1. 讀取資料
    print(f"\n[Step 1] 讀取資料")
    
    if not os.path.exists(COMMENTS_FILE):
        print(f"\n[錯誤] 找不到 {COMMENTS_FILE}")
        print("請先執行 04_topic_modeling.py")
        return
    
    if not os.path.exists(TOPIC_INFO_FILE):
        print(f"\n[錯誤] 找不到 {TOPIC_INFO_FILE}")
        print("請先執行 04_topic_modeling.py")
        return
    
    df = pd.read_csv(COMMENTS_FILE, encoding='utf-8-sig')
    topic_info = pd.read_csv(TOPIC_INFO_FILE, encoding='utf-8-sig')
    
    print(f"  [OK] 讀取 {len(df):,} 則評論")
    print(f"  [OK] 讀取 {len(topic_info)} 個主題資訊")
    
    # 2. 生成命名報告
    print(f"\n[Step 2] 生成命名報告（三重驗證）")
    df_reports, df_comments = generate_naming_report(df, topic_info, stopwords)
    
    # 3. 儲存 Excel
    print(f"\n[Step 3] 儲存 Excel 報告")
    save_to_excel(df_reports, df_comments, NAMING_GUIDE_FILE)
    
    # 4. 儲存文字摘要
    print(f"\n[Step 4] 生成文字摘要")
    generate_text_summary(df_reports, NAMING_SUMMARY_FILE)
    
    # 5. 生成對照表
    print(f"\n[Step 5] 生成主題名稱對照表")
    generate_naming_map(df_reports, NAMING_MAP_FILE)
    
    # 6. 完成摘要
    print("\n" + "=" * 80)
    print("主題命名指南生成完成")
    print("=" * 80)
    
    print(f"\n【輸出檔案】")
    print(f"  1. {NAMING_GUIDE_FILE}")
    print(f"     → Excel 格式，包含：")
    print(f"       • 主題命名摘要")
    print(f"       • 關鍵詞分析（BERTopic + 評論詞頻 + 整合）")
    print(f"       • 各主題 50 則代表性評論")
    print(f"  2. {NAMING_SUMMARY_FILE}")
    print(f"     → 文字摘要（口試準備）")
    print(f"  3. {NAMING_MAP_FILE}")
    print(f"     → 主題名稱對照表（CSV）")
    
    print(f"\n【推薦主題名稱】（候選2，平衡風格）")
    for _, row in df_reports.iterrows():
        print(f"  Topic {row['topic_id']}: {row['candidate_2']}")
    
    print(f"\n【下一步】")
    print(f"  1. 開啟 {NAMING_GUIDE_FILE}")
    print(f"  2. 比較「BERTopic關鍵詞」vs「評論高頻詞」vs「整合關鍵詞」")
    print(f"  3. 檢視各主題的 50 則代表性評論")
    print(f"  4. 確定最終主題名稱")
    print(f"  5. 如需修改，更新 {NAMING_MAP_FILE}")
    
    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()