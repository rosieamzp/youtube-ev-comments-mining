"""
03_clean_comments.py - 清理評論資料

功能：
1. 移除表情符號、網址、重複內容
2. 嚴格過濾非台灣內容（簡體中文、港澳用語）
3. 移除垃圾訊息、過短/過長評論
4. 保留所有研究需要的欄位

執行方式：python 03_clean_comments.py
輸入：comments_raw.csv
輸出：comments_cleaned.csv

作者：陳莘惠
更新：2025/12
"""

import pandas as pd
import re
import importlib.util
import sys
import os

# 設定 Windows 終端編碼
if os.name == 'nt':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 載入 00_config.py 作為 config 模組
spec = importlib.util.spec_from_file_location("config", "00_config.py")
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

class CommentCleaner:
    """評論清理器"""
    
    # 簡體字特徵集（繁體不會出現的字）
    SIMPLIFIED_CHARS = set('视频用户优惠购买订阅软件硬件程序信息数据网络')
    
    # 大陸/港澳用語
    MAINLAND_TERMS = [
        # 簡體常見詞
        '视频', '用户', '粉丝', '软件', '硬件', '程序', '信息', '数据',
        # 大陸網路用語
        '点赞', '关注', '订阅', '博主', '牛逼', '牛B', '傻逼',
        '厉害了', '给力', '老铁', 'yyds',
        # 大陸品牌
        '比亚迪', 'BYD', '蔚来', '小鹏', '理想', '问界',
        '抖音', '快手', 'B站', '哔哩哔哩', '小红书',
        # 港澳粵語
        '係', '唔', '嘅', '啲', '咁', '冇', '佢',
        '好似', '點解', '點樣', '邊個',
    ]
    
    def __init__(self):
        pass
    
    def clean_text(self, text):
        """
        清理文字
        移除：表情、網址、email、多餘空白
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # 移除表情符號
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F900-\U0001F9FF"
            "\U00002600-\U000027BF"
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # 移除網址
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # 移除 email
        text = re.sub(r'\S+@\S+', '', text)
        
        # 壓縮空白
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def count_chinese(self, text):
        """計算中文字數"""
        if pd.isna(text):
            return 0
        return len(re.findall(r'[\u4e00-\u9fff]', str(text)))
    
    def has_simplified_chinese(self, text):
        """檢查是否包含簡體字"""
        if pd.isna(text) or not text:
            return False
        
        # 計算簡體字比例
        chinese_count = self.count_chinese(text)
        if chinese_count < 5:
            return False
        
        simplified_count = sum(1 for char in text if char in self.SIMPLIFIED_CHARS)
        ratio = simplified_count / chinese_count
        
        return ratio > 0.1  # 超過 10% 就判定為簡體
    
    def has_mainland_terms(self, text):
        """檢查是否包含大陸/港澳用語"""
        if pd.isna(text) or not text:
            return False
        
        text_lower = str(text).lower()
        for term in self.MAINLAND_TERMS:
            if term.lower() in text_lower:
                return True
        return False
    
    def is_spam(self, text):
        """判斷是否為垃圾訊息"""
        if not text or len(text.strip()) == 0:
            return True
        
        text_lower = text.lower()
        
        # 檢查垃圾關鍵字
        for spam_word in config.SPAM_KEYWORDS:
            if spam_word.lower() in text_lower:
                return True
        
        # 台灣手機號碼
        if re.search(r'09\d{8}', text):
            return True
        
        # 過多重複符號
        if re.search(r'(.)\1{10,}', text):
            return True
        
        return False
    
    def clean_dataframe(self, df):
        """
        清理整個 DataFrame
        
        返回：
            清理後的 DataFrame
        """
        print(f"\n[清理前] {len(df):,} 則評論")
        initial_count = len(df)
        
        # 1. 基本文字清理
        print(f"\n[Step 1] 基本文字清理")
        df['text_cleaned'] = df['text'].apply(self.clean_text)
        df['char_count'] = df['text_cleaned'].apply(self.count_chinese)
        
        # 2. 移除空白內容
        before = len(df)
        df = df[df['text_cleaned'].str.len() > 0]
        print(f"  移除空白: {before - len(df):,} 則")
        
        # 3. 過濾簡體中文
        print(f"\n[Step 2] 過濾簡體中文")
        before = len(df)
        df = df[~df['text_cleaned'].apply(self.has_simplified_chinese)]
        print(f"  移除簡體: {before - len(df):,} 則")
        
        # 4. 過濾大陸/港澳用語
        print(f"\n[Step 3] 過濾大陸/港澳用語")
        before = len(df)
        df = df[~df['text_cleaned'].apply(self.has_mainland_terms)]
        print(f"  移除大陸用語: {before - len(df):,} 則")
        
        # 5. 過濾過短評論
        print(f"\n[Step 4] 過濾長度")
        before = len(df)
        df = df[df['char_count'] >= config.MIN_COMMENT_LENGTH]
        print(f"  移除過短(<{config.MIN_COMMENT_LENGTH}字): {before - len(df):,} 則")
        
        before = len(df)
        df = df[df['char_count'] <= config.MAX_COMMENT_LENGTH]
        print(f"  移除過長(>{config.MAX_COMMENT_LENGTH}字): {before - len(df):,} 則")
        
        # 6. 過濾垃圾訊息
        print(f"\n[Step 5] 過濾垃圾訊息")
        before = len(df)
        df = df[~df['text_cleaned'].apply(self.is_spam)]
        print(f"  移除垃圾: {before - len(df):,} 則")
        
        # 7. 移除重複內容
        print(f"\n[Step 6] 移除重複內容")
        before = len(df)
        df = df.drop_duplicates(subset=['text_cleaned'], keep='first')
        print(f"  移除重複: {before - len(df):,} 則")
        
        # 8. 重設索引
        df = df.reset_index(drop=True)
        
        final_count = len(df)
        retention_rate = final_count / initial_count * 100
        
        print(f"\n[清理完成]")
        print(f"  保留: {final_count:,} 則 ({retention_rate:.1f}%)")
        print(f"  移除: {initial_count - final_count:,} 則")
        
        return df


def main():
    print("=" * 70)
    print("Step 3: 清理評論資料")
    print("=" * 70)
    
    # 讀取原始資料
    input_file = config.COMMENTS_RAW_FILE
    
    if not os.path.exists(input_file):
        print(f"\n[錯誤] 找不到 {input_file}")
        print("請先執行 02_collect_comments.py")
        return
    
    print(f"\n[讀取] {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"\n[原始資料]")
    print(f"  評論數: {len(df):,}")
    print(f"  影片數: {df['video_id'].nunique()}")
    print(f"  欄位: {df.columns.tolist()}")
    
    # 執行清理
    cleaner = CommentCleaner()
    df_clean = cleaner.clean_dataframe(df)
    
    # 整理輸出欄位
    output_columns = [
        'comment_id',
        'video_id',
        'text_cleaned',
        'char_count',
        'like_count',
        'reply_count',
        'published_at',
        'author_channel_id',
        'video_view_count',
        'video_like_count',
        'video_comment_count',
        'video_published_at',
        'channel_id',
        'channel_title',
    ]
    
    # 只保留存在的欄位
    output_columns = [col for col in output_columns if col in df_clean.columns]
    df_output = df_clean[output_columns].copy()
    
    # 儲存結果
    output_file = config.COMMENTS_CLEANED_FILE
    df_output.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 70)
    print("清理完成")
    print("=" * 70)
    print(f"儲存位置: {output_file}")
    
    # 統計資訊
    print(f"\n[統計資訊]")
    print(f"  平均字數: {df_output['char_count'].mean():.1f}")
    print(f"  平均按讚數: {df_output['like_count'].mean():.1f}")
    print(f"  平均回覆數: {df_output['reply_count'].mean():.1f}")
    print(f"  有回覆的評論: {(df_output['reply_count'] > 0).sum():,} 則")
    
    # 顯示樣本
    print(f"\n[清理後樣本（前3則）]")
    for idx, row in df_output.head(3).iterrows():
        print(f"\n[{row['char_count']}字 | 讚{row['like_count']} 回{row['reply_count']}]")
        print(f"{row['text_cleaned'][:100]}...")
    
    print(f"\n下一步：執行 04_topic_modeling.py")


if __name__ == "__main__":
    main()
