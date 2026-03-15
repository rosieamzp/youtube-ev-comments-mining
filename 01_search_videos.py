"""
01_search_videos.py - 搜尋台灣電動車相關影片

功能：
1. 用關鍵字搜尋 YouTube 影片
2. 嚴格過濾非台灣地區內容
3. 儲存影片 ID 清單供後續使用

執行方式：python 01_search_videos.py
輸出：video_list.csv（包含影片基本資訊）

作者：陳莘惠
更新：2025/12
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
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
import time

def safe_print(text):
    """安全列印，處理無法編碼的字元"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))

class VideoSearcher:
    """YouTube 影片搜尋器"""
    
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.videos = []
    
    def search_by_keyword(self, keyword):
        """
        用單一關鍵字搜尋影片
        
        參數：
            keyword: 搜尋關鍵字
        
        返回：
            影片 ID 列表
        """
        video_ids = []
        
        try:
            request = self.youtube.search().list(
                q=keyword,
                part='id',
                type='video',
                regionCode='TW',              # 限定台灣地區
                relevanceLanguage='zh-TW',    # 優先繁體中文
                publishedAfter=config.START_DATE,
                maxResults=config.MAX_VIDEOS_PER_KEYWORD,
                order='relevance'             # 按相關性排序
            )
            
            response = request.execute()
            
            for item in response.get('items', []):
                video_ids.append(item['id']['videoId'])
            
            print(f"  [OK] 關鍵字 '{keyword}' 找到 {len(video_ids)} 部候選影片")
            return video_ids
        
        except HttpError as e:
            if e.resp.status == 403:
                print(f"  [X] API 配額已用盡，請明天再試")
                return None
            print(f"  [X] 搜尋失敗: {e}")
            return []
    
    def get_video_details(self, video_ids):
        """
        取得影片詳細資訊並過濾
        
        參數：
            video_ids: 影片 ID 列表
        
        返回：
            符合條件的影片資訊列表
        """
        filtered_videos = []
        
        # API 限制一次最多 50 個 ID
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            
            try:
                request = self.youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(batch_ids)
                )
                response = request.execute()
                
                for item in response.get('items', []):
                    video_info = self._extract_video_info(item)
                    
                    # 執行過濾
                    if self._should_keep_video(video_info):
                        filtered_videos.append(video_info)
                    else:
                        print(f"  [過濾] {video_info['title'][:50]}...")
            
            except HttpError as e:
                print(f"  [X] 取得影片資訊失敗: {e}")
        
        return filtered_videos
    
    def _extract_video_info(self, item):
        """從 API 回應中提取影片資訊"""
        snippet = item['snippet']
        statistics = item.get('statistics', {})
        
        return {
            'video_id': item['id'],
            'title': snippet['title'],
            'channel_id': snippet['channelId'],
            'channel_title': snippet['channelTitle'],
            'published_at': snippet['publishedAt'],
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'comment_count': int(statistics.get('commentCount', 0)),
            'language': snippet.get('defaultLanguage', ''),
            'audio_language': snippet.get('defaultAudioLanguage', ''),
        }
    
    def _should_keep_video(self, video_info):
        """
        判斷影片是否應該保留
        
        過濾條件：
        1. 語言必須不在黑名單
        2. 標題/頻道名不能包含黑名單關鍵字
        3. 必須有評論
        """
        # 1. 檢查語言
        lang = video_info['language'].lower()
        audio_lang = video_info['audio_language'].lower()
        
        if lang in config.BLOCKED_LANGUAGES or audio_lang in config.BLOCKED_LANGUAGES:
            return False
        
        # 2. 檢查標題和頻道名
        text_to_check = (video_info['title'] + ' ' + video_info['channel_title']).lower()
        
        for blocked in config.BLOCKED_KEYWORDS:
            if blocked.lower() in text_to_check:
                return False
        
        # 3. 必須有評論
        if video_info['comment_count'] == 0:
            return False
        
        return True
    
    def save_to_csv(self, videos, filename):
        """儲存影片清單到 CSV"""
        df = pd.DataFrame(videos)
        df = df.drop_duplicates(subset=['video_id'])
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        return len(df)


def main():
    print("=" * 70)
    print("Step 1: 搜尋台灣電動車相關影片")
    print("=" * 70)
    
    # 檢查 API 金鑰
    if config.API_KEY == "YOUR_API_KEY_HERE":
        print("\n[錯誤] 請先在 00_config.py 設定 API_KEY（可複製 config.example.py 為 00_config.py）")
        return
    
    # 初始化搜尋器
    searcher = VideoSearcher(config.API_KEY)
    all_videos = []
    
    print(f"\n[搜尋設定]")
    print(f"  關鍵字數量: {len(config.SEARCH_KEYWORDS)}")
    print(f"  每關鍵字影片數: {config.MAX_VIDEOS_PER_KEYWORD}")
    print(f"  時間範圍: 2024-01-01 至今")
    
    print(f"\n[開始搜尋]")
    
    # 依序搜尋每個關鍵字
    for idx, keyword in enumerate(config.SEARCH_KEYWORDS, 1):
        print(f"\n[{idx}/{len(config.SEARCH_KEYWORDS)}] 搜尋: {keyword}")
        
        # 搜尋影片 ID
        video_ids = searcher.search_by_keyword(keyword)
        
        if video_ids is None:  # API 配額用盡
            break
        
        if not video_ids:
            continue
        
        # 取得詳細資訊並過濾
        filtered = searcher.get_video_details(video_ids)
        all_videos.extend(filtered)
        
        print(f"  [OK] 保留 {len(filtered)} 部影片")
        
        # 延遲避免觸發限制
        time.sleep(1)
    
    # 儲存結果
    if all_videos:
        output_file = config.DATA_DIR + '/video_list.csv'
        total = searcher.save_to_csv(all_videos, output_file)
        
        print("\n" + "=" * 70)
        print("搜尋完成")
        print("=" * 70)
        print(f"找到影片總數: {len(all_videos)}")
        print(f"去重後影片數: {total}")
        print(f"儲存位置: {output_file}")
        
        # 統計資訊
        df = pd.DataFrame(all_videos)
        print(f"\n[統計資訊]")
        print(f"  平均觀看數: {df['view_count'].mean():,.0f}")
        print(f"  平均評論數: {df['comment_count'].mean():,.0f}")
        print(f"  頻道數量: {df['channel_id'].nunique()}")
        
        print(f"\n下一步：執行 02_collect_comments.py")
    else:
        print("\n[警告] 未找到任何符合條件的影片")


if __name__ == "__main__":
    main()
