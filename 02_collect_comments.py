"""
02_collect_comments.py - 蒐集評論與影片資訊

功能：
1. 從影片清單中批次抓取評論
2. 抓取完整欄位：like_count, reply_count（重要！）
3. 同時蒐集影片資訊作為控制變數
4. 初步過濾簡體中文

執行方式：python 02_collect_comments.py
輸入：video_list.csv
輸出：comments_raw.csv

作者：陳莘惠
更新：2025/12
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import importlib.util
import sys
import os
import time
import opencc

# 設定 Windows 終端編碼
if os.name == 'nt':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 載入 00_config.py 作為 config 模組
spec = importlib.util.spec_from_file_location("config", "00_config.py")
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

class CommentCollector:
    """YouTube 評論蒐集器"""
    
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.converter = opencc.OpenCC('s2t')  # 簡繁轉換器
    
    def collect_from_video(self, video_id, max_results):
        """
        從單一影片抓取評論（僅頂層評論）
        
        參數：
            video_id: 影片 ID
            max_results: 最多抓取評論數
        
        返回：
            評論列表（含影片資訊）
        """
        comments = []
        
        try:
            # 先取得影片資訊（作為控制變數）
            video_info = self._get_video_info(video_id)
            
            # 抓取評論
            request = self.youtube.commentThreads().list(
                part='snippet',  # 只要 snippet 就夠了
                videoId=video_id,
                maxResults=100,
                textFormat='plainText',
                order='relevance'  # 按相關性排序
            )
            
            while request and len(comments) < max_results:
                response = request.execute()
                
                for item in response.get('items', []):
                    comment = self._extract_comment_data(item, video_info)
                    
                    # 過濾簡體中文
                    if self._is_traditional_chinese(comment['text']):
                        comments.append(comment)
                
                # 處理分頁
                if 'nextPageToken' in response and len(comments) < max_results:
                    request = self.youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=100,
                        pageToken=response['nextPageToken'],
                        textFormat='plainText',
                        order='relevance'
                    )
                else:
                    break
            
            return comments
        
        except HttpError as e:
            if 'commentsDisabled' in str(e):
                print(f"    [跳過] 影片已關閉評論")
            elif 'quotaExceeded' in str(e):
                print(f"    [錯誤] API 配額用盡")
                return None
            else:
                print(f"    [錯誤] {e}")
            return []
    
    def _get_video_info(self, video_id):
        """取得影片資訊（控制變數）"""
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                return {}
            
            item = response['items'][0]
            snippet = item['snippet']
            stats = item.get('statistics', {})
            
            return {
                'video_view_count': int(stats.get('viewCount', 0)),
                'video_like_count': int(stats.get('likeCount', 0)),
                'video_comment_count': int(stats.get('commentCount', 0)),
                'video_published_at': snippet['publishedAt'],
                'channel_id': snippet['channelId'],
                'channel_title': snippet['channelTitle'],
            }
        except Exception as e:
            print(f"    [警告] 無法取得影片資訊: {e}")
            return {}
    
    def _extract_comment_data(self, item, video_info):
        """
        從 API 回應提取評論資料
        
        重點：確保抓取 totalReplyCount（這是依變數！）
        """
        snippet = item['snippet']
        top_comment = snippet['topLevelComment']['snippet']
        
        comment_data = {
            # === 評論基本資訊 ===
            'comment_id': item['id'],
            'video_id': snippet['videoId'],
            'text': top_comment['textDisplay'],
            
            # === 依變數（重要！）===
            'like_count': top_comment['likeCount'],
            'reply_count': snippet['totalReplyCount'],  # 這則評論被回覆幾次
            
            # === 其他資訊 ===
            'published_at': top_comment['publishedAt'],
            'author_channel_id': top_comment.get('authorChannelId', {}).get('value', ''),
            
            # === 控制變數（從影片資訊帶入）===
            'video_view_count': video_info.get('video_view_count', 0),
            'video_like_count': video_info.get('video_like_count', 0),
            'video_comment_count': video_info.get('video_comment_count', 0),
            'video_published_at': video_info.get('video_published_at', ''),
            'channel_id': video_info.get('channel_id', ''),
            'channel_title': video_info.get('channel_title', ''),
        }
        
        return comment_data
    
    def _is_traditional_chinese(self, text):
        """
        檢測是否為繁體中文（簡單版）
        用 OpenCC 轉換，若轉換後改變則判定為簡體
        """
        try:
            if not text or len(text.strip()) < 5:
                return True  # 太短無法判斷，先保留
            
            converted = self.converter.convert(text)
            return converted == text  # 沒改變表示是繁體
        except:
            return True  # 轉換失敗，保守保留


def main():
    print("=" * 70)
    print("Step 2: 蒐集評論資料")
    print("=" * 70)
    
    # 檢查 API 金鑰
    if config.API_KEY == "YOUR_API_KEY_HERE":
        print("\n[錯誤] 請先在 00_config.py 設定 API_KEY")
        return
    
    # 讀取影片清單
    video_file = config.DATA_DIR + '/video_list.csv'
    
    try:
        df_videos = pd.read_csv(video_file, encoding='utf-8-sig')
        print(f"\n[讀取] 影片清單: {len(df_videos)} 部")
    except FileNotFoundError:
        print(f"\n[錯誤] 找不到 {video_file}")
        print("請先執行 01_search_videos.py")
        return
    
    # 初始化蒐集器
    collector = CommentCollector(config.API_KEY)
    all_comments = []
    
    print(f"\n[蒐集設定]")
    print(f"  每影片評論數: {config.MAX_COMMENTS_PER_VIDEO}")
    print(f"  目標總評論數: 約 {len(df_videos) * config.MAX_COMMENTS_PER_VIDEO:,} 則")
    
    print(f"\n[開始蒐集]")
    
    # 逐一處理每部影片
    for idx, row in df_videos.iterrows():
        video_id = row['video_id']
        title = row['title']
        
        print(f"\n[{idx+1}/{len(df_videos)}] {title[:50]}...")
        print(f"  影片ID: {video_id}")
        
        # 抓取評論
        comments = collector.collect_from_video(
            video_id, 
            config.MAX_COMMENTS_PER_VIDEO
        )
        
        if comments is None:  # API 配額用盡
            print("\n[系統] API 配額已用盡，停止蒐集")
            break
        
        if comments:
            all_comments.extend(comments)
            print(f"  [OK] 抓取 {len(comments)} 則評論（已過濾簡體）")
        
        # 延遲避免觸發限制
        time.sleep(1)
    
    # 儲存結果
    if all_comments:
        df_comments = pd.DataFrame(all_comments)
        output_file = config.COMMENTS_RAW_FILE
        df_comments.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 70)
        print("蒐集完成")
        print("=" * 70)
        print(f"總評論數: {len(df_comments):,}")
        print(f"影片數: {df_comments['video_id'].nunique()}")
        print(f"儲存位置: {output_file}")
        
        # 統計資訊
        print(f"\n[統計資訊]")
        print(f"  平均按讚數: {df_comments['like_count'].mean():.1f}")
        print(f"  平均回覆數: {df_comments['reply_count'].mean():.1f}")
        print(f"  有回覆的評論: {(df_comments['reply_count'] > 0).sum():,} 則")
        
        # 確認欄位完整性
        print(f"\n[欄位檢查]")
        required_fields = ['like_count', 'reply_count', 'video_view_count']
        for field in required_fields:
            if field in df_comments.columns:
                print(f"  [OK] {field}")
            else:
                print(f"  [X] {field} 缺失！")
        
        print(f"\n下一步：執行 03_clean_comments.py")
    else:
        print("\n[警告] 未蒐集到任何評論")


if __name__ == "__main__":
    main()
