"""
複製成 00_config.py 使用。金鑰可寫在下面 API_KEY，或先設環境變數 YOUTUBE_API_KEY。
00_config.py 請只留本機，不要 push 到版控。
"""
import os

API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for folder in [DATA_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

VIDEO_LIST_FILE = os.path.join(DATA_DIR, "video_list.csv")
COMMENTS_RAW_FILE = os.path.join(DATA_DIR, "comments_raw.csv")
COMMENTS_CLEANED_FILE = os.path.join(DATA_DIR, "comments_cleaned.csv")

SEARCH_KEYWORDS = [
    "台灣 電動車 車主", "電動車 長期 使用", "電動車 實際 心得", "電動車 後悔",
    "電動車 優缺點", "電動車 保養 成本", "電動車 保值", "電動車 充電 問題",
    "Model Y 車主 真實", "Model Y 長期 使用", "Model Y 後悔", "Model Y 優缺點",
    "Model 3 車主 心得", "Model 3 長期 使用", "Model 3 後悔", "Tesla 保固 問題",
    "Tesla 維修 費用", "Tesla 保養 成本",
    "n7 車主 真實", "n7 使用 心得", "n7 後悔", "n7 優缺點", "納智捷 n7 評價", "納智捷 n7 車主",
    "BMW iX1 車主", "iX1 使用 心得", "iX1 後悔", "BMW iX2 車主", "iX2 使用 心得",
    "BMW iX 車主", "iX 長期 評價", "BMW 電動車 保養",
    "Porsche Macan Electric 車主", "Macan Electric 心得", "Macan Electric 評價",
    "保時捷 電動車 車主", "保時捷 Macan 電動",
    "Mercedes EQE SUV 車主", "EQE SUV 使用 心得", "Mercedes EQA 車主", "EQA 使用 心得",
    "賓士 電動車 車主", "賓士 EQ 系列",
    "Volvo EX30 車主", "EX30 使用 心得", "EX30 評價", "Volvo 電動車 車主",
    "Model Y vs iX1", "Model Y vs n7", "豪華 電動車 比較", "電動車 vs 油車", "電動車 保險 費用",
]

START_DATE = "2024-01-01T00:00:00Z"
MAX_VIDEOS_PER_KEYWORD = 30
MAX_COMMENTS_PER_VIDEO = 200

BLOCKED_LANGUAGES = {'zh-CN', 'zh-Hans', 'en', 'en-US', 'en-GB', 'ja', 'ko', 'ms'}
BLOCKED_KEYWORDS = [
    '中国', '大陆', '大陸', '香港', 'Hong Kong', 'HK', '澳门', '澳門', 'Macau',
    '马来西亚', '馬來西亞', 'Malaysia', '视频', '用户', '优惠', '购买', '订阅',
    '比亚迪', 'BYD', '蔚来', '小鹏', '理想', '问界', '抖音', '快手', 'B站', '哔哩哔哩',
    '係', '唔', '嘅', '咁', '冇',
]

MIN_COMMENT_LENGTH = 10
MAX_COMMENT_LENGTH = 1000
SPAM_KEYWORDS = ['加line', '加賴', 'line id', '代購', '團購', '私訊', '密我']

COMMENT_FIELDS = ['comment_id', 'video_id', 'text', 'like_count', 'reply_count', 'published_at', 'author_channel_id']
VIDEO_FIELDS = ['video_id', 'title', 'view_count', 'like_count', 'comment_count', 'published_at', 'channel_id', 'channel_title']

def print_config():
    print("=" * 60)
    print("YouTube 資料蒐集設定")
    print("=" * 60)
    print(f"搜尋關鍵字數量: {len(SEARCH_KEYWORDS)}")
    print(f"資料存放: {DATA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    print_config()
