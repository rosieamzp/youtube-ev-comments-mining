"""
07_regression_analysis.py - 迴歸分析（核心貢獻）

研究題目：台灣消費者購買電動汽車考量因素之研究
目標：探討哪些購買考量因素影響消費者認同度與討論熱度

研究問題：
1. 負面評論是否更容易獲得認同？（情緒 → 按讚數）
2. 哪些購買考量因素最能引起共鳴？（主題 → 按讚數）
3. 負面評論是否更容易引發討論？（情緒 → 回覆數）
4. 哪些購買考量因素最有爭議性？（主題 → 回覆數）

統計方法：負二項迴歸（Negative Binomial Regression）
理由：like_count 和 reply_count 為計數資料且過度離散

控制變數：
- video_view_count：影片觀看數
- days_since_publish：評論發布後天數
- char_count：評論字數

輸入：output/comments_with_sentiment.csv
輸出：
  - output/regression_results.csv（迴歸係數表）
  - output/regression_summary.txt（詳細報表）
  - 視覺化圖表

執行：python 07_regression_analysis.py

作者：陳莘惠
更新：2025/12
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')
matplotlib.rc('font', family=['Microsoft JhengHei', 'Arial Unicode MS'])
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 設定參數
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

COMMENTS_FILE = os.path.join(OUTPUT_DIR, "comments_with_sentiment.csv")
REGRESSION_RESULTS_FILE = os.path.join(OUTPUT_DIR, "regression_results.csv")
REGRESSION_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "regression_summary.txt")
COEF_PLOT_FILE = os.path.join(OUTPUT_DIR, "fig_regression_coefficients.png")

# ============================================================
# 1. 資料準備
# ============================================================

def prepare_regression_data(df):
    """準備迴歸分析資料：計算天數、排除噪音、虛擬變數"""
    print(f"\n[準備迴歸資料]")
    df = df[df['topic'] != -1].copy()
    print(f"  排除噪音主題後: {len(df):,} 則")
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    df['days_since_publish'] = (pd.Timestamp.now(tz='UTC') - df['published_at']).dt.days
    df['video_view_count'] = df['video_view_count'].fillna(df['video_view_count'].median())
    df['log_video_views'] = np.log1p(df['video_view_count'])
    df['log_days'] = np.log1p(df['days_since_publish'])
    df['is_negative'] = (df['sentiment'] == 0).astype(int)
    topic_dummies = pd.get_dummies(df['topic'], prefix='topic', drop_first=False)
    df = pd.concat([df, topic_dummies], axis=1)
    print(f"  ✓ 資料準備完成")
    return df

def run_negative_binomial_regression(df, formula, model_name):
    """執行負二項迴歸"""
    print(f"\n  [{model_name}]")
    print(f"  公式: {formula}")
    try:
        model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial())
        result = model.fit()
        print(f"  ✓ 模型收斂  AIC: {result.aic:.2f}  BIC: {result.bic:.2f}")
        return result
    except Exception as e:
        print(f"  ✗ 模型失敗: {e}")
        return None

def extract_regression_results(result, model_name):
    """提取迴歸係數、p值、信賴區間"""
    if result is None:
        return pd.DataFrame()
    coef_df = pd.DataFrame({
        '模型': model_name,
        '變數': result.params.index,
        '係數': result.params.values,
        '標準誤': result.bse.values,
        'z值': result.tvalues.values,
        'p值': result.pvalues.values,
        '95%信賴區間下界': result.conf_int()[0].values,
        '95%信賴區間上界': result.conf_int()[1].values
    })
    coef_df['顯著性'] = coef_df['p值'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    )
    return coef_df

def run_all_regressions(df):
    """執行四個核心迴歸模型"""
    print(f"\n[執行迴歸分析]")
    all_results = []
    formula_1 = 'like_count ~ is_negative + log_video_views + log_days + char_count'
    result_1 = run_negative_binomial_regression(df, formula_1, '模型1: 情緒→按讚數')
    if result_1:
        all_results.append(extract_regression_results(result_1, '模型1'))
    topic_vars = [v for v in [col for col in df.columns if col.startswith('topic_')] if v != 'topic_0']
    formula_2 = f"like_count ~ {' + '.join(topic_vars)} + log_video_views + log_days + char_count"
    result_2 = run_negative_binomial_regression(df, formula_2, '模型2: 主題→按讚數')
    if result_2:
        all_results.append(extract_regression_results(result_2, '模型2'))
    formula_3 = 'reply_count ~ is_negative + log_video_views + log_days + char_count'
    result_3 = run_negative_binomial_regression(df, formula_3, '模型3: 情緒→回覆數')
    if result_3:
        all_results.append(extract_regression_results(result_3, '模型3'))
    formula_4 = f"reply_count ~ {' + '.join(topic_vars)} + log_video_views + log_days + char_count"
    result_4 = run_negative_binomial_regression(df, formula_4, '模型4: 主題→回覆數')
    if result_4:
        all_results.append(extract_regression_results(result_4, '模型4'))
    return pd.concat(all_results, ignore_index=True), [result_1, result_2, result_3, result_4]

def plot_regression_coefficients(df_results, output_file):
    """繪製迴歸係數圖（僅顯著變數）"""
    print(f"\n[繪製係數圖]")
    df_sig = df_results[(df_results['p值'] < 0.05) & (df_results['變數'] != 'Intercept')].copy()
    if len(df_sig) == 0:
        print(f"  [警告] 無顯著變數")
        return
    models = df_sig['模型'].unique()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, model in enumerate(models):
        df_model = df_sig[df_sig['模型'] == model].sort_values('係數')
        ax = axes[idx]
        y_pos = np.arange(len(df_model))
        ax.errorbar(df_model['係數'], y_pos,
            xerr=[df_model['係數'] - df_model['95%信賴區間下界'],
                  df_model['95%信賴區間上界'] - df_model['係數']],
            fmt='o', markersize=8, capsize=5, color='steelblue')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_model['變數'])
        ax.set_xlabel('係數', fontsize=11)
        ax.set_title(model, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已儲存: {output_file}")
    plt.close()

def save_detailed_summary(results_list, output_file):
    """儲存詳細迴歸報表"""
    model_names = ['模型1: 情緒 → 按讚數', '模型2: 主題 → 按讚數', '模型3: 情緒 → 回覆數', '模型4: 主題 → 回覆數']
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n迴歸分析詳細報表\n" + "=" * 70 + "\n\n")
        for idx, result in enumerate(results_list):
            if result is None:
                continue
            f.write(f"\n{model_names[idx]}\n" + "=" * 70 + "\n")
            f.write(str(result.summary()) + "\n\n")
    print(f"  ✓ 已儲存: {output_file}")

def main():
    print("=" * 70)
    print("07_regression_analysis.py - 迴歸分析")
    print("=" * 70)
    if not os.path.exists(COMMENTS_FILE):
        print(f"\n[錯誤] 找不到 {COMMENTS_FILE}，請先執行 05_sentiment_analysis.py")
        return
    df = pd.read_csv(COMMENTS_FILE, encoding='utf-8-sig')
    print(f"\n[Step 1] 讀取 {len(df):,} 則評論")
    df = prepare_regression_data(df)
    print(f"\n[Step 2] 依變數描述統計：按讚數 平均={df['like_count'].mean():.2f}，回覆數 平均={df['reply_count'].mean():.2f}")
    df_results, results_list = run_all_regressions(df)
    df_results.to_csv(REGRESSION_RESULTS_FILE, index=False, encoding='utf-8-sig')
    print(f"\n[Step 3] 係數表已儲存: {REGRESSION_RESULTS_FILE}")
    save_detailed_summary(results_list, REGRESSION_SUMMARY_FILE)
    plot_regression_coefficients(df_results, COEF_PLOT_FILE)
    print("\n" + "=" * 70)
    print("迴歸分析完成")
    print("=" * 70)
    df_sig = df_results[df_results['p值'] < 0.05]
    for _, row in df_sig.iterrows():
        direction = "正向" if row['係數'] > 0 else "負向"
        print(f"  {row['模型']}: {row['變數']} ({direction}, p={row['p值']:.4f})")
    print(f"\n[下一步] 執行 08_brand_analysis.py 或 10_update_figures.py")
    print(f"\n{'=' * 70}")

if __name__ == "__main__":
    main()
