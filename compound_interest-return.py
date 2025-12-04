import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 設定セクション
# ==========================================
ENABLE_COST = True
COST_RATE   = 0.001     # 取引コスト率 (0.1%)

# レバレッジ制御パラメータ
# 予測リターン 1% (0.01) に対して、資産の何倍(倍率)のポジションを持つか
# 例: GAIN_LEVERAGE = 50.0 なら、予測+1%で「レバレッジ0.5倍(資産の半分)」
#     予測+2%なら「レバレッジ1.0倍(資産と同額)」
GAIN_LEVERAGE = 50.0

MAX_LEVERAGE  = 2.0     # 最大レバレッジ（資産の2倍まで）
ALLOW_SHORT   = True    # 空売り許可

# ==========================================
# 1. データ生成
# ==========================================
np.random.seed(42)
days = 300
dates = pd.date_range(start='2024-01-01', periods=days)

# 上昇トレンドを持つ価格データ
trend = np.linspace(0, 1000, days)
prices = 2000 + trend + np.cumsum(np.random.randn(days) * 30)
df = pd.DataFrame(data={'Close': prices}, index=dates)

# 予測データ (ノイズあり)
noise = np.random.normal(0, 20, days)
df['Predicted_Next_Close'] = df['Close'].shift(-1) + noise

# ==========================================
# 2. 戦略ロジック: レバレッジ比率の決定
# ==========================================
# A. 予測リターン
df['Predicted_Return'] = (df['Predicted_Next_Close'] - df['Close']) / df['Close']

# B. 目標レバレッジ比率 (Target Leverage Ratio)
# これが「シグナル」になります。株数ではなく「倍率」です。
# 式: 予測リターン × ゲイン
df['Raw_Leverage'] = df['Predicted_Return'] * GAIN_LEVERAGE

# C. クリッピング (レバレッジ制限)
if ALLOW_SHORT:
    # -2.0倍 〜 +2.0倍
    df['Leverage'] = df['Raw_Leverage'].clip(lower=-MAX_LEVERAGE, upper=MAX_LEVERAGE)
else:
    # 0.0倍 〜 +2.0倍
    df['Leverage'] = df['Raw_Leverage'].clip(lower=0.0, upper=MAX_LEVERAGE)

# ==========================================
# 3. 収益計算 (リターンベース複利)
# ==========================================
# 市場リターン (%)
df['Market_Return'] = df['Close'].pct_change()

# 戦略リターン (Gross)
# レバレッジ × 市場リターン
# ※レバレッジ1.0なら市場と同じ、2.0なら2倍変動する
df['Gross_Return'] = df['Leverage'].shift(1) * df['Market_Return']

# ==========================================
# 4. コスト計算 (%)
# ==========================================
# レバレッジの変化量（ポジションを資産の何％分動かしたか）
# 例: 0.8倍 -> 1.0倍 にしたら、資産の20%分を買い増したことになる
leverage_change = df['Leverage'].diff().abs().fillna(0)

if ENABLE_COST:
    # コストも「資産に対する％」で引く
    df['Cost_Pct'] = leverage_change * COST_RATE
else:
    df['Cost_Pct'] = 0.0

# 純リターン (Net)
df['Net_Return'] = df['Gross_Return'] - df['Cost_Pct']

# ==========================================
# 5. 累積収益率 (Equity Curve)
# ==========================================
# 【重要】ここが複利計算の本体です
# (1 + r1) * (1 + r2) * ...
df['Cum_Equity'] = (1 + df['Net_Return']).cumprod()

# 初期資産を掛ければ金額になる (例: 100万円スタート)
INITIAL_CAPITAL = 1_000_000
df['Equity_Value'] = df['Cum_Equity'] * INITIAL_CAPITAL

# ==========================================
# 6. 可視化
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1, 2]})

# 上段: 価格
ax1.plot(df.index, df['Close'], label='Price', color='black')
ax1.set_title('1. Price Chart')
ax1.grid(True)

# 中段: レバレッジ推移 (倍率)
ax2.plot(df.index, df['Leverage'], label='Leverage Ratio', color='purple', drawstyle='steps-post')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_title(f'2. Leverage Ratio (Max: {MAX_LEVERAGE}x)')
ax2.set_ylabel('Leverage (x)')
ax2.grid(True)

# 下段: 資産曲線
cost_str = "ON" if ENABLE_COST else "OFF"
ax3.plot(df.index, df['Equity_Value'], label=f'Total Equity (Compounding)', color='blue', linewidth=2)
ax3.axhline(INITIAL_CAPITAL, color='red', linestyle='--', label='Initial Capital')
ax3.set_title(f'3. Equity Curve (Vectorized Compounding / Cost: {cost_str})')
ax3.set_ylabel('Equity (JPY)')
ax3.legend(loc='upper left')
ax3.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 結果サマリー
# ==========================================
final_ret = df['Cum_Equity'].iloc[-1] - 1
print(f"--- Vectorized Backtest Report ---")
print(f"最終収益率: {final_ret:.2%}")
print(f"最終資産額: {int(df['Equity_Value'].iloc[-1]):,} 円")
