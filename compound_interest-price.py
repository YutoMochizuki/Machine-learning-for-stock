import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 設定セクション (Configuration)
# ==========================================
INITIAL_CAPITAL = 1_000_000 # 初期資金 100万円
ENABLE_COST     = True
COST_PER_SHARE  = 5.0       # 1株あたり手数料 5円

# 複利モデルの制御パラメータ
# 「資産の何％をリスクにさらすか」を決める
# 例: GAIN_RISK = 5.0 -> 予測+1%なら、資産の5% (0.01 * 5) をポジションにする
GAIN_RISK       = 50.0

MAX_LEVERAGE    = 2.0       # 最大レバレッジ（資産の2倍までしか持たない等）
ALLOW_SHORT     = True      # 空売り許可

# ==========================================
# 1. データ生成
# ==========================================
np.random.seed(42)
days = 300
dates = pd.date_range(start='2024-01-01', periods=days)

# 価格データ (2000円スタート)
# 上昇トレンドを作って複利の効果を見やすくします
trend = np.linspace(0, 1000, days)
prices = 2000 + trend + np.cumsum(np.random.randn(days) * 30)
df = pd.DataFrame(data={'Close': prices}, index=dates)

# 予測データ (精度そこそこ良い設定にします)
noise = np.random.normal(0, 20, days)
df['Predicted_Next_Close'] = df['Close'].shift(-1) + noise
df['Predicted_Return'] = (df['Predicted_Next_Close'] - df['Close']) / df['Close']

# ==========================================
# 2. 複利シミュレーション (Loop)
# ==========================================
# 結果を格納するリスト
history_equity = []
history_shares = []
history_cost = []

# 初期状態
current_equity = INITIAL_CAPITAL
current_shares = 0

print("Simulation Start...")

for i in range(len(df)):
    # 最終日は予測がないので取引終了
    if i == len(df) - 1:
        history_equity.append(current_equity)
        history_shares.append(current_shares)
        history_cost.append(0)
        break

    # 今日のデータ
    price = df['Close'].iloc[i]
    pred_ret = df['Predicted_Return'].iloc[i]

    # --- A. 目標ポジション額の決定 (複利の核) ---
    # 目標投資額 = 現在の総資産 × 予測リターン × ゲイン
    # 資産が増えれば、ここの金額も自動的に増える！
    target_value = current_equity * pred_ret * GAIN_RISK

    # レバレッジ制限 (資産 × Max倍 まで)
    max_pos_value = current_equity * MAX_LEVERAGE
    target_value = np.clip(target_value, -max_pos_value, max_pos_value)

    if not ALLOW_SHORT:
        target_value = max(0, target_value)

    # --- B. 目標株数への換算 ---
    target_shares = int(target_value / price) # 整数株

    # --- C. 売買実行 & コスト ---
    diff_shares = abs(target_shares - current_shares)

    if ENABLE_COST:
        cost = diff_shares * COST_PER_SHARE
    else:
        cost = 0.0

    # 資産からコストを引く
    current_equity -= cost

    # --- D. 翌日への持ち越し処理 (時価評価) ---
    # ポジション更新
    current_shares = target_shares

    # 翌日の価格変動による資産増減
    # (今日の終値 -> 翌日の終値 の変化分を資産に加える)
    next_price = df['Close'].iloc[i+1]
    price_change = next_price - price
    profit = current_shares * price_change

    current_equity += profit

    # 破産判定 (資産がマイナスになったら終了)
    if current_equity <= 0:
        current_equity = 0

    # 記録
    history_equity.append(current_equity)
    history_shares.append(current_shares)
    history_cost.append(cost)

# 結果をDataFrameに統合 (行数を合わせる)
# ループ処理の都合上、最後の1行が欠ける場合があるので調整
df = df.iloc[:len(history_equity)].copy()
df['Equity'] = history_equity
df['Shares'] = history_shares
df['Daily_Cost'] = history_cost

# ==========================================
# 3. 可視化
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1, 2]})

# 上段: 価格
ax1.plot(df.index, df['Close'], label='Price', color='black')
ax1.set_title('1. Price Chart')
ax1.grid(True)

# 中段: 保有株数 (ここが重要！)
# 資産が増えるにつれて、保有株数が指数関数的に増えていく様子を確認してください
ax2.plot(df.index, df['Shares'], label='Position Size (Shares)', color='green', drawstyle='steps-post')
ax2.set_title('2. Position Size (Compounding Effect)')
ax2.set_ylabel('Shares')
ax2.grid(True)

# 下段: 総資産曲線 (Equity Curve)
ax3.plot(df.index, df['Equity'], label='Total Equity (JPY)', color='blue', linewidth=2)
ax3.axhline(INITIAL_CAPITAL, color='red', linestyle='--', label='Initial Capital')
ax3.set_title('3. Equity Curve (Compounding)')
ax3.set_ylabel('Equity (JPY)')
ax3.legend(loc='upper left')
ax3.grid(True)

plt.tight_layout()
plt.show()

# ==========================================
# 結果サマリー
# ==========================================
final_equity = df['Equity'].iloc[-1]
return_pct = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

print(f"--- Compound Strategy Report ---")
print(f"初期資金: {INITIAL_CAPITAL:,} 円")
print(f"最終資産: {int(final_equity):,} 円")
print(f"収益率  : {return_pct:.2%}")
print(f"最大保有株数: {df['Shares'].abs().max()} 株")
