import scipy.stats as stats

# ==========================================
# 損益分布解析 (Distribution Analysis)
# ==========================================

# 1. データの抽出（NaN除去）
# 日次リターン（%）を使用します。金額ベースの場合は df['Net_PnL'] に変更してください。
returns = df['Net_Strategy_Return'].dropna()
market_returns = df['Actual_Return'].dropna()

# 2. 基本統計量の計算 (Moments)
mu = returns.mean()
sigma = returns.std()
skew = returns.skew()
kurt = returns.kurt() # Excess Kurtosis (正規分布=0)

# VaR (Value at Risk) - 95%信頼区間の最大損失
# 下位5%の点（ここより悪くなる確率は5%しかない）
var_95 = np.percentile(returns, 5)

print(f"--- 統計的性質 (Statistics) ---")
print(f"平均 (Mean Daily Return): {mu:.4%}")
print(f"標準偏差 (Volatility):   {sigma:.4%}")
print(f"歪度 (Skewness):        {skew:.4f}  (>0: 右裾が長い/損小利大傾向)")
print(f"尖度 (Kurtosis):        {kurt:.4f}  (>0: ファットテール/極端な値が多い)")
print(f"95% VaR (1日あたり):    {var_95:.4%} (これ以上の損失確率は5%)")

# ==========================================
# 3. ヒストグラムと正規分布の可視化
# ==========================================
plt.figure(figsize=(10, 6))

# A. 戦略リターンのヒストグラム (実データ)
# bins='auto' で最適なビニングを自動計算
count, bins, ignored = plt.hist(returns, bins=50, density=True, alpha=0.6, color='blue', label='Strategy Returns')

# B. 正規分布のフィット線 (理論値)
# 平均と分散が同じ正規分布を重ね書きして、形状の違い（非正規性）を見る
plt.plot(bins, stats.norm.pdf(bins, mu, sigma), linewidth=2, color='red', linestyle='--', label='Normal Dist Fit')

# C. VaRラインの描画
plt.axvline(var_95, color='black', linestyle=':', linewidth=2, label=f'95% VaR ({var_95:.2%})')

# D. ゼロライン
plt.axvline(0, color='green', linewidth=1)

plt.title('Profit/Loss Probability Distribution (Daily Returns)')
plt.xlabel('Daily Return')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 4. Q-Qプロット (正規性の検定)
# ==========================================
# データが正規分布からどれだけ乖離しているかを視覚的に確認
plt.figure(figsize=(6, 6))
stats.probplot(returns, dist="norm", plot=plt)
plt.title('Q-Q Plot (Strategy Returns vs Normal)')
plt.grid(True)
plt.show()
