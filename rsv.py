import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# ランダムシードの固定
np.random.seed(42)

# ==========================================
# 1. ダミーデータの生成 (真の物理現象のシミュレーション)
# ==========================================
# 論文の式(25)-(27)に基づき、"真のボラティリティ"と"観測データ"を作ります

n_days = 1000
mu_true = -1.0       # ボラティリティの平均レベル
phi_true = 0.95      # 持続性 (慣性)
sigma_eta_true = 0.2 # Volatility of Volatility (システムノイズ)
xi_true = -0.5       # RVのバイアス (観測オフセット)
sigma_u_true = 0.3   # RVの観測誤差

# 真の状態変数 h (潜在ボラティリティ) の生成
h_true = np.zeros(n_days)
h_true[0] = mu_true
for t in range(1, n_days):
    # 状態方程式: AR(1)プロセス
    h_true[t] = mu_true + phi_true * (h_true[t-1] - mu_true) + \
                np.random.normal(0, sigma_eta_true)

# 観測データの生成
# 1. リターン y_t (観測方程式1)
y_obs = np.exp(h_true / 2) * np.random.normal(0, 1, n_days)

# 2. 対数RV x_t (観測方程式2)
x_obs = xi_true + h_true + np.random.normal(0, sigma_u_true, n_days)

# データの可視化
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(h_true, color='black', label='True Latent Volatility ($h_t$)')
ax[0].set_title('Hidden State: True Volatility')
ax[1].plot(y_obs, color='blue', alpha=0.6, label='Returns ($y_t$)')
ax[1].set_title('Observation 1: Returns (Noisy)')
ax[2].plot(x_obs, color='green', alpha=0.6, label='Log RV ($x_t$)')
ax[2].set_title('Observation 2: Realized Volatility (Cleaner Proxy)')
plt.tight_layout()
plt.show()

# ==========================================
# 2. RSVモデルの構築 (PyMC)
# ==========================================
print("Building PyMC model...")

with pm.Model() as rsv_model:
    # --- A. パラメータの事前分布 (Priors) ---
    # 論文の補論や先行研究で使われる一般的な設定
    # betaは減衰係数なので0-1の範囲に制限
    phi = pm.Beta('phi', alpha=20, beta=1.5)  # 持続性は1に近い (0.9付近)
    sigma_eta = pm.InverseGamma('sigma_eta', alpha=2.5, beta=0.025) # システムノイズ
    mu = pm.Normal('mu', mu=0, sigma=10)      # 平均レベル

    xi = pm.Normal('xi', mu=0, sigma=10)      # RVのバイアス
    # Inverse Gamma分布は分散の事前分布としてよく使われる
    sigma_u = pm.InverseGamma('sigma_u', alpha=2.5, beta=0.025) # RVの観測ノイズ

    # --- B. 状態方程式 (State Equation) ---
    # h_t の遷移: h[t] ~ N(mu + phi*(h[t-1]-mu), sigma_eta)
    # PyMCの GaussianRandomWalk や AR を使えますが、ここでは明示的に書きます
    # (計算高速化のため、本来はScan等を使いますが、可読性重視でAR1として定義)

    h = pm.AR(
        'h',
        rho=[phi],
        sigma=sigma_eta,
        constant=True,
        init_dist=pm.Normal.dist(mu=mu, sigma=sigma_eta / pm.math.sqrt(1 - phi**2)),
        shape=n_days
    )
    # 注: pm.ARの仕様上、平均0に回帰するので、実際には (h - mu) をモデリングする形になりますが、
    # ここでは簡易的に h そのものを潜在変数として扱います。
    # 正確には h_centered ~ AR(phi), h = mu + h_centered です。

    # --- C. 観測方程式 (Observation Equations) ---

    # 1. リターンの尤度: y_t ~ N(0, exp(h/2))
    # 物理的解釈: 振幅 exp(h/2) で変調されたホワイトノイズ
    pm.Normal(
        'y_obs',
        mu=0,
        sigma=pm.math.exp(h / 2),
        observed=y_obs
    )

    # 2. RVの尤度: x_t ~ N(xi + h, sigma_u)
    # 物理的解釈: 真の値 h にバイアス xi と誤差 sigma_u が乗ったもの
    pm.Normal(
        'x_obs',
        mu=xi + h,
        sigma=sigma_u,
        observed=x_obs
    )

    # ==========================================
    # 3. MCMCサンプリング (推定の実行)
    # ==========================================
    print("Sampling... (This may take a while)")
    # NUTSサンプラーを使用 (論文のMulti-move samplerの現代版)
    trace = pm.sample(draws=1000, tune=1000, chains=2, target_accept=0.9)

# ==========================================
# 4. 結果の確認 (Trace Plot & Posterior)
# ==========================================
print(az.summary(trace, var_names=['phi', 'mu', 'xi', 'sigma_eta', 'sigma_u']))

# 真のボラティリティと、推定されたボラティリティ(事後平均)の比較
h_mean = trace.posterior['h'].mean(dim=("chain", "draw")).values

plt.figure(figsize=(12, 6))
plt.plot(h_true, 'k-', label='True Volatility ($h_t$)', alpha=0.5)
plt.plot(h_mean, 'r--', label='Estimated Volatility (Posterior Mean)')
plt.title('RSV Model: True vs Estimated Latent State')
plt.legend()
plt.grid(True)
plt.show()
