import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# --- 1. モデル定義関数群 ---

def get_leverage_effect(z, tau1, tau2):
    """
    レバレッジ関数 tau(z) の計算
    """
    return tau1 * z + tau2 * (z**2 - 1)

def calculate_conditional_variance(params, x):
    """
    パラメータと過去のRealized Measure (x) から条件付き分散 (h) を計算
    Log-Linear GARCH Equation: log(h_t) = omega + beta * log(h_{t-1}) + gamma * x_{t-1}
    """
    omega, beta, gamma = params[0], params[1], params[2]
    n = len(x)
    log_h = np.zeros(n)

    # 初期値: 全体の分散の対数などで近似
    # ここでは便宜的に x (log RV) の平均から推定
    log_h[0] = np.mean(x)

    # 再帰計算 (ここが計算のボトルネックになる部分)
    for t in range(1, n):
        log_h[t] = omega + beta * log_h[t-1] + gamma * x[t-1]

    return np.exp(log_h)

def realized_garch_neg_likelihood(params, r, x):
    """
    最小化するための負の対数尤度関数
    params: [omega, beta, gamma, xi, phi, tau1, tau2, sigma_u]
    """
    # パラメータのアンパック
    omega, beta, gamma, xi, phi, tau1, tau2, sigma_u = params

    # 計算安定性のための安全装置（boundsでも制御するが念のため）
    if sigma_u <= 1e-6:
        return 1e10

    # 1. 条件付き分散 h_t の取得
    h = calculate_conditional_variance(params, x)

    # 2. 標準化残差 z_t の計算
    # r_t = sqrt(h_t) * z_t  =>  z_t = r_t / sqrt(h_t)
    z = r / np.sqrt(h)

    # 3. リターン方程式の対数尤度 (Return LL)
    ll_r = -0.5 * (np.log(2 * np.pi) + np.log(h) + (r**2) / h)

    # 4. 観測方程式の対数尤度 (Measurement LL)
    # x_t = xi + phi * log(h_t) + tau(z_t) + u_t
    tau_z = get_leverage_effect(z, tau1, tau2)
    u = x - (xi + phi * np.log(h) + tau_z)

    ll_m = -0.5 * (np.log(2 * np.pi) + np.log(sigma_u**2) + (u**2) / (sigma_u**2))

    # 負の対数尤度の総和を返す
    return -np.sum(ll_r + ll_m)

# --- 2. 実行・解析スクリプト ---

def run_analysis():
    # --- A. 合成データの生成 (Simulation) ---
    np.random.seed(42)
    n_obs = 3000

    # 真のパラメータ (True Parameters)
    # omega, beta, gamma, xi, phi, tau1, tau2, sigma_u
    true_params = [0.02, 0.6, 0.4, -0.2, 1.0, -0.15, 0.05, 0.4]

    # データ格納用配列
    log_h_true = np.zeros(n_obs)
    r_sim = np.zeros(n_obs)
    x_sim = np.zeros(n_obs)
    z_sim = np.random.normal(0, 1, n_obs)
    u_sim = np.random.normal(0, true_params[7], n_obs)

    # 初期値設定
    log_h_true[0] = 0.0
    r_sim[0] = np.sqrt(np.exp(log_h_true[0])) * z_sim[0]
    x_sim[0] = true_params[3] + true_params[4]*log_h_true[0] + \
               get_leverage_effect(z_sim[0], true_params[5], true_params[6]) + u_sim[0]

    # 時系列生成ループ
    omega, beta, gamma = true_params[0], true_params[1], true_params[2]
    xi, phi = true_params[3], true_params[4]
    tau1, tau2 = true_params[5], true_params[6]

    for t in range(1, n_obs):
        # GARCH式: log(h_t)
        log_h_true[t] = omega + beta * log_h_true[t-1] + gamma * x_sim[t-1]

        # リターン生成
        r_sim[t] = np.sqrt(np.exp(log_h_true[t])) * z_sim[t]

        # Realized Measure (RV) 生成
        tau_z = get_leverage_effect(z_sim[t], tau1, tau2)
        x_sim[t] = xi + phi * log_h_true[t] + tau_z + u_sim[t]

    print(f"Data Generated: {n_obs} points")

    # --- B. パラメータ推定 (Optimization) ---

    # 初期値 (Initial Guess)
    # [omega, beta, gamma, xi, phi, tau1, tau2, sigma_u]
    initial_guess = [0.0, 0.5, 0.2, 0.0, 1.0, 0.0, 0.0, 0.5]

    # パラメータの範囲制約 (Bounds)
    # sigma_u > 0.001, betaは発散を防ぐため -1 ~ 1 程度に制限
    bounds = [
        (None, None),   # omega
        (-0.99, 0.99),  # beta
        (None, None),   # gamma
        (None, None),   # xi
        (None, None),   # phi
        (None, None),   # tau1
        (None, None),   # tau2
        (0.001, None)   # sigma_u
    ]

    print("Optimizing parameters...")
    result = opt.minimize(
        realized_garch_neg_likelihood,
        initial_guess,
        args=(r_sim, x_sim),
        method='L-BFGS-B', # 制約付き最適化に適した手法
        bounds=bounds,
        options={'disp': True}
    )

    est_params = result.x

    # --- C. 結果の表示と可視化 ---
    param_names = ["omega", "beta", "gamma", "xi", "phi", "tau1", "tau2", "sigma_u"]

    print("\n--- Parameter Estimation Results ---")
    print(f"{'Param':<10} | {'True':<10} | {'Estimated':<10}")
    print("-" * 36)
    for name, t_val, e_val in zip(param_names, true_params, est_params):
        print(f"{name:<10} | {t_val:>10.4f} | {e_val:>10.4f}")

    # 推定された分散を計算
    h_est = calculate_conditional_variance(est_params, x_sim)
    vol_est = np.sqrt(h_est)
    vol_true = np.sqrt(np.exp(log_h_true))

    plt.figure(figsize=(12, 6))
    # 視認性のため、直近200点のみプロット
    plt.plot(vol_true[-200:], label='True Volatility', color='black', alpha=0.7)
    plt.plot(vol_est[-200:], label='Estimated Volatility', color='red', linestyle='--', alpha=0.8)
    plt.title('Realized GARCH: True vs Estimated Volatility (Last 200 days)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_analysis()
