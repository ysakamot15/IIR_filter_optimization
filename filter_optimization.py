import pickle
import numpy as np
import matplotlib.pyplot as plt

def visualize_characteristic(H, omega_list, save_file_name):
    h = np.zeros(len(omega_list), dtype=np.complex)
    for i, omg in enumerate(omega_list):
        h[i] = H(omg)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    amp_h = np.abs(h)
    angle_h = np.angle(h)
    ax1.plot(omega_list, amp_h, marker=".")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(omega_list, angle_h, marker=".")

    fig.savefig(save_file_name)
    plt.close()

def compute_mean_squared_error(H_, ref_h, omega_list):
    h = np.zeros(len(omega_list), dtype=np.complex)
    for i, omg in enumerate(omega_list):
        h[i] = H_(omg)

    e = np.real((ref_h - h) * np.conjugate(ref_h - h))
    return np.mean(e)


def compute_abs_max_error(H_, ref_h, omega_list):
    h = np.zeros(len(omega_list), dtype=np.complex)
    for i, omg in enumerate(omega_list):
        h[i] = H_(omg)
    e = np.abs(ref_h - h)
    return np.max(e)


# フィルタの周波数特性を計算する関数
def frequency_characteristic_func(a0, p_array, q_array, omega):
    denomi = 1.0
    for p in  p_array:
        denomi *= (1 - p * np.exp(-1j * omega)) *\
            (1 - np.conjugate(p) * np.exp(-1j * omega))

    numer = 1.0
    for z in q_array:
        numer *= (1 - z * np.exp(-1j * omega)) *\
            (1 - np.conjugate(z) * np.exp(-1j * omega))
    return a0 * (numer/denomi)


def obj_func(p_array, q_array, a0, ref_h, omega_list, c, error_func):
    H = lambda omega: frequency_characteristic_func(a0, p_array, q_array, omega)
    max_p2 = np.max(np.real(p_array * np.conjugate(p_array)))
    e = error_func(H, ref_h, omega_list)

    return e + (c * max_p2 if max_p2 >= 1.0 else 0.0)

def annealing(HD, omega_list, M, N, c, L, sigma, T, alpha, error_func,
              init_a0=None, init_p_array=None, init_q_array=None):

    h = np.zeros(len(omega_list), dtype=np.complex)
    for i, omg in enumerate(omega_list):
        h[i] = HD(omg)

    # 初期化
    if init_a0 is None:
        a0 = 1.0
    else:
        a0 = init_a0
    if init_p_array is None:
        p_array = np.random.randn(M//2) + 1j * np.random.randn(M//2)
    else:
        p_array = np.copy(init_p_array)
    if init_q_array is None:
        q_array = np.random.randn(N//2) + 1j * np.random.randn(N//2)
    else:
        q_array = np.copy(init_q_array)

    # 目的関数F
    F = lambda p_array_, q_array_, a0_: \
        obj_func(p_array_, q_array_, a0_, h, omega_list, c, error_func)

    now_cost = F(p_array, q_array, a0)

    # 最終的に出力するフィルタ係数を初期化
    best_p_array = np.copy(p_array)
    best_q_array = np.copy(q_array)
    best_a0 = a0
    best_cost = now_cost

    for i in range(L):
        # ロールバック用に保持
        save_a0 = a0
        save_p_array = np.copy(p_array)
        save_q_array = np.copy(q_array)

        a0 += np.random.randn() * sigma
        p_array += (np.random.randn(M//2) + 1j * np.random.randn(M//2)) * sigma
        q_array += (np.random.randn(N//2) + 1j * np.random.randn(N//2)) * sigma

        tmp_cost = F(p_array, q_array, a0)
        if np.exp((now_cost - tmp_cost)/T) > np.random.rand():
            now_cost = tmp_cost
            print("now_cost=", now_cost, "T=", T, "i=", i)
            if now_cost < best_cost:
                best_a0 = a0
                best_p_array = np.copy(p_array)
                best_q_array = np.copy(q_array)
                best_cost = now_cost
                print("best_cost = ", best_cost)
        else:
            # ロールバック
            a0 = save_a0
            p_array = np.copy(save_p_array)
            q_array = np.copy(save_q_array)
        T *= alpha

    return best_a0, best_p_array, best_q_array, best_cost

if __name__ == '__main__':
    np.random.seed(10)

    # 理想的な周波数特性
    HD = lambda omg: np.exp(- 1j * 12 * omg) \
        if omg >= 2 * 0.2 * np.pi and omg <= 2 * 0.3 * np.pi  else 0.0

    # 誤差関数に利用する誤差計算に使用する周波数点の集合
    omega_list1 = np.linspace(0, 2 * 0.15 * np.pi, 200)
    omega_list2 = np.linspace(2 * 0.2 * np.pi, 2 * 0.3 * np.pi, 200)
    omega_list3 = np.linspace(2 * 0.35 * np.pi, np.pi, 200)
    omega_list = np.concatenate([omega_list1, omega_list2, omega_list3])

   # 理想的な周波数特性のグラフを作成
    visualize_characteristic(HD, omega_list, "ideal.png")

    # 5回焼きなましをして最もよかったものを採用
    best_cost = 1e9
    for i in range(5):
        a0_, p_array_, q_array_, cost =\
            annealing(HD, omega_list, 8, 8, 5,
                      10000, 0.01, 10, 0.998,
                      compute_mean_squared_error)
        if cost < best_cost:
            a0 = a0_
            p_array = np.copy(p_array_)
            q_array = np.copy(q_array_)
            best_cost = cost

    # 求めたフィルタの周波数特性
    H = lambda omega: frequency_characteristic_func(a0, p_array, q_array, omega)

    # 求めたフィルタの周波数特性のグラフを作成
    visualize_characteristic(H, np.arange(0, np.pi, 0.01), "solution.png")

    # 制約条件を満たしていることを確認
    for p in p_array:
        print("|p|=", np.abs(p))

    print("final cost=", best_cost)

    # 求めたフィルタ係数を保存
    save_dict = {"a0":a0, "p_array":p_array, "q_array":q_array}
    with open('filter_coef.pickle', 'wb') as f:
        pickle.dump(save_dict, f)
