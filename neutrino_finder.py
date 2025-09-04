import numpy as np
from itertools import combinations, permutations

# --- 基本設定 ---
GAMMA = 20.2045
# ニュートリノの質量二乗差 (eV^2) - 標準階層を仮定
DELTA_M21_SQ = 7.5e-5
DELTA_M31_SQ = 2.5e-3

# --- 探索するフィボナッチペアと理論値Aを事前に計算 ---
# ネガフィボナッチ数を含む隣接ペアを生成
fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
nega_fibs = [1, -1, 2, -3, 5, -8, 13, -21, 34, -55, 89, -144]
fib_pairs = set()
all_fibs = sorted(list(set(fibs + nega_fibs)))

for i in range(len(all_fibs) - 1):
    pair = tuple(sorted((all_fibs[i], all_fibs[i+1])))
    fib_pairs.add(pair)

# 各ペアから理論値Aを計算
theoretical_A = {}
for k2, k1 in fib_pairs:
    a1 = k1**2 + GAMMA * k2**2
    a2 = k2**2 + GAMMA * k1**2
    theoretical_A[(k2, k1)] = {a1, a2}


# --- RMS相対誤差を計算する関数 ---
def rms_relative_error(vec1, vec2):
    # ベクトルを自身の平均値で正規化
    norm_vec1 = vec1 / np.mean(vec1)
    norm_vec2 = vec2 / np.mean(vec2)
    return np.sqrt(np.mean(((norm_vec1 - norm_vec2) / norm_vec1)**2))

# --- メインの探索 ---
print("探索を開始します。少し時間がかかります...")
results = []
A_values = sorted(list(set(val for subset in theoretical_A.values() for val in subset)))
A_triplets = list(combinations(A_values, 3))

# 最軽質量m1の探索グリッド
for m1 in np.logspace(np.log10(5e-3), np.log10(5e-2), 100): # 探索範囲を絞り込み
    m2 = np.sqrt(m1**2 + DELTA_M21_SQ)
    m3 = np.sqrt(m1**2 + DELTA_M31_SQ)
    m_vector = np.array([m1, m2, m3])

    for a_triplet in A_triplets:
        # 3つのAの値を、3つの質量に割り当てる全パターンを試す
        for p in permutations(a_triplet):
            a_vector = np.array(p)
            error = rms_relative_error(m_vector, a_vector)
            results.append((error, m1, a_vector))

# --- 結果をソートして表示 ---
results.sort(key=lambda x: x[0])
print("探索完了。上位の結果:")
print("-" * 60)
print(f"{'順位':<4} | {'RMS相対誤差':<12} | {'最軽質量 m₁':<14} | {'理論値Aの比 (A₁:A₂:A₃)'}")
print("-" * 60)

# フィボナッチペアを逆引きする関数
def find_fib_pair(a_val):
    for pair, values in theoretical_A.items():
        if abs(a_val - list(values)[0]) < 1e-6 or abs(a_val - list(values)[1]) < 1e-6:
            return pair
    return "N/A"

for i, (error, m1, a_vector) in enumerate(results[:10], 1):
    a1, a2, a3 = sorted(a_vector) # 昇順にソート
    pair1 = find_fib_pair(a1)
    pair2 = find_fib_pair(a2)
    pair3 = find_fib_pair(a3)
    
    # わかりやすくするため、理論値の比を表示
    ratio_str = f"{a1/a1:.2f} : {a2/a1:.2f} : {a3/a1:.2f}"
    
    print(f"{i:<4} | {error*100:<11.4f}% | {m1:<14.4f}eV | {ratio_str}")
    print(f"     -> Fペア割当: {pair1}, {pair2}, {pair3}\n")
