# -*- coding: utf-8 -*-
"""
简单中介模型：group -> NE -> CL
输出：
1. 总效应 c
2. a 路径
3. b 路径
4. 直接效应 c'
5. Bootstrap 间接效应 a*b 及95%置信区间
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# =========================
# 1. 读取数据
# =========================
# 把这里改成你的文件路径
file_path = "C:/Users/Shiva/Desktop/毕业论文开题/数据/新建文件夹/pyinput.xlsx"

# 如果是 Excel
df = pd.read_excel(file_path)

# 如果你是 CSV，就改成下面这行
# df = pd.read_csv(file_path)

# 只保留需要的列，并删除缺失值
df = df[["group", "NE", "CL"]].dropna().copy()

# 确保 group 是数值型，且编码为 0/1
df["group"] = pd.to_numeric(df["group"], errors="coerce")
df["NE"] = pd.to_numeric(df["NE"], errors="coerce")
df["CL"] = pd.to_numeric(df["CL"], errors="coerce")
df = df.dropna()

print("样本量:", len(df))
print(df.head())


# =========================
# 2. 拟合三个回归模型
# =========================
# c 路径：总效应
model_c = smf.ols("CL ~ group", data=df).fit()

# a 路径
model_a = smf.ols("NE ~ group", data=df).fit()

# b 路径 + c' 路径
model_b = smf.ols("CL ~ group + NE", data=df).fit()

print("\n" + "=" * 80)
print("总效应模型：CL ~ group")
print("=" * 80)
print(model_c.summary())

print("\n" + "=" * 80)
print("a路径模型：NE ~ group")
print("=" * 80)
print(model_a.summary())

print("\n" + "=" * 80)
print("b路径与直接效应模型：CL ~ group + NE")
print("=" * 80)
print(model_b.summary())


# =========================
# 3. 提取路径系数
# =========================
c = model_c.params["group"]
a = model_a.params["group"]
b = model_b.params["NE"]
c_prime = model_b.params["group"]

print("\n" + "=" * 80)
print("路径系数")
print("=" * 80)
print(f"总效应 c       = {c:.4f}")
print(f"a 路径         = {a:.4f}")
print(f"b 路径         = {b:.4f}")
print(f"直接效应 c'    = {c_prime:.4f}")
print(f"间接效应 a*b   = {a*b:.4f}")


# =========================
# 4. Bootstrap 间接效应检验
# =========================
def bootstrap_indirect_effect(
    data: pd.DataFrame,
    x: str,
    m: str,
    y: str,
    n_boot: int = 5000,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n = len(data)
    ab_list = []

    for _ in range(n_boot):
        # 有放回抽样
        sample_idx = rng.integers(0, n, n)
        boot_df = data.iloc[sample_idx].copy()

        # 估计 a 路径
        model_a_boot = smf.ols(f"{m} ~ {x}", data=boot_df).fit()

        # 估计 b 路径与 c' 路径
        model_b_boot = smf.ols(f"{y} ~ {x} + {m}", data=boot_df).fit()

        a_boot = model_a_boot.params[x]
        b_boot = model_b_boot.params[m]
        ab_list.append(a_boot * b_boot)

    ab_array = np.array(ab_list)

    result = {
        "indirect_effect_mean": np.mean(ab_array),
        "bootstrap_se": np.std(ab_array, ddof=1),
        "ci_low_2.5%": np.percentile(ab_array, 2.5),
        "ci_high_97.5%": np.percentile(ab_array, 97.5),
        "all_ab": ab_array,
    }
    return result


boot_result = bootstrap_indirect_effect(
    data=df,
    x="group",
    m="NE",
    y="CL",
    n_boot=5000,
    seed=42
)

print("\n" + "=" * 80)
print("Bootstrap 间接效应检验")
print("=" * 80)
print(f"间接效应均值    = {boot_result['indirect_effect_mean']:.4f}")
print(f"Bootstrap SE     = {boot_result['bootstrap_se']:.4f}")
print(f"95% CI 下限      = {boot_result['ci_low_2.5%']:.4f}")
print(f"95% CI 上限      = {boot_result['ci_high_97.5%']:.4f}")

if boot_result["ci_low_2.5%"] > 0 or boot_result["ci_high_97.5%"] < 0:
    print("结论：95%置信区间不包含0，间接效应显著。")
else:
    print("结论：95%置信区间包含0，间接效应不显著。")