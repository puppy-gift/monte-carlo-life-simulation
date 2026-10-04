# -*- coding: utf-8 -*-
"""
同一学校出来的人，为什么出了社会差距越拉越大？
—— 带【完整轨迹记录 + 反事实策略对比 + 神级词条校准】的蒙特卡洛模拟

直接运行：  python this_file.py
输出：
  - simulation_summary.csv         每人一行汇总
  - counterfactual_summary.csv     反事实：同一个人 8 种策略对比
  - wash_improvement.csv           每种洗词条策略的有效词条总提升量
  - trajectories.npz               所有人的逐年轨迹（npz 压缩）
  - world_0000.csv ...             指定世界的逐帧长表
  - fig1~fig9 *.png                可视化图
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

# ============================================================================
# ==============================  CONFIG  ====================================
# ============================================================================
CONFIG = {
    # ---------------- 基本规模 ----------------
    "n_people": 100,
    "n_worlds": 1000,
    "n_years": 40,
    "seed": 20240501,

    # ---------------- 词条 ----------------
    "traits": ["智商", "学习能力", "自制力", "情商", "判断力",
               "批判性思维", "承担责任", "抗压", "无用词条"],
    "useless_trait": "无用词条",
    "corr_base": 0.10,
    "corr_pairs": {
        ("学习能力", "智商"): 0.50,
        ("学习能力", "自制力"): 0.40,
        ("自制力", "承担责任"): 0.40,
        ("判断力", "批判性思维"): 0.50,
        ("情商", "判断力"): 0.30,
        ("抗压", "自制力"): 0.30,
    },
    "corr_scale": 1.0,

    # 神级词条
    "god_trait_prob": 0.02,          # 每个人 2% 概率获得一个神级词条
    "god_trait_bonus": 4.0,          # 神级词条的绝对水平：设为 max(原值, 4.0)
    "god_income_bonus": 0.0,         # 神级者在收入公式里的额外对数加成（校准自动调整）

    "school_weights": {"智商": 0.45, "学习能力": 0.40, "自制力": 0.15},

    "social_weights": {
        "智商": 0.13, "学习能力": 0.09, "自制力": 0.19, "情商": 0.11,
        "判断力": 0.19, "批判性思维": 0.09, "承担责任": 0.10, "抗压": 0.10,
    },
    "trait_scale": 1.6,
    "feedback_ramp_start": 0.35,
    "feedback_ramp_years": 12.0,

    "energy_per_year": 1.0,
    "gain_per_energy": 0.02,
    "spillover": 0.50,

    "family_lognorm_mu": 0.2,
    "family_lognorm_sigma": 0.7,
    "rich_prob": 0.03,
    "rich_multiplier": 6.0,
    "safety_net_multiple": 3.0,

    "industries": ["土木", "互联网", "金融", "制造", "教育"],
    "civil_idx": 0,
    "civil_boom_years_range": (8, 21),
    "civil_boom_bonus": 0.45,
    "civil_crash_drift": -0.10,
    "industry_rw_sigma": 0.12,
    "industry_rw_meanrev": 0.98,
    "industry_choice_temp": 3.0,
    "switch_threshold": 0.60,
    "switch_cost": 0.80,
    "switch_cooldown": 5,

    # 拥挤效应与 all in 破产风险
    "congestion_factor": 0.50,
    "all_in_risk_factor": 0.30,
    "all_in_risk_threshold": 0.50,

    "base_income": 1.0,
    "min_income": 0.20,
    "luck_sigma": 0.25,
    "good_event_prob": 0.010,
    "good_event_mu": 1.20,
    "good_event_sigma": 0.50,
    "bad_event_prob": 0.015,
    "bad_event_lo": 0.15,
    "bad_event_hi": 0.55,

    "save_base": 0.10,
    "save_per_conscientious": 0.12,
    "save_min": 0.02,
    "save_max": 0.60,
    "invest_base": 0.20,
    "invest_per_judgment": 0.08,
    "invest_per_family": 0.04,
    "invest_min": 0.05,
    "invest_max": 0.85,
    "invest_ret_base": 0.05,
    "invest_ret_per_judgment": 0.03,
    "invest_ret_sigma": 0.22,
    "blowup_prob_base": 0.020,
    "blowup_prob_per_judgment": 0.008,
    "blowup_prob_min": 0.002,
    "blowup_prob_max": 0.30,

    # ---------------- 轨迹导出选项 ----------------
    "export_csv_worlds": (0, 1, 2),
    "export_traits_hist": True,
}

# ============================================================================
# ==============================  工具函数  ==================================
# ============================================================================

def set_chinese_font():
    candidates = ["SimHei", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB",
                  "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Source Han Sans SC",
                  "Heiti SC", "STHeiti", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.sans-serif"] = [c]
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[字体] 使用中文字体: {c}")
            return c
    plt.rcParams["axes.unicode_minus"] = False
    print("[字体] 未找到可用中文字体，图中中文可能显示为方块。")
    return None


def build_corr_matrix(traits, pairs, base, useless):
    n = len(traits)
    idx = {t: i for i, t in enumerate(traits)}
    C = np.full((n, n), base, dtype=float)
    np.fill_diagonal(C, 1.0)
    for (a, b), v in pairs.items():
        i, j = idx[a], idx[b]
        C[i, j] = C[j, i] = v
    u = idx[useless]
    C[u, :] = 0.0
    C[:, u] = 0.0
    C[u, u] = 1.0
    return C


def nearest_pd_corr(C, eps=1e-8):
    C = (C + C.T) / 2.0
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    C2 = (V * w) @ V.T
    d = np.sqrt(np.diag(C2))
    C2 = C2 / np.outer(d, d)
    np.fill_diagonal(C2, 1.0)
    return C2


def gini_per_row(X):
    X = np.sort(np.clip(X, 0, None), axis=1)
    n = X.shape[1]
    idx = np.arange(1, n + 1)
    num = 2.0 * (X * idx).sum(axis=1)
    den = X.sum(axis=1) * n
    return num / np.maximum(den, 1e-12) - (n + 1.0) / n


# ============================================================================
# ==============================  核心模拟  ==================================
# ============================================================================

def simulate(cfg, wash_strategy=None, hedge_flag=None,
             return_trajectories=True, verbose=True):
    """
    如果 wash_strategy / hedge_flag 为 None，则内部随机生成；
    否则使用传入的数组（长度 P）。
    """
    rng = np.random.default_rng(cfg["seed"])

    P = int(cfg["n_people"])
    W = int(cfg["n_worlds"])
    Y = int(cfg["n_years"])

    traits_names = list(cfg["traits"])
    T = len(traits_names)
    ti = {n: i for i, n in enumerate(traits_names)}
    useless_i = ti[cfg["useless_trait"]]

    industries = list(cfg["industries"])
    n_ind = len(industries)
    civil_i = cfg["civil_idx"]

    # ---------- 1. 相关矩阵 & 元能力 ----------
    C = build_corr_matrix(traits_names, cfg["corr_pairs"], cfg["corr_base"],
                          cfg["useless_trait"])
    if cfg["corr_scale"] != 1.0:
        off = ~np.eye(T, dtype=bool)
        C[off] *= cfg["corr_scale"]
    C = nearest_pd_corr(C)
    L = np.linalg.cholesky(C)

    row_sum = C.sum(axis=1) - 1.0
    row_sum[useless_i] = -np.inf
    meta_idx = int(np.argmax(row_sum))
    meta_name = traits_names[meta_idx]
    if verbose:
        print(f"[元能力] 关联总和最高的词条：{meta_name}"
              f"（关联总和 {row_sum[meta_idx]:.3f}）")

    # ---------- 2. 生成这批人 ----------
    Z = rng.standard_normal((P, T))
    traits0 = Z @ L.T

    # 神级词条：直接设为绝对水平 max(原值, 4.0)
    valid_idx = np.array([i for i in range(T) if i != useless_i])
    god_flag = rng.random(P) < cfg["god_trait_prob"]
    god_trait = np.full(P, -1, dtype=int)
    ng = int(god_flag.sum())
    god_persons = np.where(god_flag)[0] if ng > 0 else np.array([], dtype=int)
    chosen = np.array([], dtype=int)
    if ng > 0:
        chosen = rng.choice(valid_idx, size=ng, replace=True)
        god_trait[god_persons] = chosen
        for p, j in zip(god_persons, chosen):
            traits0[p, j] = max(traits0[p, j], cfg["god_trait_bonus"])

    # 断言：所有神级词条的数值必须 >= god_trait_bonus
    if ng > 0:
        for p in range(P):
            if god_trait[p] >= 0:
                j = god_trait[p]
                assert traits0[p, j] >= cfg["god_trait_bonus"] - 1e-9, \
                    f"神级词条数值异常: 人 {p} 词条 {traits_names[j]} 数值={traits0[p, j]}"

    # 打印神级者信息，核对标签和数值一致
    if verbose:
        if ng > 0:
            print(f"\n[神级词条] {P} 人中 {ng} 人拥有神级词条"
                  f"（绝对水平 >= {cfg['god_trait_bonus']:.1f}）：")
            for p, j in zip(god_persons, chosen):
                print(f"  人物 #{p:3d}: 词条「{traits_names[j]}」 = {traits0[p, j]:+.4f}")
        else:
            print("[神级词条] 本次没有神级者。")

    family = rng.lognormal(cfg["family_lognorm_mu"], cfg["family_lognorm_sigma"], P)
    rich_mask = rng.random(P) < cfg["rich_prob"]
    family[rich_mask] *= cfg["rich_multiplier"]

    school_w = np.array([cfg["school_weights"].get(n, 0.0) for n in traits_names])
    school_score = traits0 @ school_w
    school_rank = stats.rankdata(-school_score, method="ordinal").astype(int)

    wash_names = ["专洗元能力", "随机洗", "专洗无用词条", "不洗"]
    if wash_strategy is None:
        wash_strategy = rng.choice(len(wash_names), size=P, p=[0.25, 0.25, 0.25, 0.25])
    else:
        wash_strategy = np.asarray(wash_strategy, dtype=int)

    if hedge_flag is None:
        hedge_flag = rng.random(P) < 0.5
    else:
        hedge_flag = np.asarray(hedge_flag, dtype=bool)

    # ---------- 3. 逐年词条演变 ----------
    g = cfg["energy_per_year"] * cfg["gain_per_energy"]
    traits_y = np.zeros((Y, P, T), dtype=np.float32)
    cur = traits0.copy()
    for y in range(Y):
        traits_y[y] = cur
        for p in range(P):
            s = wash_strategy[p]
            if s == 3:
                continue
            if s == 0:
                j = meta_idx
            elif s == 1:
                j = int(rng.integers(0, T))
            else:
                j = useless_i
            d = cfg["spillover"] * C[j, :] * g
            d[j] = g
            cur[p] += d

    # ---------- 4. 收入系数（含反馈延迟 + 神级额外加成） ----------
    soc_w = np.array([cfg["social_weights"].get(n, 0.0) for n in traits_names])
    raw = traits_y @ soc_w
    ramp = np.minimum(
        1.0,
        cfg["feedback_ramp_start"] +
        (1.0 - cfg["feedback_ramp_start"]) * np.arange(Y) /
        max(cfg["feedback_ramp_years"] - 1.0, 1.0)
    )
    # 神级者的额外对数加成（每年固定，不衰减）
    god_bonus_vec = np.where(god_trait >= 0, cfg["god_income_bonus"], 0.0)  # (P,)
    trait_coef = np.exp(cfg["trait_scale"] * raw * ramp[:, None]
                        + god_bonus_vec[None, :])

    # ---------- 5. 行业景气 ----------
    log_bo = np.zeros((W, Y, n_ind))
    log_bo[:, 0, :] = rng.normal(0.0, cfg["industry_rw_sigma"], (W, n_ind))
    for y in range(1, Y):
        log_bo[:, y, :] = (cfg["industry_rw_meanrev"] * log_bo[:, y - 1, :] +
                           rng.normal(0.0, cfg["industry_rw_sigma"], (W, n_ind)))

    lo, hi = cfg["civil_boom_years_range"]
    crash_year = rng.integers(lo, hi, size=W)
    yidx = np.arange(Y)[None, :]
    pre = yidx < crash_year[:, None]
    post = np.maximum(yidx - crash_year[:, None], 0)
    log_bo[:, :, civil_i] += np.where(pre, cfg["civil_boom_bonus"], 0.0)
    log_bo[:, :, civil_i] += np.where(~pre, cfg["civil_crash_drift"] * post, 0.0)

    bo = np.exp(log_bo)
    rel_bo = bo / np.maximum(bo.mean(axis=2, keepdims=True), 1e-9)

    logits = log_bo[:, 0, :] * cfg["industry_choice_temp"]
    logits -= logits.max(axis=1, keepdims=True)
    p_ind = np.exp(logits)
    p_ind /= p_ind.sum(axis=1, keepdims=True)
    cum = np.cumsum(p_ind, axis=1)
    u = rng.random((W, P))
    ind = np.clip((u[:, :, None] > cum[:, None, :]).sum(axis=2), 0, n_ind - 1)
    ind = ind.astype(np.int64)
    ind0 = ind.copy()

    # ---------- 6. 个体参数 ----------
    save_rate = np.clip(cfg["save_base"] +
                        cfg["save_per_conscientious"] * traits0[:, ti["自制力"]],
                        cfg["save_min"], cfg["save_max"])
    invest_frac = np.clip(cfg["invest_base"] +
                          cfg["invest_per_judgment"] * traits0[:, ti["判断力"]] +
                          cfg["invest_per_family"] * np.log1p(family),
                          cfg["invest_min"], cfg["invest_max"])
    ret_mu = cfg["invest_ret_base"] + cfg["invest_ret_per_judgment"] * traits0[:, ti["判断力"]]
    blow_prob = np.clip(cfg["blowup_prob_base"] -
                        cfg["blowup_prob_per_judgment"] * traits0[:, ti["判断力"]],
                        cfg["blowup_prob_min"], cfg["blowup_prob_max"])
    meta_val = traits0[:, meta_idx]
    alpha = np.clip(0.30 + 0.12 * meta_val, 0.10, 0.65)
    switch_cost_p = cfg["switch_cost"] * np.exp(-0.5 * meta_val)

    # ---------- 7. 状态 + 轨迹记录 ----------
    wealth = np.tile(family, (W, 1)).astype(float)
    net_left = np.tile(family * cfg["safety_net_multiple"], (W, 1)).astype(float)
    bankrupt = np.zeros((W, P), dtype=bool)
    total_income = np.zeros((W, P), dtype=float)
    last_switch = np.full((W, P), -999, dtype=int)

    wealth_hist = np.zeros((Y, W, P), dtype=np.float32)
    income_hist = np.zeros((Y, W, P), dtype=np.float32)
    industry_hist = np.zeros((Y, W, P), dtype=np.int8)
    bankrupt_hist = np.zeros((Y, W, P), dtype=bool)
    switch_hist = np.zeros((Y, W, P), dtype=bool)
    net_left_hist = np.zeros((Y, W, P), dtype=np.float32)

    rows = np.arange(W)[:, None]

    # ---------- 8. 逐年演化 ----------
    for y in range(Y):
        rel_y = rel_bo[:, y, :]
        cur_rel = rel_y[rows, ind]

        # 各行业当前人数比例（用于拥挤效应）
        counts = np.zeros((W, n_ind))
        for w in range(W):
            counts[w] = np.bincount(ind[w], minlength=n_ind)
        share = counts / P

        # 8.1 换行业（拥挤惩罚 + Gumbel 随机性）
        allowed = (y - last_switch) >= cfg["switch_cooldown"]
        want = (cur_rel < cfg["switch_threshold"]) & allowed & (~bankrupt)
        if want.any():
            base_scores = rel_y - cfg["congestion_factor"] * share
            gumbel_noise = rng.gumbel(size=(W, P, n_ind))
            scores = base_scores[:, None, :] + gumbel_noise
            best_ind = np.argmax(scores, axis=2)
            wealth = wealth - np.where(want, switch_cost_p[None, :], 0.0)
            ind = np.where(want, best_ind, ind)
            last_switch = np.where(want, y, last_switch)
            cur_rel = rel_y[rows, ind]

        # 8.2 行业系数（含拥挤惩罚）
        person_share = share[rows, ind]
        congestion_penalty = 1.0 - cfg["congestion_factor"] * person_share

        ind_factor = np.where(
            hedge_flag[None, :],
            alpha[None, :] + (1.0 - alpha[None, :]) * cur_rel,
            cur_rel
        ) * congestion_penalty

        # 8.3 运气因子
        luck = rng.lognormal(-0.5 * cfg["luck_sigma"] ** 2,
                             cfg["luck_sigma"], (W, P))
        good = rng.random((W, P)) < cfg["good_event_prob"]
        luck = np.where(good,
                        luck * rng.lognormal(cfg["good_event_mu"],
                                             cfg["good_event_sigma"], (W, P)),
                        luck)
        bad = rng.random((W, P)) < cfg["bad_event_prob"]
        luck = np.where(bad,
                        luck * rng.uniform(cfg["bad_event_lo"],
                                           cfg["bad_event_hi"], (W, P)),
                        luck)

        # 8.4 年收入
        income = cfg["base_income"] * trait_coef[y][None, :] * ind_factor * luck
        income = np.where(bankrupt, cfg["min_income"], income)
        total_income += income

        # 8.5 储蓄
        wealth += income * save_rate[None, :] * (~bankrupt)

        # 8.6 投资
        investable = np.maximum(wealth, 0.0) * invest_frac[None, :]
        investable = np.where(bankrupt, 0.0, investable)
        ret = rng.normal(ret_mu[None, :], cfg["invest_ret_sigma"], (W, P))
        blow = rng.random((W, P)) < blow_prob[None, :]
        ret = np.where(blow, -1.0, ret)
        wealth += investable * ret

        # 8.7 all in 的行业风险：行业不景气时额外扣除收入
        all_in_mask = ~hedge_flag[None, :]
        risk_condition = (cur_rel < cfg["all_in_risk_threshold"]) & all_in_mask
        wealth -= np.where(risk_condition, cfg["all_in_risk_factor"] * income, 0.0)

        # 8.8 安全垫 / 破产
        neg = wealth < 0
        cover = np.where(neg, np.minimum(-wealth, net_left), 0.0)
        wealth += cover
        net_left = np.maximum(net_left - cover, 0.0)
        newly = (wealth < 0) & (~bankrupt)
        bankrupt |= newly
        wealth = np.where(bankrupt, 0.0, wealth)

        # 8.9 记录轨迹
        if return_trajectories:
            wealth_hist[y] = wealth
            income_hist[y] = income
            industry_hist[y] = ind
            bankrupt_hist[y] = bankrupt
            switch_hist[y] = want
            net_left_hist[y] = net_left

    # ---------- 9. 每年排名 ----------
    if return_trajectories:
        order = np.argsort(-wealth_hist, axis=2, kind="stable")
        rank_hist = np.empty((Y, W, P), dtype=np.int16)
        ranks_full = np.empty((Y, W, P), dtype=np.int16)
        ranks_full[...] = np.arange(1, P + 1, dtype=np.int16)
        np.put_along_axis(rank_hist, order, ranks_full, axis=2)
    else:
        rank_hist = None

    # ---------- 10. 汇总统计 ----------
    final_w = wealth_hist[-1] if return_trajectories else wealth

    order_final = np.argsort(-final_w, axis=1)
    is_richest = np.zeros((W, P), dtype=bool)
    np.put_along_axis(is_richest, order_final[:, :1], True, axis=1)
    richest_count = is_richest.sum(axis=0)

    poor_final = bankrupt | (final_w <= 0)
    poor_count = poor_final.sum(axis=0)

    rank_desc = np.argsort(np.argsort(-final_w, axis=1), axis=1)
    avg_final_rank = rank_desc.mean(axis=0) + 1.0

    data = {
        "编号": np.arange(P),
        "家底": family,
        "学校成绩": school_score,
        "学校排名": school_rank,
    }
    for i, n in enumerate(traits_names):
        data[f"初始_{n}"] = traits0[:, i]
    data["神级词条"] = [traits_names[g] if g >= 0 else "无" for g in god_trait]
    data["洗词条策略"] = [wash_names[s] for s in wash_strategy]
    data["对冲策略"] = np.where(hedge_flag, "对冲", "all in")
    for i, ind_name in enumerate(industries):
        data[f"初始行业_{ind_name}比例"] = (ind0 == i).mean(axis=0)

    data["平均终身总收入"] = total_income.mean(axis=0)
    data["终身总收入中位数"] = np.median(total_income, axis=0)
    data["平均第40年财富"] = final_w.mean(axis=0)
    data["第40年财富中位数"] = np.median(final_w, axis=0)
    data["当首富次数"] = richest_count
    data["变成穷光蛋次数"] = poor_count
    data["破产出局次数"] = bankrupt.sum(axis=0)
    data["平均最终财富排名"] = avg_final_rank
    data["破产率"] = poor_count / float(W)

    df = pd.DataFrame(data)

    res = {
        "df": df,
        "cfg": cfg,
        "ind0": ind0,
        "bankrupt": bankrupt,
        "final_wealth": final_w,
        "hedge_flag": hedge_flag,
        "wash_strategy": wash_strategy,
        "wash_names": wash_names,
        "meta_name": meta_name,
        "meta_idx": meta_idx,
        "family": family,
        "traits0": traits0,
        "traits_names": traits_names,
        "industries": industries,
        "traits_hist": traits_y,
        "god_trait": god_trait,
    }
    if return_trajectories:
        res.update({
            "wealth_hist": wealth_hist,
            "income_hist": income_hist,
            "industry_hist": industry_hist,
            "bankrupt_hist": bankrupt_hist,
            "switch_hist": switch_hist,
            "rank_hist": rank_hist,
            "net_left_hist": net_left_hist,
        })
    return res


# ============================================================================
# ==============================  神级加成校准  ==============================
# ============================================================================

def calibrate_god_bonus(cfg, max_iter=15, target_percentile=0.20, calib_worlds=200):
    """
    逐步加大 god_income_bonus，使所有神级者的平均财富排名进入前 target_percentile。
    校准用较少的世界数（calib_worlds），返回校准后的 cfg 副本。
    """
    print("\n" + "=" * 78)
    print(f"【神级加成校准】目标：神级者平均财富排名进入前 "
          f"{target_percentile*100:.0f}%（<= {cfg['n_people']*target_percentile:.0f} 名）")
    print("=" * 78)

    cfg = dict(cfg)
    cfg["n_worlds"] = min(cfg["n_worlds"], calib_worlds)
    cfg["export_traits_hist"] = False
    cfg["export_csv_worlds"] = ()

    P = cfg["n_people"]
    target_rank = P * target_percentile

    for trial in range(max_iter):
        res = simulate(cfg, return_trajectories=False, verbose=False)
        df = res["df"]
        god_mask = df["神级词条"] != "无"
        if not god_mask.any():
            print("[神级校准] 本次没有神级者，跳过校准。")
            return cfg
        avg_rank = df.loc[god_mask, "平均最终财富排名"].mean()
        avg_wealth = df.loc[god_mask, "平均第40年财富"].mean()
        all_avg = df["平均第40年财富"].mean()
        print(f"[神级校准] 试验 {trial+1:2d}: god_income_bonus="
              f"{cfg['god_income_bonus']:.3f}, 神级者平均排名={avg_rank:6.2f} "
              f"(目标<= {target_rank:.2f}), 神级者平均财富={avg_wealth:.2f} "
              f"(全体 {all_avg:.2f})")
        if avg_rank <= target_rank:
            print(f"[神级校准] 达标。最终 god_income_bonus = "
                  f"{cfg['god_income_bonus']:.3f}")
            print(f"           （即神级者年收入额外 ×"
                  f"{np.exp(cfg['god_income_bonus']):.3f}）")
            return cfg
        cfg["god_income_bonus"] += 0.20

    print(f"[神级校准] 达到最大迭代次数，最终 god_income_bonus = "
          f"{cfg['god_income_bonus']:.3f}（仍未达标，可加大 max_iter 或步长）")
    return cfg


# ============================================================================
# ==============================  反事实分析  ================================
# ============================================================================

def counterfactual_analysis(cfg):
    """
    对每个人、每种（洗词条策略 × 对冲策略）组合，在同一组随机种子下各跑一遍。
    输出同一个人换了策略后，财富中位数、首富次数、破产率的变化。
    """
    print("\n" + "=" * 78)
    print("【反事实实验】同一个人 × 8 种策略组合 × 同一组世界")
    print("=" * 78)

    wash_names = ["专洗元能力", "随机洗", "专洗无用词条", "不洗"]
    hedge_names = ["all in", "对冲"]
    P = cfg["n_people"]

    records = []
    wash_improvements = {w: [] for w in wash_names}
    civil_breakdown = []  # 起始行业为土木的子样本

    for wi, wash_name in enumerate(wash_names):
        for hi, hedge_name in enumerate(hedge_names):
            wash_arr = np.full(P, wi, dtype=int)
            hedge_arr = np.full(P, hi == 1, dtype=bool)
            res = simulate(cfg, wash_strategy=wash_arr, hedge_flag=hedge_arr,
                           return_trajectories=False, verbose=False)
            df = res["df"]
            for p in range(P):
                records.append({
                    "person_id": p,
                    "wash_strategy": wash_name,
                    "hedge_strategy": hedge_name,
                    "中位财富": df.loc[p, "第40年财富中位数"],
                    "首富次数": df.loc[p, "当首富次数"],
                    "破产率": df.loc[p, "破产率"],
                    "平均最终排名": df.loc[p, "平均最终财富排名"],
                })

            # 每种洗词条策略下的有效词条总提升量
            th = res["traits_hist"]
            traits_names = res["traits_names"]
            useless_i = traits_names.index(cfg["useless_trait"])
            valid_mask = np.ones(len(traits_names), dtype=bool)
            valid_mask[useless_i] = False
            total_gain = (th[-1, :, valid_mask] - th[0, :, valid_mask]).sum(axis=1)
            wash_improvements[wash_name].append(total_gain.mean())

            # 起始行业为土木的子样本
            ind0 = res["ind0"]                                # (W, P)
            civil_mask = (ind0 == cfg["civil_idx"])           # (W, P)
            if civil_mask.any():
                cb = res["bankrupt"][civil_mask]              # 破产标记
                cw = res["final_wealth"][civil_mask]
                civil_breakdown.append({
                    "wash_strategy": wash_name,
                    "hedge_strategy": hedge_name,
                    "样本数": int(civil_mask.sum()),
                    "平均第40年财富": float(cw.mean()),
                    "破产率": float(cb.mean()),
                })

    cf_df = pd.DataFrame(records)

    base = cf_df[(cf_df["wash_strategy"] == "不洗") & (cf_df["hedge_strategy"] == "all in")]
    base = base.set_index("person_id")[["中位财富", "首富次数", "破产率"]]
    base.columns = ["基准中位财富", "基准首富次数", "基准破产率"]

    cf_df = cf_df.merge(base, left_on="person_id", right_index=True, how="left")
    cf_df["中位财富变化"] = cf_df["中位财富"] - cf_df["基准中位财富"]
    cf_df["首富次数变化"] = cf_df["首富次数"] - cf_df["基准首富次数"]
    cf_df["破产率变化"] = cf_df["破产率"] - cf_df["基准破产率"]

    out = os.path.join(os.getcwd(), "counterfactual_summary.csv")
    cf_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[文件] 反事实汇总已保存：{out}")

    print("\n各策略组合相对于「不洗 + all in」的平均变化（跨所有人）：")
    summary = cf_df.groupby(["wash_strategy", "hedge_strategy"]).agg(
        平均中位财富变化=("中位财富变化", "mean"),
        平均首富次数变化=("首富次数变化", "mean"),
        平均破产率变化=("破产率变化", "mean"),
    ).reset_index()
    with pd.option_context("display.float_format", lambda x: f"{x:+.4f}"):
        print(summary.to_string(index=False))

    print("\n每种洗词条策略在 40 年里的有效词条总提升量（平均）：")
    for w in wash_names:
        print(f"  {w}: {np.mean(wash_improvements[w]):+.4f}")

    imp_df = pd.DataFrame({
        "洗词条策略": wash_names,
        "有效词条总提升量": [np.mean(wash_improvements[w]) for w in wash_names],
    })
    imp_path = os.path.join(os.getcwd(), "wash_improvement.csv")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
    print(f"[文件] 洗词条提升量已保存：{imp_path}")

    # 单独输出：起始行业为土木的人里，all in 和 对冲 的破产率对比
    if civil_breakdown:
        cbd = pd.DataFrame(civil_breakdown)
        # 只取"洗词条策略=不洗"的行，避免重复计数；若无则聚合全部
        sub = cbd[cbd["wash_strategy"] == "不洗"]
        if sub.empty:
            sub = cbd.groupby(["hedge_strategy"]).agg(
                样本数=("样本数", "sum"),
                平均第40年财富=("平均第40年财富", "mean"),
                破产率=("破产率", "mean"),
            ).reset_index()
        print("\n起始行业为「土木」的人：all in vs 对冲 的破产率对比")
        with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
            print(sub[["hedge_strategy", "样本数", "平均第40年财富", "破产率"]]
                  .to_string(index=False))

    return cf_df


# ============================================================================
# ==============================  报告输出  ==================================
# ============================================================================

def report(res):
    cfg = res["cfg"]
    df = res["df"]
    W = cfg["n_worlds"]
    wash_names = res["wash_names"]
    traits_names = res["traits_names"]
    industries = res["industries"]

    print("\n" + "=" * 78)
    print("【一】每人一行汇总表（前 20 行）")
    print("=" * 78)
    show_cols = (["编号", "家底", "学校排名", "神级词条", "洗词条策略", "对冲策略"] +
                 [f"初始行业_{n}比例" for n in industries] +
                 ["平均终身总收入", "第40年财富中位数", "当首富次数", "变成穷光蛋次数"])
    with pd.option_context("display.max_columns", 40, "display.width", 220,
                           "display.float_format", lambda x: f"{x:,.3f}"):
        print(df[show_cols].head(20).to_string(index=False))

    print("\n" + "=" * 78)
    print("【二】统计结论")
    print("=" * 78)

    sp_school = stats.spearmanr(df["学校排名"], df["平均最终财富排名"])[0]
    print(f"\n① 学校排名 与 最终财富排名 的 Spearman 相关系数：{sp_school:+.3f}")

    print("\n② 家底 / 各词条 与 最终财富（平均第40年财富）的 Pearson 相关系数：")
    corr_rows = []
    for col in ["家底"] + [f"初始_{n}" for n in traits_names]:
        r = np.corrcoef(df[col].values, df["平均第40年财富"].values)[0, 1]
        corr_rows.append((col.replace("初始_", ""), r))
    corr_df = pd.DataFrame(corr_rows, columns=["变量", "与最终财富相关系数"])
    corr_df["|r|"] = corr_df["与最终财富相关系数"].abs()
    corr_df = corr_df.sort_values("|r|", ascending=False)
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"   （元能力词条 = {res['meta_name']}）")

    print("\n③ 四种「洗词条策略」对比（按人平均）：")
    g1 = df.groupby("洗词条策略").agg(
        人数=("编号", "count"),
        平均终身总收入=("平均终身总收入", "mean"),
        平均第40年财富=("平均第40年财富", "mean"),
        平均破产率=("破产率", "mean"),
    ).reindex(wash_names)
    print(g1.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\n④ 「all in」 vs 「对冲」对比（按人平均）：")
    g2 = df.groupby("对冲策略").agg(
        人数=("编号", "count"),
        平均终身总收入=("平均终身总收入", "mean"),
        平均第40年财富=("平均第40年财富", "mean"),
        平均破产率=("破产率", "mean"),
        首富次数合计=("当首富次数", "sum"),
    ).reindex(["all in", "对冲"])
    print(g2.to_string(float_format=lambda x: f"{x:,.4f}"))

    print("\n⑤ 初始选择「土木」行业的人：all in vs 对冲（逐世界-人观测）：")
    civil_mask = (res["ind0"] == cfg["civil_idx"])
    if civil_mask.sum() > 0:
        cw = res["final_wealth"][civil_mask]
        ch = res["hedge_flag"][np.where(civil_mask)[1]]
        cb = res["bankrupt"][civil_mask]
        dfc = pd.DataFrame({
            "对冲策略": np.where(ch, "对冲", "all in"),
            "第40年财富": cw,
            "破产": cb,
        })
        g3 = dfc.groupby("对冲策略").agg(
            样本数=("第40年财富", "count"),
            平均第40年财富=("第40年财富", "mean"),
            财富中位数=("第40年财富", "median"),
            破产率=("破产", "mean"),
        ).reindex(["all in", "对冲"])
        print(g3.to_string(float_format=lambda x: f"{x:,.4f}"))
    else:
        print("   （本次模拟中没有人初始选择土木行业）")

    print("\n⑥ 是否有神级词条（按人平均）：")
    df["有神级"] = df["神级词条"] != "无"
    g4 = df.groupby("有神级").agg(
        人数=("编号", "count"),
        平均第40年财富=("平均第40年财富", "mean"),
        平均最终财富排名=("平均最终财富排名", "mean"),
    )
    print(g4.to_string(float_format=lambda x: f"{x:,.4f}"))

    # ⑥-b 神级词条能否抵消其他词条的弱点？
    if df["有神级"].any():
        valid_traits = [n for n in traits_names if n != cfg["useless_trait"]]
        weak_others = np.zeros(len(df), dtype=bool)
        for p in range(len(df)):
            god_name = df.loc[p, "神级词条"]
            others = [n for n in valid_traits if n != god_name]
            weak_others[p] = df.loc[p, [f"初始_{n}" for n in others]].mean() < -0.5
        sub = df[df["有神级"] & weak_others]
        print("\n⑥-b 神级能否抵消其他词条的弱点：")
        if len(sub) > 0:
            print(f"     「其他有效词条平均 < -0.5」但拥有神级词条的人：{len(sub)} 人")
            print(f"     其平均最终财富排名：{sub['平均最终财富排名'].mean():.2f} "
                  f"（全体平均 {df['平均最终财富排名'].mean():.2f}）")
            print(f"     其平均第40年财富：{sub['平均第40年财富'].mean():.3f} "
                  f"（全体平均 {df['平均第40年财富'].mean():.3f}）")
        else:
            print("     本次模拟中没有「其他词条都偏弱但有神级词条」的样本。")

    return df


# ============================================================================
# ==============================  统计图表  ==================================
# ============================================================================

def make_plots(res, outdir):
    cfg = res["cfg"]
    df = res["df"]
    Y = cfg["n_years"]
    wash_names = res["wash_names"]
    traits_names = res["traits_names"]

    wh = np.clip(res["wealth_hist"], 0, None)

    # ---------- 图 1：对数财富标准差随时间 ----------
    log_wh = np.log1p(wh)
    std_curve = log_wh.std(axis=2).mean(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(range(1, Y + 1), std_curve, color="#c0392b", lw=2.2,
            label="对数财富标准差")
    ax.set_xlabel("工作年数")
    ax.set_ylabel("对数财富标准差")
    ax.set_title("差距随时间扩大：同一批人的财富分化（1000个世界平均）")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig1_差距随时间扩大.png"), dpi=150)
    plt.close()

    # ---------- 图 2：学校排名 vs 最终财富排名 ----------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(df["学校排名"], df["平均最终财富排名"], s=38,
               c=df["家底"], cmap="viridis", alpha=0.9, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("学校成绩排名（1 = 最好）")
    ax.set_ylabel("最终财富平均排名（1 = 最富）")
    ax.set_title("学校排名 vs 最终财富排名（点色 = 家底）")
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("家底")
    ax.grid(alpha=0.25)
    sp = stats.spearmanr(df["学校排名"], df["平均最终财富排名"])[0]
    ax.text(0.03, 0.97, f"Spearman r = {sp:+.3f}", transform=ax.transAxes,
            va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig2_学校排名_vs_最终财富.png"), dpi=150)
    plt.close()

    # ---------- 图 3：各策略组财富分布箱线图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    groups_a = [df.loc[df["洗词条策略"] == k, "平均第40年财富"].values for k in wash_names]
    axes[0].boxplot(groups_a, labels=wash_names, showmeans=True)
    axes[0].set_title("不同「洗词条策略」的最终财富分布")
    axes[0].set_ylabel("平均第40年财富")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(alpha=0.25, axis="y")

    hedge_labels = ["all in", "对冲"]
    groups_b = [df.loc[df["对冲策略"] == k, "平均第40年财富"].values for k in hedge_labels]
    axes[1].boxplot(groups_b, labels=hedge_labels, showmeans=True)
    axes[1].set_title("「all in」 vs 「对冲」的最终财富分布")
    axes[1].grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig3_策略组财富分布.png"), dpi=150)
    plt.close()

    # ---------- 图 4：平均财富排名前10 vs 破产次数前10 属性对比 ----------
    top_rich = df.nsmallest(10, "平均最终财富排名")
    top_poor = df.nlargest(10, "变成穷光蛋次数")
    rich_means = [top_rich[f"初始_{n}"].mean() for n in traits_names]
    poor_means = [top_poor[f"初始_{n}"].mean() for n in traits_names]
    all_means = [df[f"初始_{n}"].mean() for n in traits_names]

    x = np.arange(len(traits_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, rich_means, width, label="平均财富排名前10",
           color="#e67e22", edgecolor="k", linewidth=0.5)
    ax.bar(x + width / 2, poor_means, width, label="变穷光蛋次数 TOP10",
           color="#7f8c8d", edgecolor="k", linewidth=0.5)
    ax.plot(x, all_means, "k^--", ms=6, lw=1.2, label="全体平均")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(traits_names, rotation=20)
    ax.set_ylabel("词条初始值（标准分）")
    ax.set_title("平均财富排名前10 与 破产 TOP10 的属性画像对比")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig4_首富与破产属性对比.png"), dpi=150)
    plt.close()

    print("[图] fig1~fig4 已保存。")


# ============================================================================
# ==============================  过程可视化  ================================
# ============================================================================

def make_process_plots(res, outdir, world_id=0):
    cfg = res["cfg"]
    Y = cfg["n_years"]
    P = cfg["n_people"]
    W = cfg["n_worlds"]
    industries = res["industries"]

    if not (0 <= world_id < W):
        world_id = 0

    wealth_w = res["wealth_hist"][:, world_id, :]

    fig, ax = plt.subplots(figsize=(11, 6))
    for p in range(P):
        ax.plot(np.arange(1, Y + 1), wealth_w[:, p], lw=0.7, alpha=0.55)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("工作年数")
    ax.set_ylabel("财富（symlog 轴）")
    ax.set_title(f"世界 #{world_id}：100 人的财富轨迹")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig5_世界{world_id}_财富轨迹.png"), dpi=150)
    plt.close()

    rank_w = res["rank_hist"][:, world_id, :]
    fig, ax = plt.subplots(figsize=(11, 6))
    for p in range(P):
        ax.plot(np.arange(1, Y + 1), rank_w[:, p], lw=0.7, alpha=0.55)
    ax.invert_yaxis()
    ax.set_xlabel("工作年数")
    ax.set_ylabel("财富排名（1 = 最富）")
    ax.set_title(f"世界 #{world_id}：100 人的财富排名演变")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig6_世界{world_id}_排名演变.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 6))
    p10 = np.percentile(res["wealth_hist"], 10, axis=(1, 2))
    p50 = np.percentile(res["wealth_hist"], 50, axis=(1, 2))
    p90 = np.percentile(res["wealth_hist"], 90, axis=(1, 2))
    p99 = np.percentile(res["wealth_hist"], 99, axis=(1, 2))
    ax.fill_between(np.arange(1, Y + 1), p10, p90, alpha=0.20,
                    color="#2471a3", label="P10~P90 区间")
    ax.plot(np.arange(1, Y + 1), p50, lw=2.4, color="#2c3e50", label="中位数 P50")
    ax.plot(np.arange(1, Y + 1), p90, lw=2.0, color="#c0392b", label="P90")
    ax.plot(np.arange(1, Y + 1), p99, lw=1.6, color="#8e44ad", ls="--", label="P99")
    ax.plot(np.arange(1, Y + 1), p10, lw=1.6, color="#7f8c8d", ls=":", label="P10")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("工作年数")
    ax.set_ylabel("财富（symlog 轴）")
    ax.set_title("财富分布随时间的演化（1000 个世界平均）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "fig7_财富分布演化.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 6))
    n_ind = len(industries)
    counts = np.zeros((Y, n_ind))
    for y in range(Y):
        for i in range(n_ind):
            counts[y, i] = (res["industry_hist"][y, world_id, :] == i).sum()
    ax.stackplot(np.arange(1, Y + 1), counts.T, labels=industries, alpha=0.85)
    ax.set_xlabel("工作年数")
    ax.set_ylabel("人数")
    ax.set_title(f"世界 #{world_id}：各行业从业人数演变")
    ax.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig8_世界{world_id}_行业演变.png"), dpi=150)
    plt.close()

    df = res["df"]
    rich_p = int(df.loc[df["当首富次数"].idxmax(), "编号"])
    poor_p = int(df.loc[df["变成穷光蛋次数"].idxmax(), "编号"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, pid, tag in zip(axes, [rich_p, poor_p], ["首富最多", "破产最多"]):
        w_line = res["wealth_hist"][:, world_id, pid]
        i_line = res["income_hist"][:, world_id, pid]
        ax.plot(np.arange(1, Y + 1), w_line, lw=2.0, color="#c0392b", label="财富")
        ax.set_xlabel("工作年数")
        ax.set_ylabel("财富", color="#c0392b")
        ax.tick_params(axis="y", labelcolor="#c0392b")
        ax.set_title(f"世界 #{world_id} · 人物 #{pid}（{tag}）")
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, Y + 1), i_line, lw=1.4, color="#2471a3",
                 ls="--", label="年收入")
        ax2.set_ylabel("年收入", color="#2471a3")
        ax2.tick_params(axis="y", labelcolor="#2471a3")
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"fig9_世界{world_id}_两个代表性人物.png"), dpi=150)
    plt.close()

    print(f"[图] fig5~fig9 过程可视化图已保存（世界 #{world_id}）。")


# ============================================================================
# ==============================  轨迹导出  ==================================
# ============================================================================

def export_trajectories(res, outdir):
    cfg = res["cfg"]
    Y = cfg["n_years"]
    P = cfg["n_people"]
    W = cfg["n_worlds"]
    industries = res["industries"]

    npz_path = os.path.join(outdir, "trajectories.npz")
    save_dict = {
        "wealth": res["wealth_hist"],
        "income": res["income_hist"],
        "industry": res["industry_hist"],
        "bankrupt": res["bankrupt_hist"],
        "switch": res["switch_hist"],
        "rank": res["rank_hist"],
        "net_left": res["net_left_hist"],
        "family": res["family"].astype(np.float32),
        "traits0": res["traits0"].astype(np.float32),
        "hedge_flag": res["hedge_flag"],
        "wash_strategy": res["wash_strategy"],
        "ind0": res["ind0"],
        "god_trait": res["god_trait"],
        "industries": np.array(industries),
        "traits_names": np.array(res["traits_names"]),
        "wash_names": np.array(res["wash_names"]),
        "meta_name": np.array([res["meta_name"]]),
    }
    if cfg.get("export_traits_hist", True):
        save_dict["traits_hist"] = res["traits_hist"]
    np.savez_compressed(npz_path, **save_dict)
    print(f"[导出] 完整轨迹 npz：{npz_path}")

    for w in cfg.get("export_csv_worlds", (0, 1, 2)):
        w = int(w)
        if w < 0 or w >= W:
            continue
        person_id = np.tile(np.arange(P), Y)
        year = np.repeat(np.arange(1, Y + 1), P)
        w_flat = res["wealth_hist"][:, w, :].reshape(-1)
        i_flat = res["income_hist"][:, w, :].reshape(-1)
        ind_flat = res["industry_hist"][:, w, :].reshape(-1)
        b_flat = res["bankrupt_hist"][:, w, :].reshape(-1)
        r_flat = res["rank_hist"][:, w, :].reshape(-1)
        s_flat = res["switch_hist"][:, w, :].reshape(-1)

        df = pd.DataFrame({
            "year": year,
            "person_id": person_id,
            "wealth": w_flat,
            "income": i_flat,
            "industry": [industries[int(i)] for i in ind_flat],
            "bankrupt": b_flat,
            "rank": r_flat,
            "switched_industry": s_flat,
        })
        path = os.path.join(outdir, f"world_{w:04d}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[导出] 世界 {w} 逐帧 CSV：{path}")


# ============================================================================
# ==============================  敏感性测试  ================================
# ============================================================================

def _key_metrics(res):
    df = res["df"]
    sp = stats.spearmanr(df["学校排名"], df["平均最终财富排名"])[0]
    meta_adv = (df.loc[df["洗词条策略"] == "专洗元能力", "平均第40年财富"].mean()
                - df.loc[df["洗词条策略"] == "不洗", "平均第40年财富"].mean())
    hedge_adv = (df.loc[df["对冲策略"] == "对冲", "平均第40年财富"].mean()
                 - df.loc[df["对冲策略"] == "all in", "平均第40年财富"].mean())
    family_corr = np.corrcoef(df["家底"].values, df["平均第40年财富"].values)[0, 1]
    meta_name = res["meta_name"]
    meta_corr = np.corrcoef(df[f"初始_{meta_name}"].values,
                            df["平均第40年财富"].values)[0, 1]
    return {
        "学校-财富Spearman": sp,
        "元能力洗词条收益": meta_adv,
        "对冲相对all_in收益": hedge_adv,
        "家底-财富相关": family_corr,
        f"元能力({meta_name})-财富相关": meta_corr,
    }


def sensitivity_test(base_cfg, n_worlds_small=200):
    print("\n" + "=" * 78)
    print("【三】敏感性测试（关键参数调高/调低，看主要结论会不会翻转）")
    print(f"      敏感性测试使用 {n_worlds_small} 个世界（主实验为 {base_cfg['n_worlds']} 个）")
    print("=" * 78)

    social_weights_equal = dict(base_cfg["social_weights"])
    social_weights_equal["自制力"] = 0.10

    variants = [
        ("基准", {}),
        ("相关性 低 (corr_scale=0.4)", {"corr_scale": 0.4}),
        ("相关性 高 (corr_scale=1.8)", {"corr_scale": 1.8}),
        ("溢出系数 低 (0.15)", {"spillover": 0.15}),
        ("溢出系数 高 (0.85)", {"spillover": 0.85}),
        ("神级概率 低 (0.5%)", {"god_trait_prob": 0.005}),
        ("神级概率 高 (8%)", {"god_trait_prob": 0.08}),
        ("安全垫 低 (×1)", {"safety_net_multiple": 1.0}),
        ("安全垫 高 (×8)", {"safety_net_multiple": 8.0}),
        ("反馈延迟 长 (25年)", {"feedback_ramp_years": 25.0}),
        ("反馈延迟 短 (3年)", {"feedback_ramp_years": 3.0}),
        ("自制力权重=0.10（与其他词条相同）", {"social_weights": social_weights_equal}),
    ]

    rows = []
    for name, override in variants:
        cfg = dict(base_cfg)
        cfg.update(override)
        cfg["n_worlds"] = n_worlds_small
        cfg["seed"] = base_cfg["seed"] + 777
        cfg["export_traits_hist"] = False
        cfg["export_csv_worlds"] = ()
        res = simulate(cfg, return_trajectories=False, verbose=False)
        m = _key_metrics(res)
        m["变体"] = name
        rows.append(m)

    sdf = pd.DataFrame(rows).set_index("变体")
    with pd.option_context("display.max_columns", 20, "display.width", 220,
                           "display.float_format", lambda x: f"{x:+.4f}"):
        print(sdf.to_string())

    print("\n[解读] 看各列符号是否随参数变化而翻转：")
    print("  · 学校-财富Spearman 一直很小 → 学校成绩对最终财富的预测力很弱")
    print("  · 元能力洗词条收益 > 0 说明洗元能力比不洗更值钱；若为负则结论翻转")
    print("  · 对冲相对all_in收益 > 0 说明对冲更稳更值钱；若为负则结论翻转")
    print("  · 自制力权重与其他词条相同时，看「元能力洗词条收益」是否仍然成立")


# ============================================================================
# ==============================  main  ======================================
# ============================================================================

def main():
    set_chinese_font()
    outdir = os.getcwd()

    print("=" * 78)
    print("蒙特卡洛模拟（完整轨迹 + 反事实策略对比 + 神级词条校准）")
    print("=" * 78)

    # 1. 先校准神级加成：保证神级者平均财富排名进入前 20%
    calibrated = calibrate_god_bonus(CONFIG, max_iter=15,
                                     target_percentile=0.20, calib_worlds=200)
    CONFIG["god_income_bonus"] = calibrated["god_income_bonus"]

    # 2. 用校准后的参数跑主实验
    res = simulate(CONFIG, return_trajectories=True, verbose=True)

    df = report(res)
    df.to_csv(os.path.join(outdir, "simulation_summary.csv"),
              index=False, encoding="utf-8-sig")
    print(f"\n[文件] 汇总表：simulation_summary.csv")

    # 核对：所有神级者的标签与数值
    print("\n[核对] 神级者标签 vs 数值（来自汇总表）：")
    for _, row in df.iterrows():
        if row["神级词条"] != "无":
            val = row[f"初始_{row['神级词条']}"]
            ok = "OK" if val >= CONFIG["god_trait_bonus"] - 1e-9 else "!!!异常!!!"
            print(f"  #{int(row['编号']):3d}  {row['神级词条']}  = {val:+.4f}  {ok}")

    export_trajectories(res, outdir)
    make_plots(res, outdir)
    make_process_plots(res, outdir, world_id=0)

    # 反事实实验
    counterfactual_analysis(CONFIG)

    # 敏感性测试
    sensitivity_test(CONFIG, n_worlds_small=200)

    print("\n全部完成。")


if __name__ == "__main__":
    main()