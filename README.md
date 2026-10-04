# 同一学校出来的人，为什么出了社会差距越拉越大？
# Why Do People From the Same School Diverge So Much After Graduation?

> 一个用蒙特卡洛模拟回答「出身 vs 天赋 vs 运气 vs 策略」谁在拉开人生差距的 Python 实验。
>
> A Python Monte Carlo experiment asking: **which one actually drives life outcomes — background, talent, luck, or strategy?**

100 个人，同一所学校毕业，40 年职业生涯，1000 个平行世界。
每个人出身和天赋只生成一次，唯一变化的是运气。
看一看：**学校成绩能预测最终财富吗？洗词条真的有用吗？对冲策略值不值？神级天赋能不能翻盘？**

100 people, same school, 40-year careers, 1000 parallel worlds.
Background and talent are generated once. Only luck changes across worlds.
Let's see: **Does school rank predict final wealth? Does "grinding traits" pay off? Is hedging worth it? Can a god-tier talent flip the game?**

这不是一段鸡汤，而是一个可复现、可调参、可诚实输出反例的模拟实验。

This is not a motivational essay. It's a reproducible, tweakable simulation that will honestly report counter-intuitive results.

---

## ✨ 功能特性 · Features

| 中文 | English |
|---|---|
| 🎲 **100 人 × 1000 个平行世界 × 40 年**，向量化实现，普通笔记本几分钟跑完 | 🎲 **100 people × 1000 parallel worlds × 40 years**, vectorized — runs in minutes on a laptop |
| 🧬 **9 个人格词条**（智商 / 学习能力 / 自制力 / 情商 / 判断力 / 批判性思维 / 承担责任 / 抗压 / 无用词条），通过相关矩阵耦合 | 🧬 **9 trait dimensions** (IQ, learning, conscientiousness, EQ, judgment, critical thinking, responsibility, resilience, useless trait), coupled via a correlation matrix |
| 🌟 **神级词条**：2% 概率把某个词条直接拉到 +4σ，并有自动校准机制保证「神级确实能翻盘」 | 🌟 **God-tier trait**: 2% chance to set one trait to +4σ, with an auto-calibration loop ensuring it truly can flip the game |
| 🏫 **学校成绩**只考智商、学习能力、自制力；**社会收入**考全部有效词条，权重还和学校不同 | 🏫 **School score** only tests IQ, learning, and conscientiousness; **social income** uses all valid traits with different weights |
| 💪 **洗词条系统**：每年投入精力提升词条，通过相关网络溢出，四种策略 | 💪 **Trait-grinding system**: annual effort improves a trait, spills over via the correlation network; 4 strategies available |
| 🏗️ **5 个行业**（土木 / 互联网 / 金融 / 制造 / 教育），土木会随机某年崩盘 | 🏗️ **5 industries** (Civil, Internet, Finance, Manufacturing, Education); Civil crashes at a random year |
| 🛡️ **all in vs 对冲**：all in 承担真实破产风险，对冲者可迁移能力部分不受行业影响 | 🛡️ **All-in vs hedge**: all-in takes real bankruptcy risk; hedgers' portable skill part is industry-independent |
| 🧮 **拥挤效应**：越多人挤进同一行业，行业内平均收入越低 | 🧮 **Congestion**: the more people crowd into an industry, the lower the average income |
| 📉 **真实破产机制**：家底提供安全垫，耗尽后跌破 0 就出局 | 📉 **Realistic bankruptcy**: family wealth provides a safety net; once depleted, dropping below 0 ends participation |
| 🔁 **反事实实验**：同一个人，8 种策略组合，同一组随机种子各跑一遍 | 🔁 **Counterfactual design**: for each person, run all 8 strategy combos on the same random seeds |
| 📊 **完整轨迹导出**：每年的财富、收入、行业、排名、破产、换行 | 📊 **Full trajectories exported**: yearly wealth, income, industry, rank, bankruptcy, switches |
| 🔬 **敏感性测试**：把关键参数调高/调低，看结论会不会翻转 | 🔬 **Sensitivity tests**: perturb key parameters and check if conclusions flip |
| 🎨 **中文字体自动配置**，输出 PNG 图表避免乱码 | 🎨 **Auto Chinese font config** for matplotlib to avoid tofu glyphs |

---

## 📦 安装 · Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install numpy pandas matplotlib scipy
```

Python 版本要求：**3.9+**（用到了 `np.random.default_rng`、`np.put_along_axis`）。
Requires Python **3.9+** (uses `np.random.default_rng` and `np.put_along_axis`).

---

## 🚀 快速开始 · Quick Start

```bash
python monte_carlo_simulation.py
```

运行完会得到 / Outputs:

| 文件 / File | 说明 / Description |
|---|---|
| `simulation_summary.csv` | 每人一行汇总 / One row per person |
| `counterfactual_summary.csv` | 反事实：同一个人 8 种策略对比 / 8-strategy counterfactual per person |
| `wash_improvement.csv` | 各洗词条策略的有效词条总提升量 / Total valid-trait gain per grind strategy |
| `trajectories.npz` | 完整轨迹压缩包，做动画用 / Full trajectories, animation-ready |
| `world_0000.csv` ~ `world_0002.csv` | 指定世界的逐帧长表 / Frame-by-frame long tables for chosen worlds |
| `fig1_差距随时间扩大.png` | 对数财富标准差随时间 / Log-wealth std-dev over time |
| `fig2_学校排名_vs_最终财富.png` | 学校排名 vs 最终财富散点图 / School rank vs final wealth |
| `fig3_策略组财富分布.png` | 各策略组财富箱线图 / Wealth boxplots by strategy group |
| `fig4_首富与破产属性对比.png` | 高排名 vs 破产者属性画像 / Attributes of top-ranked vs bankrupt |
| `fig5_世界0_财富轨迹.png` | 单个世界 100 人财富轨迹 / 100 wealth trajectories in one world |
| `fig6_世界0_排名演变.png` | 单个世界 100 人排名演变 / Rank evolution in one world |
| `fig7_财富分布演化.png` | P10/P50/P90/P99 随时间演化 / Percentile bands over time |
| `fig8_世界0_行业演变.png` | 各行业从业人数演变 / Industry headcount over time |
| `fig9_世界0_两个代表性人物.png` | 首富最多 vs 破产最多两人的时间线 / Timelines of the top-winner and top-bankrupt |

控制台打印：学校—财富相关性、家底—财富相关性、各词条—财富相关性、洗词条策略对比、all in vs 对冲对比、土木出身的破产率对比、神级者画像、反事实变化表、敏感性测试表。

Console prints: school-wealth correlation, family-wealth correlation, trait-wealth correlations, grind strategy comparison, all-in vs hedge, Civil-born bankruptcy rates, god-tier profile, counterfactual deltas, sensitivity table.

---

## 🎛️ 配置 · Configuration

所有参数集中在代码顶部的 `CONFIG` 字典里，带中文注释。
All parameters live in the `CONFIG` dict at the top of the script, with Chinese comments.

```python
CONFIG = {
    "n_people": 100,           # 人数 / number of people
    "n_worlds": 1000,          # 平行世界数 / number of parallel worlds
    "n_years": 40,             # 职业生涯年数 / career length
    "seed": 20240501,          # 随机种子 / random seed

    "god_trait_prob": 0.02,    # 神级词条概率 / god-trait probability
    "god_trait_bonus": 4.0,    # 神级词条绝对水平 / god-trait absolute level

    "spillover": 0.50,         # 洗词条的关联溢出系数 / grind spillover
    "safety_net_multiple": 3.0,# 安全垫 = 家底 × 该倍数 / safety net multiplier

    "congestion_factor": 0.50, # 行业拥挤效应 / congestion intensity
    "all_in_risk_factor": 0.30,# all in 行业风险 / all-in industry risk

    "export_csv_worlds": (0, 1, 2),  # 导出哪些世界的 CSV / worlds to export
}
```

### 常见调参 · Common Tweaks

| 想做什么 / Goal | 改哪个参数 / Parameter |
|---|---|
| 跑快一点（先看结构）/ Run faster | `n_worlds=200`, `n_people=50` |
| 换一批人 / 换一组世界 / New batch of people | `seed` |
| 加大运气影响 / More luck | `luck_sigma` → 0.4~0.6 |
| 让家底更重要 / More family wealth | `rich_prob`↑, `safety_net_multiple`↑ |
| 让对冲更有价值 / Hedge more valuable | `all_in_risk_factor`↑ |
| 让洗词条更有用 / Grind more useful | `gain_per_energy`↑, `spillover`↑ |
| 看别的世界 / Watch another world | `export_csv_worlds`, or `make_process_plots(..., world_id=N)` |

---

## 📖 轨迹数据格式 · Trajectory Format

```python
import numpy as np
d = np.load("trajectories.npz")

d["wealth"]        # (Y=40, W=1000, P=100) 每年、每世界、每人的财富
                   # yearly wealth per world per person
d["income"]        # (40, 1000, 100)        年收入 / yearly income
d["industry"]      # (40, 1000, 100)        行业索引 / industry index
d["rank"]          # (40, 1000, 100)        财富排名（1 = 最富）/ wealth rank
d["bankrupt"]      # (40, 1000, 100)        当年是否已破产 / already bankrupt?
d["switch"]        # (40, 1000, 100)        当年是否换行 / switched this year?
d["net_left"]      # (40, 1000, 100)        安全垫剩余 / safety net remaining
d["traits_hist"]   # (40, 100)              词条逐年演变 / yearly trait evolution
d["traits0"]       # (100, 9)               初始词条（神级已赋值）/ initial traits
d["god_trait"]     # (100,)                 神级词条索引，-1 = 无 / god-trait index
d["ind0"]          # (1000, 100)            每人在每个世界的初始行业 / initial industry
d["hedge_flag"]    # (100,)                 True = 对冲 / hedge, False = all in
d["wash_strategy"]# (100,)                 0/1/2/3 = 专洗元能力/随机/无用/不洗
d["industries"]    # ["土木","互联网","金融","制造","教育"]
d["traits_names"]  # 词条名列表 / trait names
d["wash_names"]    # 洗词条策略名 / grind strategy names
d["meta_name"]     # 元能力词条名 / meta-ability trait name
```

**用法示例 / Example**: world 0, person 5, year 10 (index 9):

```python
d["wealth"][9, 0, 5]   # 当年财富 / wealth
d["income"][9, 0, 5]   # 当年收入 / income
d["rank"][9, 0, 5]     # 当年排名 / rank
d["industry"][9, 0, 5] # 当年行业索引 / industry index
```

`world_0000.csv` 是长表（每行 = 一个人一年），字段包括 `year, person_id, wealth, income, industry, bankrupt, rank, switched_industry`，可以直接拖进 `matplotlib.animation` 或 `manim` 做逐帧动画。

`world_0000.csv` is a long table (one row = one person-year) with columns `year, person_id, wealth, income, industry, bankrupt, rank, switched_industry` — ready for `matplotlib.animation` or `manim`.

---

## 🔬 反事实实验设计 · Counterfactual Design

你可能会问：**「专洗元能力」组的人本来就比「不洗」组强怎么办？**
You might ask: **What if the "grind meta" group was already stronger than the "no grind" group?**

我们用一个简单粗暴的办法排除这种干扰：
We eliminate this confound with a straightforward approach:

> 对**同一个人**，分别让他采用 8 种「洗词条策略 × 对冲策略」组合，在**同一组随机种子**下各跑一遍。
>
> For the **same person**, run all 8 (grind × hedge) combos on the **same random seeds**.

这样策略组之间的差异就完全来自策略本身，而不是天赋。代码里的 `counterfactual_analysis()` 负责这件事，输出：
Differences between strategies then come purely from the strategy, not from talent. `counterfactual_analysis()` produces:

- 每个人在 8 种策略下的中位财富、首富次数、破产率
  Per-person median wealth, richest-count, bankruptcy rate under each strategy
- 相对于基准（不洗 + all in）的变化
  Deltas vs. baseline (no-grind + all-in)
- 每种策略组合的平均变化（跨所有人）
  Average deltas per combo

**结论如果和「洗词条有用」「对冲有用」的直觉相反，也会如实输出，不做任何硬编码。**
**If results contradict the intuition that "grinding helps" or "hedging helps," they are reported as-is — no hard-coding.**

---

## 📐 实验设计要点 · Design Notes

### 1. 出身与天赋只在出生时抽一次
### 1. Background and talent drawn once at birth

100 个人的家底、9 个词条、神级词条、洗词条策略、对冲策略都只生成一次。1000 个平行世界里的差异**完全来自运气**。

All 100 people's family wealth, 9 traits, god-tier flag, grind strategy, and hedge strategy are generated once. Cross-world differences come **purely from luck**.

### 2. 词条之间有相关性
### 2. Traits are correlated

相关矩阵写在 `CONFIG["corr_pairs"]` 里，如果不正定会自动修正到最近的正定矩阵（特征值截断 + 重新标准化）。

The correlation matrix lives in `CONFIG["corr_pairs"]`; if non-PD, it's projected to the nearest PD matrix (eigenvalue clipping + renormalization).

「无用词条」和所有词条相关性为 0，用来做安慰剂对照——如果它和财富显著相关，说明代码有 bug。

The "useless trait" has zero correlation with everything — a placebo control. If it ever correlates with wealth, there's a bug.

### 3. 神级词条是绝对水平
### 3. God-tier is an absolute level

被选中的词条直接设为 `max(原值, 4.0)`，语义是「不管原天赋如何，这个维度都高到离谱」。

The chosen trait is set to `max(original, 4.0)` — meaning "no matter your baseline, this dimension is absurdly high."

神级者的年收入额外乘 `exp(god_income_bonus)`。`god_income_bonus` 由 `calibrate_god_bonus()` 自动调整，**目标：神级者平均财富排名进入前 20%**。如果调不上去会如实打印，不硬凑。

God-tier individuals get an extra `exp(god_income_bonus)` multiplier. The value is calibrated by `calibrate_god_bonus()` with target **"god-tier mean wealth rank ≤ top 20%"**. If it can't reach the target, it says so.

### 4. 学校 vs 社会考的不是同一套东西
### 4. School and society test different things

- 学校成绩 = 智商 × 0.45 + 学习能力 × 0.40 + 自制力 × 0.15
  School score = IQ × 0.45 + learning × 0.40 + conscientiousness × 0.15
- 社会收入 = 全部有效词条的加权和（自制力和判断力权重最高）
  Social income = weighted sum of all valid traits (conscientiousness and judgment weighted highest)

也就是说，学校只考 3 个词条，社会考 8 个。这是模拟「学校成绩不预测最终财富」的核心机制。

School tests 3 traits; society uses 8. This is the core mechanism behind "school rank doesn't predict final wealth."

### 5. 反馈延迟
### 5. Feedback delay

词条对收入的影响不是一开始就 100% 发挥，前 12 年逐步从 35% 涨到 100%。模拟「毕业头几年看不出差距，十年后差距才明显」。

Trait effects ramp from 35% to 100% over the first 12 years — "differences aren't visible for years, then suddenly they are."

### 6. 家底 = 初始财富 + 安全垫
### 6. Family wealth = initial capital + safety net

家底既给你一笔起始资金，也给你「跌到 0 以下时家里能兜底」的次数和金额。安全垫耗尽后再跌破 0，就破产出局，之后只能拿最低收入，不能再投资和换行业。

Family wealth provides both a starting capital and a safety net for going below zero. Once exhausted, dropping below zero again means permanent bankruptcy — minimum income only, no more investing or switching.

### 7. 运气是乘法而不是加法
### 7. Luck is multiplicative, not additive

年收入 = 基础收入 × 词条系数 × 行业系数 × 运气因子。

Yearly income = base × trait coefficient × industry coefficient × luck.

运气因子是对数正态 + 小概率大好事件 + 小概率大坏事件（大坏事件左偏）。乘法结构让「好运的人越来越好，坏运的人越来越难翻身」。

Luck is lognormal plus small-probability good events plus small-probability bad events (left-skewed). Multiplicative structure makes winners keep winning and losers struggle to recover.

### 8. 行业选择有历史依赖
### 8. Industry choice is path-dependent

毕业时按当时景气越高的行业越容易被选中（`industry_choice_temp` 控制敏感度），模拟「选的时候看起来没错」。土木崩盘后允许换行，但需要付出转换成本，元能力越高转换成本越低。

At graduation, hotter industries are more likely to be chosen (`industry_choice_temp` controls sensitivity) — "it looked right at the time." After Civil crashes, switching is allowed but costly; higher meta-ability means cheaper switches.

---

## 📊 主要输出结论 · Key Outputs

运行后控制台会打印 / Console prints:

- **①** 学校排名 vs 最终财富排名的 Spearman 相关系数
  Spearman correlation between school rank and final wealth rank
- **②** 家底、各词条与最终财富的 Pearson 相关系数
  Pearson correlations of family wealth and each trait with final wealth
- **③** 四种洗词条策略的平均财富和破产率对比
  Grind strategies: avg wealth and bankruptcy rates
- **④** all in vs 对冲的平均财富、破产率、首富次数对比
  All-in vs hedge: avg wealth, bankruptcy, richest-count
- **⑤** 初始选择土木的人里，all in vs 对冲的破产率对比
  Civil-born: all-in vs hedge bankruptcy rates
- **⑥** 有 / 无神级词条的财富和排名对比，以及「神级能否抵消其他词条弱点」的诊断
  God-tier wealth/rank comparison and "can god-tier offset other weaknesses?"
- **⑦** 反事实实验：同一个人换策略后，财富中位数、首富次数、破产率的变化
  Counterfactual deltas per person per strategy
- **⑧** 每种洗词条策略在 40 年里的有效词条总提升量
  Total valid-trait gain per grind strategy over 40 years
- **⑨** 敏感性测试：关键参数调高/调低后，主要结论会不会翻转
  Sensitivity tests: do conclusions flip under parameter changes?

---

## ⚠️ 已知局限 · Known Limitations

这是一个**玩具模型**，不是社会学研究。它的目的是演示「同一批人 + 不同运气 + 不同策略」下的分化机制，不是在预测真实社会。

This is a **toy model**, not sociology. Its purpose is to demonstrate divergence mechanics under "same people + different luck + different strategies," not to predict real society.

具体局限包括 / Specific limitations:

- 词条之间只有线性相关，没有复杂的非线性交互
  Traits only have linear correlations, no complex non-linear interactions
- 收入结构是乘法的，真实收入曲线要复杂得多
  Multiplicative income, whereas real income curves are much more complex
- 行业景气是随机游走，土木的崩盘是硬编码的单一事件
  Industry booms are random walks; Civil's crash is a single hard-coded event
- 家底只有「初始财富 + 安全垫」两重作用，没有代际传递
  Family wealth only acts as initial capital and safety net; no intergenerational transmission
- 神级词条的校准目标是「平均排名进入前 20%」，这是一个人为设定的目标
  God-tier calibration target ("mean rank ≤ top 20%") is arbitrary
- 破产出局后只能拿最低收入，是简化处理，真实世界还有翻身机会
  Permanent bankruptcy with minimum income is a simplification; reality has second chances

**如果你发现某个结论和你的直觉相反，那正是这个项目最有意思的地方——去看一眼是哪一步的机制导致的，或者改参数验证一下。**

**If a result contradicts your intuition, that's the most interesting part — trace which mechanism causes it, or tweak parameters and verify.**

---

## 🗂️ 项目结构 · Project Structure

```
.
├── monte_carlo_simulation.py    # 主脚本（单文件）/ main script (single file)
├── README.md
├── simulation_summary.csv       # 运行后生成 / generated
├── counterfactual_summary.csv   # 运行后生成 / generated
├── wash_improvement.csv         # 运行后生成 / generated
├── trajectories.npz             # 运行后生成 / generated
├── world_0000.csv ...           # 运行后生成 / generated
└── fig1_*.png ... fig9_*.png    # 运行后生成 / generated
```

代码内部结构 / Internal structure:

| 函数 / Function | 作用 / Purpose |
|---|---|
| `set_chinese_font()` | 配置中文字体 / set Chinese font |
| `build_corr_matrix()` / `nearest_pd_corr()` | 构造 / 修正相关矩阵 / build / fix correlation matrix |
| `gini_per_row()` | 按行算基尼系数 / per-row Gini |
| `simulate(cfg, ...)` | 核心模拟 / core simulation |
| `calibrate_god_bonus()` | 校准神级收入加成 / calibrate god-tier bonus |
| `counterfactual_analysis()` | 反事实实验 / counterfactual experiments |
| `report()` | 打印统计结论 / print stats |
| `make_plots()` | fig1~fig4 / stats plots |
| `make_process_plots()` | fig5~fig9 / process plots |
| `export_trajectories()` | 导出 npz + CSV / export npz + CSV |
| `sensitivity_test()` | 敏感性测试 / sensitivity tests |
| `main()` | 串起来 / orchestrate |

---

## 🎬 用它做可视化视频 · Making Videos

`trajectories.npz` 和 `world_XXXX.csv` 就是为视频设计的。

`trajectories.npz` and `world_XXXX.csv` are designed for animation.

**推荐视频结构 / Suggested video structure:**

1. **开头**：100 个点站在同一条起跑线上（可以用初始家底做散点）
   **Opening**: 100 dots at the starting line (scatter by initial family wealth)
2. **第 1~5 年**：几乎看不出差距，点还挤在一起
   **Years 1–5**: dots barely separate
3. **第 10~20 年**：开始分化，土木崩盘，all in 的人开始掉队
   **Years 10–20**: divergence begins, Civil crashes, all-in players fall behind
4. **第 30~40 年**：头部和尾部的距离达到几个数量级
   **Years 30–40**: top and bottom diverge by orders of magnitude
5. **结尾**：叠加 fig1 的对数财富标准差曲线，说明差距随时间扩大
   **Ending**: overlay fig1's log-wealth std-dev curve

**推荐的 matplotlib 动画骨架 / Animation skeleton:**

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

d = np.load("trajectories.npz")
wealth = d["wealth"]         # (Y, W, P)
Y, W, P = wealth.shape

fig, ax = plt.subplots(figsize=(10, 6))
scat = ax.scatter(np.arange(P), wealth[0, 0, :])
ax.set_ylim(0, np.percentile(wealth, 99))

def update(y):
    scat.set_offsets(np.c_[np.arange(P), wealth[y, 0, :]])
    return scat,

ani = animation.FuncAnimation(fig, update, frames=Y, interval=200, blit=True)
ani.save("wealth_evolution.gif", writer="pillow")
```

---

## 🤝 贡献 · Contributing

欢迎提 Issue 和 PR，尤其是 / Issues and PRs welcome, especially:

- 想加新的机制（比如代际传递、教育回报、行业周期）
  New mechanisms (intergenerational transmission, education returns, industry cycles)
- 发现某个参数下结论翻转了，想讨论
  Cases where conclusions flip under certain parameters
- 用 `trajectories.npz` 做的可视化视频
  Visualizations built on `trajectories.npz`
- 想翻译成其他语言
  Translations

---

## 📄 License

MIT License

---

## 🙏 致谢 · Acknowledgements

灵感来自「蒙特卡洛模拟人生」的经典思想实验。

Inspired by the classic "Monte Carlo simulation of life" thought experiment.

如果你用这个模型做出了有意思的视频或结论，欢迎在 Issue 里分享。

If you make interesting videos or reach interesting conclusions, share them in Issues.
