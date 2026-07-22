# Individual Reward Wrapper 使用说明

本文说明如何使用 [`src/tabx/wrappers/individual_reward.py`](../src/tabx/wrappers/individual_reward.py) 为 ally 增加 per-agent shaping，并在 **MAPPO / IPPO** 训练中接入。该模块是独立 wrapper，默认不改动原有 baseline；需要时手动插入环境栈即可做 A/B。

---

## 1. 做什么

在保留环境原始 **team reward**（dense HP 差 + terminal 胜负）的基础上，为每个可控 agent 追加个体项：

```text
r_i = team_coef * r_team
    + damage_coef * norm_damage_i
    + heal_coef   * norm_heal_i
    - damage_taken_coef * norm_damage_taken_i
    - death_coef  * death_i
```

| 项 | 含义 | 归一化 |
|---|---|---|
| `norm_damage` | 本步对敌有效伤害 | `/ target.max_health` |
| `norm_heal` | 本步治疗量（`damage_dealt < 0`） | `/ target.max_health` |
| `norm_damage_taken` | 本步自身 HP 下降 | `/ own.max_health` |
| `death` | 本步由存活变为死亡 | `{0,1}` |

要点：

- **被友方治疗**：接收者 HP 上升 → `norm_damage_taken = 0`，接收者无加成；治疗者通过 `norm_heal` 拿分。
- **kill / assist**：只写入 `info["individual_reward"]`，**不进主 reward**。
- 默认系数偏保守，team 信号仍占主导，便于和纯 team reward 对比。

---

## 2. 默认系数

```python
IndividualRewardConfig(
    enabled=True,
    team_coef=1.0,
    damage_coef=0.05,
    heal_coef=0.05,
    damage_taken_coef=0.03,
    death_coef=0.5,
)
```

A/B 建议：

| 实验 | 设置 |
|---|---|
| A 纯 team | 不接 wrapper，或 `enabled=False` |
| B hybrid（默认） | 使用上表默认系数 |
| C 更强个体 | 适当增大 `damage_coef` / `heal_coef`（如 `0.1`），观察是否伤胜率 |

---

## 3. Wrapper 顺序（重要）

必须接在 **`TABXEnemyHeuristicWrapper` 之后**（此时 reward 已是 per-agent dict），且通常在 **`TABXAutoResetWrapper` 之前**：

```text
TABX
  → (optional) TABXEnemyAllyFlipWrapper
  → TABXLogWrapper
  → TABXEnemyHeuristicWrapper      # 产出 {ally_i: r_team, __all__: r_team}
  → TABXIndividualRewardWrapper    # 改成 hybrid r_i
  → TABXAutoResetWrapper
```

错误顺序：放在 `EnemyHeuristic` 之前时，`step` 返回的是 team 向量而非 dict，wrapper 会报错。

---

## 4. 最小用法示例

```python
from src.tabx import TABX, build_batched_env_params_and_config
from src.tabx.wrappers.wrappers import (
    TABXAutoResetWrapper,
    TABXEnemyHeuristicWrapper,
    TABXLogWrapper,
)
from src.tabx.wrappers.individual_reward import (
    IndividualRewardConfig,
    TABXIndividualRewardWrapper,
)

env_params, tabx_config = build_batched_env_params_and_config(
    scenario_names="elbow",
    heuristic_param_names="medium",
    n_repeat=8,
)
env = TABX(cfg=tabx_config)
env = TABXLogWrapper(env)
env = TABXEnemyHeuristicWrapper(env)
env = TABXIndividualRewardWrapper(
    env,
    IndividualRewardConfig(enabled=True),
)
env = TABXAutoResetWrapper(env)
```

关闭 shaping（等价于原始 team reward）：

```python
env = TABXIndividualRewardWrapper(env, IndividualRewardConfig(enabled=False))
```

`info` 中额外字段：

- `info["individual_reward"]`：各 agent 分项、`kill_log` / `assist_log` 等
- `info["individual_reward_config"]`：当前系数快照

---

## 5. 如何改 MAPPO / IPPO 训练入口

官方脚本：

- MAPPO：[`src/baseline/mappo_rnn.py`](../src/baseline/mappo_rnn.py) 的 `make_train`
- IPPO：[`src/baseline/ippo_rnn.py`](../src/baseline/ippo_rnn.py) 的 `make_train`

两处结构相同：在 `TABXEnemyHeuristicWrapper` 与 `TABXAutoResetWrapper` 之间插入一行即可。

### 5.1 修改 MAPPO（`mappo_rnn.py`）

1. 增加 import：

```python
from src.tabx.wrappers.individual_reward import (
    IndividualRewardConfig,
    TABXIndividualRewardWrapper,
)
```

2. 在 `make_train` 里改训练环境（约 99–101 行附近）：

```python
env = TABXLogWrapper(env)
env = TABXEnemyHeuristicWrapper(env)
env = TABXIndividualRewardWrapper(env, IndividualRewardConfig())  # 新增
env = TABXAutoResetWrapper(env)
```

3. **评估环境**（`eval_env`）一般**不要**加个体 shaping，便于和论文/旧实验比胜率；若也要看 shaping 下的行为，再同样包一层。

### 5.2 修改 IPPO（`ippo_rnn.py`）

与 MAPPO 相同，约 109–111 行：

```python
env = TABXLogWrapper(env)
env = TABXEnemyHeuristicWrapper(env)
env = TABXIndividualRewardWrapper(env, IndividualRewardConfig())  # 新增
env = TABXAutoResetWrapper(env)
```

### 5.3 可选：用 Config 开关做 A/B

在对应脚本的 `Config` 中增加：

```python
USE_INDIVIDUAL_REWARD: bool = False
```

组装环境时：

```python
env = TABXLogWrapper(env)
env = TABXEnemyHeuristicWrapper(env)
if config["USE_INDIVIDUAL_REWARD"]:
    env = TABXIndividualRewardWrapper(env, IndividualRewardConfig())
env = TABXAutoResetWrapper(env)
```

启动示例：

```bash
# 纯 team reward（默认）
python -m src.baseline.mappo_rnn --scenario elbow

# 打开 hybrid individual reward（若已加 USE_INDIVIDUAL_REWARD）
python -m src.baseline.mappo_rnn --scenario elbow --use-individual-reward
```

IPPO 同理：`python -m src.baseline.ippo_rnn ...`

### 5.4 若用 `marl_baseline.py`

文件：[`src/baseline/marl_baseline.py`](../src/baseline/marl_baseline.py) 中 `BaseMARLTrainer.__init__` 组装环境处（约 359–361 行）：

```python
env = TABXLogWrapper(env)
env = TABXEnemyHeuristicWrapper(env)
env = TABXIndividualRewardWrapper(env, IndividualRewardConfig())  # 新增
self.env = TABXAutoResetWrapper(env)
```

`algorithm=mappo` / `ippo` 时都会走同一套环境栈。

---

## 6. 训练与评估建议

1. **主指标仍看胜率 / team return**；个体项只是 shaping。
2. 对比 A/B 时固定 seed、scenario、heuristic、`NUM_ENVS` / `TOTAL_TIMESTEPS`。
3. `TABXLogWrapper` 在 individual wrapper **内侧**时，日志里的 `episode_returns` 仍基于 **team reward**（在 heuristic 广播之后、shaping 之前进入 Log 的路径取决于调用栈）。当前推荐栈下，Log 先收到 team 向量/再经 heuristic；若需要记录 hybrid return，可另行在外侧加日志或读 `info["individual_reward"]`。
4. QMIX / VDN 更假设团队标量再混合；个体 shaping 的 A/B **优先用 MAPPO / IPPO**。

---

## 7. 测试

```bash
python -m unittest tests.test_individual_reward -v
```

覆盖：伤害计入、治疗只给治疗者、承伤/死亡惩罚、kill 仅日志不进 reward。
