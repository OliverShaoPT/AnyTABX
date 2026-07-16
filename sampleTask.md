# 采集任务说明手册

本文说明当前仓库里“采集任务”的完整逻辑，以及如何生成、评估、加载和查看任务文件。这里的“采集任务”指的是通过 `src/tabx/sample_task.py` 生成一批固定的 TABX 任务 bank，再用 `src/tabx/eval_task.py` 做一致性评估与筛选。

## 1. 整体流程

当前任务采集是一个闭环流程：

1. 先根据配置采样候选任务。
2. 再对任务做静态平衡、模拟平衡、胜率筛选和去重。
3. 通过后把任务保存成一个单文件 task bank。
4. 需要时再用同一份 task bank 做 heuristic 评估、难度分类或再次过滤。
5. 运行环境时，把 task bank 读取成批量 `env_params` 交给 `TABX.reset()`。

仓库里这些逻辑主要分布在以下文件：

- [src/tabx/sample_task.py](src/tabx/sample_task.py)
- [src/tabx/eval_task.py](src/tabx/eval_task.py)
- [src/tabx/task_generators.py](src/tabx/task_generators.py)
- [src/tabx/sample_task.py](src/tabx/sample_task.py) 中的 `load_task_bank()` / `save_task_bank()` / `build_batched_env_params_from_task_file()`

## 2. 采集任务的逻辑

### 2.1 任务是怎么生成的

任务采样入口是 `SampleConfig`，对应 `src/tabx/sample_task.py` 里的命令行配置。它控制的核心内容包括：

- 任务数量 `n_tasks`
- 随机种子 `seed`
- 是否走程序化生成 `programmatic_ratio`
- 最大队伍人数与最大 zone 数量 `max_n_ally`、`max_n_enemy`、`max_n_zone`
- 是否开启开放式采样 `open_ended_generation`
- 是否启用自由生成、父代 mutation、父代 crossover
- 任务平衡相关阈值、胜率筛选阈值、模拟预算、CPU 核数
- 使用哪个 physics preset 和 heuristic preset

任务候选的构造来源主要是 `src/tabx/task_generators.py`，里面定义了几类结构化偏置：

- 组合 archetype，例如 frontline/backline、assassin、ranged、healer 等
- 布局 archetype，例如 face_off、crossfire、encircle、ambush 等
- zone archetype，例如 void、lava、bush、swamp 以及它们的组合关系
- 距离、分散度、zone 强度等离散/连续采样桶

采样时会把这些维度组合起来，形成一个包含 `scenario` 和 `zone_scenario` 的 task。

### 2.2 采样后做什么筛选

采样并不是直接落盘，而是会经过一系列过滤与修复：

- 静态平衡修复：先用角色和属性估计两队强度，若失衡太大就尝试缩放。
- 模拟平衡修复：会做实际 rollout，检查血量差、伤害分布、截断率、是否完全无交互等。
- 静态 engagement / stomp 过滤：避免一开局就明显不合理或极端碾压的场景。
- 胜率过滤：用 heuristic policy 在双方阵营上测试，保留胜率落在指定区间的任务。
- 去重与 archive 约束：防止 task hash、参数签名或行为签名重复。
- 自适应采样 / Map-Elites：优先保留覆盖度更高、质量更好的候选任务。

这也是为什么采集任务的输出不是简单随机样本，而是“经过平衡与筛选的任务 bank”。

### 2.3 输出文件是什么

`save_task_bank()` 会把结果写成一个单 JSON 文件，结构大致是：

- `schema_version`
- `manifest`
- `tasks`

其中 `manifest` 会记录：

- 生成器来源
- 随机种子
- 任务数
- 物理参数 preset
- heuristic preset
- schema 限制，也就是 `max_n_ally`、`max_n_enemy`、`max_n_zone`
- 如果启用了过滤/采样协议，还会写入 `filter_protocol` 和 `generator_protocol`

每个 task 都保留原本的 `scenario` 和 `zone_scenario`，并在保存时写入统一的 `body_radii` 字段，同时兼容历史别名 `body_radiuss`。

## 3. 怎么生成任务 bank

最直接的方式是运行 `src/tabx/sample_task.py`：

```bash
uv run python src/tabx/sample_task.py --help
```

常用示例：

```bash
uv run python src/tabx/sample_task.py \
  --output sampled_tasks.json \
  --n-tasks 100 \
  --seed 0 \
  --physics default \
  --heuristic expert
```

如果你想减少采样时间，可以先把规模调小：

```bash
uv run python src/tabx/sample_task.py \
  --output sampled_tasks.json \
  --n-tasks 20 \
  --cpu-cores 1 \
  --collection-candidate-budget 200
```

说明：

- `--output` 决定任务 bank 写到哪里。
- `--n-tasks` 是最终需要保留的任务数。
- `--seed` 控制整套采样过程的随机性。
- `--physics` 和 `--heuristic` 决定后续默认使用哪个参数预设。
- `--cpu-cores` 决定并行采样 worker 数量。

采样完成后，脚本会打印类似信息：

```text
Saved 100 tasks to sampled_tasks.json with schema limits (...)
```

## 4. 怎么评估任务 bank

评估入口是 `src/tabx/eval_task.py`。它会对同一批固定任务，用 heuristic policy 在双方阵营上做 rollout，输出胜率、回合长度、伤害、治疗、首次交战步数等统计。

运行方式：

```bash
uv run python src/tabx/eval_task.py --help
```

常用示例：

```bash
uv run python src/tabx/eval_task.py \
  --task-file sampled_tasks.json \
  --output task_eval.json \
  --num-seeds 32 \
  --seed 0
```

你还可以启用这些常见操作：

- `--evaluate-flipped`：把双方队伍互换后再评估一次。
- `--classify-difficulty`：按胜率区间给任务分难度。
- `--filtered-task-output`：把通过一致性筛选的任务再导出成一个新 bank。
- `--difficulty-flip-mode require_consistent`：要求正反双方评估都稳定。

评估输出里一般会包含：

- 原始任务的汇总统计
- 每个任务的胜率区间和分类结果
- 如果开启筛选，还会生成一个新的过滤后 task bank

## 5. 怎么把任务加载进环境

如果你已经有一个 task bank 文件，可以直接载入成批量环境参数：

```python
from src.tabx.sample_task import build_batched_env_params_from_task_file
from src.tabx.tabx import TABX

env_params, cfg = build_batched_env_params_from_task_file("sampled_tasks.json")
env = TABX(cfg=cfg)
obs, state = env.reset(rng, env_params)
```

如果你手里是内存里的任务列表，也可以用：

```python
from src.tabx.sample_task import build_batched_env_params_from_tasks
```

`build_batched_env_params_from_task_file()` 会自动：

- 读取 task bank
- 校验 `schema_version`
- 读取 `manifest` 里的 `physics` 和 `heuristic`
- 按最大队伍数和 zone 数做 padding
- 返回适合 `TABX.reset()` 的 `env_params`

## 6. 任务文件长什么样

一个 task bank 里的单个任务，核心字段通常包括：

- `scenario`
- `zone_scenario`
- `grid_info`
- `task_id`
- `metadata`（如果采样或评估过程中附加了额外信息）

`scenario` 里记录单位位置、朝向、血量、速度、攻击、阵营等；`zone_scenario` 里记录 zone 的位置、椭圆轴、类型与效果值。

如果你想看一个静态场景，可以直接用仓库里的可视化脚本把 task bank 画出来：

```bash
uv run python scripts/visualize_initial_scenes.py sampled_tasks.json --output initial_scenes.png
```

## 7. 推荐的实际操作顺序

1. 先生成小规模 bank，确认采样参数和输出结构。
2. 用 `eval_task.py` 做一次 heuristic 评估，确认任务不是明显偏置或失衡。
3. 再把通过的 bank 接到训练或基准测试里。
4. 如果需要复现，固定 `seed`、`physics`、`heuristic` 和采样阈值。

## 8. 一句话总结

当前采集任务的本质是：通过程序化采样 + 平衡修复 + 胜率筛选，生成一个可复用、可评估、可直接喂给 `TABX.reset()` 的固定任务银行。