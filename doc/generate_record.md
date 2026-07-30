# Generate Record：Env-centric 采集与 Agent-centric 拆分

实现目录：[`generate/`](../generate/)。设计对齐 [`OFFLINE_SEQUENCE.md`](OFFLINE_SEQUENCE.md)，采集风格参考 AnyMDP `gen_anymdp_record.py`（behavior + off-policy reference labeling）。

---

## 1. 目标

两步流水线：

1. **Online 采集（环境中心）**：按 **总 `total_timesteps`** 滚动；中途可多次 `reset`。每个 `record-XXXXXX/` 同时保存：
   - **behavior**：实际与环境交互的动作
   - **reference**：同一观测下 oracle RL coach 的动作标签
2. **离线拆分（agent 中心）**：只拆 **友方 ally**；字段分文件，按 timestep 对齐。

`reward_individual` **仅落盘消融**，不改变采集用的 team reward，也不作为 OmniRL 默认 credit（见 [`individual_reward.md`](individual_reward.md) §8）。

---

## 2. Task Package 输入

每个 task 一个子目录：

```text
{task_packages_root}/task_{index:05d}_{task_id}/
  task.json                 # mini task bank（schema_version=1.0，tasks 长度 1）
  oracle/
    config.json             # 含 algorithm, HIDDEN_SIZE, …
    best.safetensors        # 或 final.safetensors
  meta.json                 # 可选
```

从现有 task bank + `train_parallel` / `marl_baseline` ckpt 树打包：

```bash
python -m generate.pack_task_packages \
  --task_bank task_outputs/tasks_20.json \
  --ckpt_root ./ckpt/marl_baseline \
  --output_root ./data/task_packages \
  --algorithm mappo \
  --seed 0
```

---

## 3. Step 1：Env-centric

```bash
python -m generate.env_centric \
  --task_packages_root ./data/task_packages \
  --output_root ./data/records \
  --total_timesteps 4096 \
  --records_per_task 2 \
  --workers 4 \
  --min_behavior_steps 64 \
  --behavior_switch_prob 0.2
```

默认每条 record 用 `time + pid` 采样 seed（写入 `meta.json`）。需要复现时再加 `--seed <int>`。

```bash
# optional reproducibility
python -m generate.env_centric ... --seed 0
```

### 3.1 Behavior / Reference

| 序列 | 含义 |
|---|---|
| `actions_behavior` | 行为策略动作，**进入** `env.step` |
| `actions_reference` | 同一 obs 上 **纯 oracle** 硬标签（argmax） |
| `actions_reference_distribution` | 同一 obs 上 oracle RL 的 soft 分布 `(T, n_ally, A)`，供 KL（HVAC `label_action_distribution`） |

Enemy 始终由 `TABXEnemyHeuristicWrapper` 控制（preset 来自 task bank manifest）。

### 3.2 同刻全体 ally 共用同一 policy（已实现）

首版：`t` 时刻所有 ally 的 `behavior_policy_id[t]` **相同**；共享一个 behavior 实例出各自动作。

### 3.3 Behavior 切换冷却

| 参数 | 默认 | 含义 |
|---|---|---|
| `--min_behavior_steps` | 64 | 至少连续这么多步才允许切换 |
| `--behavior_switch_prob` | 0.2 | 冷却满足后，每步以该概率重采样 |

默认 **reset 不强制换 policy**（仍受冷却约束）。

### 3.4 总 timestep + 多次 reset

- Record 长度 = `total_timesteps`，不是固定 episode 数。
- Episode 结束 → 手动 `env.reset`，继续追加。
- **必存**边界信号：

| 字段 | 语义 |
|---|---|
| `reset.npy` | 该步是新 episode 的**首步**（reset 后） |
| `done.npy` | 该步 transition **结束后** episode 结束 |
| `truncation.npy` | 是否因达到 `max_episode_steps` **超时**终止（非超时终局为 0） |
| `is_win.npy` | team 0（ally）是否在该步获胜；**非终止步为 0** |
| `episode_id.npy` | 同 record 内第几个 episode |

说明：`done=1` 且 `truncation=0` 多为正常胜负终局；`done=1` 且 `truncation=1` 为超时（超时仍可能按 HP 判 `is_win`）。

### 3.5 输出布局

```text
{output_root}/task_{index:05d}_{task_id}/record-{id:06d}/
  meta.json
  actions_behavior.npy      # (T, n_ally)
  actions_reference.npy     # (T, n_ally) hard argmax
  actions_reference_distribution.npy  # (T, n_ally, A) oracle softmax
  reward_team.npy           # (T, n_ally) team 广播标量
  reward_individual.npy     # (T, n_ally) 消融用 hybrid
  done.npy, truncation.npy, is_win.npy
  reset.npy, episode_id.npy, behavior_policy_id.npy
  visible_matrix.npy        # (T, N, N)
  obs_flat.npy              # (T, n_ally, obs_dim) 便于第二步拆分
  unit_*.npy                # 全局单位快照
```

并行：改配置后一键启动（均匀分 task、CPU/GPU）：

```bash
# 编辑 generate/configs/record_gen.yaml，然后：
./generate/generate_records.sh

# 或指定另一份配置：
./generate/generate_records.sh generate/configs/my_run.yaml
```

配置里已包含：`task_bank` / `ckpt_root` / `task_packages_root` / `output_root`、
`total_records` / `total_timesteps`、`device` / `gpu_ids` / `workers_per_gpu`、
`behavior_mix` 等。

- `total_records` 在 tasks 间均分（100 task × 1000 record → 每 task 10 条）。
- Worker 按 `record_id % n_workers` 分片（worker0: `record-000000,000004,…`）。
- 目录在 **写完一条 record 落盘时** 再创建（不预建空文件夹）。
- 也可用 `python -m generate.parallel_records --config ...`。

旧入口 `python -m generate.env_centric` 仍可用；子进程默认 `JAX_PLATFORMS=cpu`。

---

## 4. Step 2：Agent-centric

```bash
python -m generate.agent_centric --records_root ./data/records
# 或
python -m generate.agent_centric --record_dir ./data/records/task_00000_xxx/record-000000
```

```text
record-XXXXXX/agent_centric/{ally_key}/
  obs_static.npy
  obs_dynamic.npy
  obs_dynamic_mask.npy
  behavior_action.npy
  reference_action.npy                 # hard argmax of oracle
  reference_action_distribution.npy    # (T, A) oracle softmax; KL soft target
  reward_team.npy
  reward_individual.npy
  done.npy
  truncation.npy            # 超时终止
  is_win.npy                # team0 胜；非终止步为 0
  reset.npy                 # 透传，提示 episode 跳变
  episode_id.npy
  behavior_policy_id.npy
  meta.json
```

目录名与环境 `ally_keys` 一致（常见为 `unit_00`…，不是 `ally_0`）。同一 agent 下各文件第 `t` 行对齐。`obs_static` = own(14) + zones；`obs_dynamic` = 其他单位槽 (N−1, 16)，`mask` 标可见非零槽。

> **NaN 说明**：padding 单位曾在 `ParsedState` 里对 `max_health=0` / `attack_cooldown=0` 做除法得到 NaN，再与 visibility 相乘仍为 NaN（`NaN*0=NaN`）。已在 `ParsedState.from_state` 改为安全除法；`env_centric` 落盘时额外 `nan_to_num` 兜底（与 `marl_baseline` 训练侧一致）。

---

## 5. TODO（未实现，后续消融）

- [ ] **同一时刻不同友军使用不同 behavior policy**（per-ally 异构 mix）。当前强制共享，便于先跑通 ICL 蒸馏与分布覆盖；异构混部需另设开关与测试。

---

## 6. 与 credit / OmniRL 的关系

- 采集与 oracle 均基于 **team reward** 训练的 coach。
- `reward_individual` 只供消融，**不要**默认当 OmniRL step credit。
- ICL 适配队友：靠 agent-centric 序列中的队友动态 token / 多 agent 上下文，而不是改 reward（见 `OFFLINE_SEQUENCE.md` latent token 与 credit 讨论）。

---

## 7. 模块索引

| 模块 | 作用 |
|---|---|
| `generate/pack_task_packages.py` | bank + ckpt → task packages |
| `generate/configs/record_gen.yaml` | 全部生产参数（路径、产量、设备、behavior 比例） |
| `generate/parallel_records.py` | 均匀分片 + CPU/GPU 并行编排 |
| `generate/record_worker.py` | spawn worker（先设 CUDA 再 import JAX） |
| `generate/generate_records.sh` | 加载 config 一键启动 |
| `generate/env_centric.py` | Step 1 单条/旧并行采集 |
| `generate/agent_centric.py` | Step 2 拆分 |
| `generate/behavior_mix.py` | 加权 mix + 冷却切换 |
| `generate/oracle_loader.py` | 加载 safetensors coach |
| `generate/policies.py` | 共享 ally behavior / reference |
| `generate/dump_schema.py` | 字段与 IO |
