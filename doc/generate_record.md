# Generate Record：Env-centric 采集与 Agent-centric 拆分

实现目录：`[generate/](../generate/)`。设计对齐 `[OFFLINE_SEQUENCE.md](OFFLINE_SEQUENCE.md)`，采集风格参考 AnyMDP `gen_anymdp_record.py`（behavior + off-policy reference labeling）。

---

## 1. 目标

两步流水线：

1. **Online 采集（环境中心）**：按 **总** `total_timesteps` 滚动；中途可多次 `reset`。每个 `record-XXXXXX/` 同时保存：
  - **behavior**：实际与环境交互的动作
  - **reference**：同一观测下 oracle RL coach 的动作标签
2. **离线拆分（agent 中心）**：只拆 **友方 ally**；字段分文件，按 timestep 对齐。

`reward_individual` **仅落盘消融**，不改变采集用的 team reward，也不作为 OmniRL 默认 credit（见 `[individual_reward.md](individual_reward.md)` §8）。

---

## 2. Coach root 输入（单路径）

`marl_baseline` / `train_parallel` 每个 task 的训练叶子目录自带 task + RL：

```text
{coach_root}/.../{algorithm}/task-{index:06d}-{task_id}/seed-{seed}/
  task.json                 # mini task bank（tasks 长度 1）
  meta.json                 # task_index / task_id / seed / algorithm
  config.json
  best.safetensors
  final.safetensors
```

Generate record **只需要** `coach_root`：递归发现上述叶子，每个子文件夹对应一个环境。

训练结束（`marl_baseline`，`COACH_EVAL=true` 默认开）会在该叶子 `task.json` 的 `tasks[0].metadata.coach_eval` 写入 `oracle_pure` / `heuristic_advanced` 胜率与 `best_policy`。评测用 `vmap` 并行 episode（默认 `COACH_EVAL_PARALLEL_ENVS=32`）。也可事后跑：

```bash
python tools/compare_oracle_vs_advanced.py --coach_root ... --write_task_json --parallel_envs 32
```

```bash
# 编辑 generate/configs/record_gen.yaml 里的 coach_root，然后：
./generate/generate_records.sh
```

旧版「先 pack 再生成」仍可用：提供 `task_bank` + `ckpt_root`，并在空的 `coach_root`/`task_packages_root` 下自动打包（`python -m generate.pack_task_packages`）。

---



## 3. Step 1：Env-centric

```bash
python -m generate.env_centric \
  --task_packages_root ./ckpt/marl_baseline \
  --output_root ./data/records \
  --total_timesteps 4096 \
  --records_per_task 2 \
  --workers 4 \
  --min_behavior_steps 64 \
  --behavior_switch_prob 0.2
```

（`--task_packages_root` 可直接指向 `coach_root`。）

默认每条 record 用 `time + pid` 采样 seed（写入 `meta.json`）。需要复现时再加 `--seed <int>`。

```bash
# optional reproducibility
python -m generate.env_centric ... --seed 0
```



### 3.1 Behavior / Reference


| 序列                               | 含义                                                                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `actions_behavior`               | 行为策略动作，**进入** `env.step`                                                                                                                                |
| `actions_reference`              | 同一 obs 上硬标签：默认 **oracle argmax**；`best_teacher_reference=true` 时用 `task.json` → `metadata.coach_eval.best_policy`（`oracle_pure` 或 `heuristic_advanced`） |
| `actions_reference_distribution` | **始终** oracle RL 的 soft 分布 `(T, n_ally, A)`，供 KL（HVAC `label_action_distribution`）                                                                      |


Enemy 始终由 `TABXEnemyHeuristicWrapper` 控制（preset 来自 task bank manifest）。

`coach_eval` 由训练结束时的 oracle vs advanced 评测写入（或 `tools/compare_oracle_vs_advanced.py --write_task_json`）。缺字段时开关打开会回退到 `oracle_pure`。

### 3.2 Ally policy 采样模式


| 配置 `independent_ally_policies` | 行为                                                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `false`（默认）                    | 每个 trial 全体 ally **共用**一个 mix 采样；`behavior_policy_id[t, :]` 同行相同                                                                |
| `true`                         | 每个 ally **独立**从同一 `behavior_mix` CDF 采样；落盘 `(T, n_ally)` 可互不相同；**每次 episode reset 后各 ally 单独 resample**；episode 内冷却切换也按 ally 独立 |


两种模式下 `winrate_adapt` 都只重加权共享 CDF：胜率偏低 → 提高强策略权重 → 每个 agent 抽到好策略的概率一起上升（独立模式下表现为「队伍里好策略占比」升高）。

### 3.3 Behavior 切换冷却


| 参数                          | 默认   | 含义                                                     |
| --------------------------- | ---- | ------------------------------------------------------ |
| `mid_episode_policy_switch` | true | 是否允许 trial 内中途换策略；`false` 则只在 episode reset 时 resample |
| `--min_behavior_steps`      | 64   | 至少连续这么多步才允许切换（仅 `mid_episode_policy_switch=true` 时生效）  |
| `--behavior_switch_prob`    | 0.2  | 冷却满足后，每步以该概率重采样（仅中途切换开启时）                              |


默认 **每次 env reset 都会强制重采样** behavior policy。`mid_episode_policy_switch=true` 时，同一 episode 内另受冷却约束；关掉后整局策略固定到下次 reset。

### 3.3.1 按 Task 自适应胜率（可选）

同一套全局 `behavior_mix` 在不同 task 上胜率可能差很多。打开 `winrate_adapt` 后，每个 task 会：

1. 在临时目录用 `adapt_pilot_records` 条 **试跑**估胜率（**不计入**最终输出，试跑后删除）
2. 只重加权 ally mix（**不改 enemy**）：
  - 先调 `strength`（强组质量上限 `adapt_strength_max`，默认 0.7，弱组至少保留 30%）
  - 仍偏低再调 `oracle_focus`，把强组内质量往 **focus teacher** 集中（默认 `oracle_pure`；`best_teacher_reference=true` 时跟 `coach_eval.best_policy`）
  - 强组 = `oracle_pure` / 低ε `oracle_eps` / `advanced`；高ε `oracle_eps0.3` 算弱组
3. Mix CDF 为运行时输入，权重变化不必重新 JIT
4. 找到（或尽力）最终 mix 后，**再生成全部正式 records**（`meta.adapt` 记 strength / oracle_focus / focus_policy / WR / status）


| 配置                       | 默认    | 含义                                            |
| ------------------------ | ----- | --------------------------------------------- |
| `winrate_adapt`          | false | 是否启用                                          |
| `win_rate_min`           | 0.30  | 胜率下限；过低则先抬 strength，再抬 oracle_focus           |
| `win_rate_max`           | null  | 上限；**不设置 / null = 不约束上限**                     |
| `adapt_pilot_records`    | 4     | 每轮试跑 record 条数（临时）                            |
| `adapt_max_iters`        | 3     | 最多试跑/调整轮数                                     |
| `adapt_strength_max`     | 0.7   | 强组质量上限（保留弱策略）                                 |
| `best_teacher_reference` | false | 硬标签与 adapt focus 是否跟 `coach_eval.best_policy` |


与 `independent_ally_policies=true` 兼容：adapt 仍只改共享 CDF，从而抬高每个 ally 抽到强策略的概率。

验收：`python tools/summarize_env_winrate.py --records_root ... --by_task`。若 `strength=strength_max` 且 `oracle_focus=1` 仍低于下限，记 `adapt_failed` 并保留尽力 mix。

### 3.4 总 timestep + 多次 reset

- Record 长度 = `total_timesteps`，不是固定 episode 数。
- Episode 结束 → 手动 `env.reset`，继续追加。
- **必存**边界信号：


| 字段               | 语义                                           |
| ---------------- | -------------------------------------------- |
| `reset.npy`      | 该步是新 episode 的**首步**（reset 后）                |
| `done.npy`       | 该步 transition **结束后** episode 结束             |
| `truncation.npy` | 是否因达到 `max_episode_steps` **超时**终止（非超时终局为 0） |
| `is_win.npy`     | team 0（ally）是否在该步获胜；**非终止步为 0**              |
| `episode_id.npy` | 同 record 内第几个 episode                        |


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
  reset.npy, episode_id.npy
  behavior_policy_id.npy    # (T, n_ally) mix policy_id（目前各 ally 相同）
  policy_tag.npy            # (T, n_ally) 质量序：0…7；agent 转换后 mask 步为 8
  visible_matrix.npy        # (T, N, N)
  obs_flat.npy              # (T, n_ally, obs_dim) 便于第二步拆分
  unit_*.npy                # 全局单位快照
```

`policy_id` **/** `policy_tag` **对照（默认** `behavior_mix`**，已按质量排序，**`policy_id == policy_tag`**）**


| policy_id | name                       | policy_tag |
| --------- | -------------------------- | ---------- |
| 0         | heuristic_random           | 0          |
| 1         | heuristic_novice           | 1          |
| 2         | heuristic_medium_eps0.5    | 2          |
| 3         | heuristic_medium           | 3          |
| 4         | ，heuristic_advanced        | 4          |
| 5         | oracle_eps0.3              | 5          |
| 6         | oracle_eps0.1              | 6          |
| 7         | oracle_pure                | 7          |
| —         | mask（仅 agent-centric 转换写入） | 8          |


`policy_tag` 阶梯：0 random → 1 novice → 2 medium+ε → 3 medium → 4 advanced → 5 oracle_eps高ε → 6 oracle_eps低ε → 7 pure oracle → **8 mask**。  
env 落盘不含 8；`python -m generate.agent_centric` 在 `policy_mask==1` 的步把 `policy_tag` 写成 8。`meta.json` 含 `policy_tag_legend` 与 `policy_id_mapping`。

并行（**生产路径：GPU**）：改配置后一键启动：

```bash
# 编辑 generate/configs/record_gen.yaml（默认 device=gpu），然后：
./generate/generate_records.sh

# 或指定另一份配置：
./generate/generate_records.sh generate/configs/my_run.yaml
```

配置里已包含：`coach_root` / `output_root`、
`total_records` / `total_timesteps`、`device` / `gpu_ids` / `workers_per_gpu`、
`behavior_mix` 等。

- `total_records` 在 tasks 间均分（`// n_tasks`，余数补给前面的 task）。
- 默认 `schedule: task`：**整 task 亲和**——每个 worker 吃完所属 task 的全部 record，进程内复用 env/oracle + warmup，摊薄 JAX JIT；`active_workers = min(slots, n_tasks)`。
- `schedule: record` 为 **legacy**：把 `(task, record_id)` 摊平后 round-robin，易多进程同时冷编译；**GPU 生产勿用**。
- `stagger_s`：worker 启动前 sleep `worker_id * stagger_s`，错开编译高峰。
- GPU：`XLA_PYTHON_CLIENT_PREALLOCATE=false`；建议每卡 `workers_per_gpu` 取 2–4（与旧粘住进程脚本同量级）。
- `device: cpu` 仅本地 debug；**不要**用 CPU 量产 record。
- 默认 `scan_rollout: true`：整条 record 在设备上 `lax.scan`，结束时一次性 `device_get` 落盘（减少逐步 Python/同步）。`false` 回退逐步循环；scan 路径的 behavior 切换用 JAX RNG（`meta.switcher_rng=jax`），与 numpy switcher 序列不必 bit 一致。
- `parallel_envs: B`（需 `scan_rollout`）：对同一 task 用 `vmap` 一次跑 B 个环境；该 task 的 N 条 record 分 `ceil(N/B)` 轮，最后一轮 pad 到 B 以复用同一份 JIT。提高 B 时建议 `workers_per_gpu: 1`，避免多进程 × B 爆显存。
- 目录在 **写完一条 record 落盘时** 再创建（不预建空文件夹）。
- 也可用 `python -m generate.parallel_records --config ...`。

进度监控：

- 终端进度条：`completed/total`、`rec/s`、`ETA`、最近完成的 worker/task/record。
- 时间拆分（每个 task 首次进入 worker 时）：
  - `setup_s`：建 env + 加载 oracle
  - `compile_s`：对 `behavior_mix` **每条 policy** warmup 几步 + reset / reward shaping（不落盘），尽量一次编译完
  - `generate_s`：每条 record 的纯采集时间
- 快照：`{output_root}/generation_progress.json`（含 setup/compile/generate 累计）。
- 明细：`{output_root}/generation_progress.jsonl`（`compile_done` / `record_done` 事件）。
- 结束 summary：`setup` / `compile` / `generate` / `generate_rate`。

旧入口 `python -m generate.env_centric` 仍可用（小规模/调试）。

通信 dump（`attack_target` / teacher `m*`）走同一套 GPU worker，不要用旧的 CPU sidecar：

```bash
./generate/generate_records_comm.sh
# 或
python -m generate.env_centric_comm --config generate/configs/record_gen.yaml
```

`dump_attack_target` 编进 scan JIT（默认 false，旧 schema 不变）。`python -m generate.env_centric_comm` 强制打开。拆分用 `python -m generate.agent_centric_comm`。

---



## 4. Step 2：Agent-centric

推荐：输入顶层 env 目录，输出到**单个扁平文件夹**（不按 task 分层；可用 `--shuffle` 打乱处理顺序；`--workers` 多进程）：

```bash
python -m generate.agent_centric \
  --records_root /path/to/0803_overfit_16task_val \
  --output_root /path/to/0803_overfit_16task_val_agent \
  --mask_prob 0.3 \
  --seed 0 \
  --shuffle \
  --workers 8
```

也支持单条 / 旧嵌套布局（不传 `--output_root` 时写到 `{record}/agent_centric/{ally}/`）：

```bash
python -m generate.agent_centric --records_root ./data/records
python -m generate.agent_centric --record_dir ./data/records/task_00000_xxx/record-000000
```

`policy_mask`**（转换时生成）**：`--mask_prob` 控制（默认 `0.3`）。每个 agent 序列只采样一次；若命中则 **整条序列** 的 `policy_tag` 都设为 **8（mask）**，`policy_mask` 全为 `1`。

**扁平布局**（`--output_root`）：

```text
{output_root}/
  {task_dir}__record-XXXXXX__{ally_key}/
    obs_static.npy
    obs_dynamic.npy
    ...
    policy_tag.npy            # (T,)
    policy_mask.npy           # (T,) 1=masked
    meta.json
```

**嵌套布局**（默认兼容）：

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
  policy_tag.npy
  policy_mask.npy
  meta.json
```

目录名与环境 `ally_keys` 一致（常见为 `unit_00`…，不是 `ally_0`）。同一 agent 下各文件第 `t` 行对齐。`obs_static` = own(14) + zones；`obs_dynamic` = 其他单位槽 (N−1, 16)，`obs_dynamic_mask` 标可见非零槽。训练侧 `discover_agent_centric_dirs` 会优先识别扁平根目录下的 agent 子目录。

> **NaN 说明**：padding 单位曾在 `ParsedState` 里对 `max_health=0` / `attack_cooldown=0` 做除法得到 NaN，再与 visibility 相乘仍为 NaN（`NaN*0=NaN`）。已在 `ParsedState.from_state` 改为安全除法；`env_centric` 落盘时额外 `nan_to_num` 兜底（与 `marl_baseline` 训练侧一致）。

---



## 5. TODO（后续消融）

- [x] **同一时刻不同友军使用不同 behavior policy**：见 `independent_ally_policies`（§3.2）。

---



## 6. 与 credit / OmniRL 的关系

- 采集与 oracle 均基于 **team reward** 训练的 coach。
- `reward_individual` 只供消融，**不要**默认当 OmniRL step credit。
- ICL 适配队友：靠 agent-centric 序列中的队友动态 token / 多 agent 上下文，而不是改 reward（见 `OFFLINE_SEQUENCE.md` latent token 与 credit 讨论）。

---



## 7. 模块索引


| 模块                                 | 作用                                            |
| ---------------------------------- | --------------------------------------------- |
| `generate/pack_task_packages.py`   | （legacy）bank + ckpt → packages                |
| `generate/configs/record_gen.yaml` | 全部生产参数（默认 GPU：`coach_root`、产量、设备、behavior）    |
| `generate/parallel_records.py`     | 均匀分片 + GPU 并行编排（主入口；父进程 JAX-free，避免占 GPU 0）   |
| `generate/record_worker.py`        | spawn worker（先设 CUDA 再 import JAX；进程内 warmup） |
| `generate/progress.py`             | 进度条与 setup/compile/generate 计时                |
| `generate/scan_rollout.py`         | 设备侧整 record `lax.scan`（`scan_rollout: true`）  |
| `generate/generate_records.sh`     | 加载 config 一键启动                                |
| `generate/generate_records_comm.sh` | 同上，强制 `dump_attack_target`（comm schema）        |
| `generate/env_centric.py`          | Step 1 单条采集 + `RecordGenContext`              |
| `generate/env_centric_comm.py`     | comm 入口：复用 `parallel_records`，打开 attack_target |
| `generate/agent_centric.py`        | Step 2 拆分                                     |
| `generate/agent_centric_comm.py`   | Step 2 + teacher `m*` / `visible_ally`          |
| `generate/intent_label.py`         | pack(mode, focus) teacher intent                |
| `generate/behavior_mix.py`         | 加权 mix + 冷却切换                                 |
| `generate/oracle_loader.py`        | 加载 safetensors coach                          |
| `generate/policies.py`             | 共享 ally behavior / reference                  |
| `generate/dump_schema.py`          | 字段与 IO                                        |


