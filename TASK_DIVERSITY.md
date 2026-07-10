# AnyTABX Task Diversity 调研

## 1. 目标与结论

本文把一个 task 定义为一组可复现的战局参数：

```text
task = 单位编成 + 单位属性 + 初始布局 + zone + 物理参数 + 对手参数 + 终局规则 + seed
```

当前目标是生成一个**离线、固定、可复现**的 task 题库，并以仓库自带的 heuristic AI 作为难度基准。

核心结论：

1. AnyTABX 已经有适合批量生成 task 的数据结构和 JAX 并行能力，但标准环境中的战局骨架主要来自固定 JSON；真正的运行时随机初始化并不多。
2. 现有 UED 能随机化 zone、部分单位属性和 heuristic 参数，但不能随机化单位类型、单位数量、位置和朝向。只依赖现有 UED，生成量很大，行为多样性仍可能不足。
3. “双方 heuristic AI 对战，微小 buff 后能赢、微小 debuff 后会输”是有价值的**临界敏感性测试**，但不应独立作为难度或质量定义。
4. 更可靠的流程是：

```text
候选生成
  → 静态合法性检查
  → 低成本交战性检查
  → heuristic 多 seed 难度评估
  → 阵营/地图翻转检查
  → 行为与参数去重
  → 分层覆盖选择
  → 固化 task + manifest
```

---

## 2. 当前 task 是怎样进入环境的

task 在代码里不是一个独立的 `Task` 类，而是 `env_params`：

```python
env_params = {
    "scenario": VectorizedScenario,
    "zone_scenario": ZoneScenario,
    "physics_params": PhysicsParams,
    "heuristic_params": TABXHeuristicParam,
}
```

主要调用链：

```text
场景名 / physics 名 / heuristic 名
  → build_batched_env_params_and_config()
  → build_batched_scenarios()
  → load_scenario_from_json()
  → padding 到统一 max_n_ally/max_n_enemy/max_n_zone
  → TABX.reset(key, env_params)
  → 单位和 zone 被实例化
```

关键位置：

- task 总装配入口：[`src/tabx/utils.py`](src/tabx/utils.py) 的 `build_batched_env_params_and_config()`
- 场景加载与 padding：[`src/tabx/scenarios/utils.py`](src/tabx/scenarios/utils.py) 的 `load_scenario_from_json()`、`build_batched_scenarios()`、`generate_padded_unit_scenario()`、`generate_padded_zone_scenario()`
- task 数据结构：[`src/tabx/scenarios/scenario.py`](src/tabx/scenarios/scenario.py) 的 `VectorizedScenario`、`ZoneScenario`
- 环境物化和运行：[`src/tabx/tabx.py`](src/tabx/tabx.py) 的 `TABX.reset()`、`TABX.step()`

`build_batched_scenarios()` 会自动取一批场景中的最大单位数和最大 zone 数，再用 `is_disabled=True` 的空槽补齐。因此，在预先固定 `max_n_ally`、`max_n_enemy`、`max_n_zone` 的前提下，可以让不同编成共享同一套 observation/action schema。

---

## 3. 当前已有的多样性元素

### 3.1 场景资产

场景注册在 [`src/tabx/scenarios/constants.py`](src/tabx/scenarios/constants.py)，通过扫描目录自动生成名称列表。

当前共有 42 个 JSON：

| 类型 | 数量 | 位置 | 运行时如何使用 |
|---|---:|---|---|
| 训练单位编成 | 4 | [`src/tabx/scenarios/units/`](src/tabx/scenarios/units/) | 与 zone 名组合 |
| 训练 zone | 5（含 `void`） | [`src/tabx/scenarios/zones/`](src/tabx/scenarios/zones/) | 与单位场景组合 |
| 手工 challenge | 11 | [`src/tabx/scenarios/challenges/`](src/tabx/scenarios/challenges/) | 单位和 zone 一体化 |
| eval 单位编成 | 14 | [`src/tabx/scenarios/eval_scenarios/units/`](src/tabx/scenarios/eval_scenarios/units/) | held-out 评估 |
| eval zone | 8 | [`src/tabx/scenarios/eval_scenarios/zones/`](src/tabx/scenarios/eval_scenarios/zones/) | held-out 评估 |

普通场景名采用 `{单位场景}_{zone 场景}`，例如：

```text
1F1M3A1Hvs2F1S1K1A1H_2L
```

没有 zone 后缀时，`load_scenario_from_json()` 默认加载 `void`。

这里的“组合”是配置组合，不是每次 reset 自动随机抽取。标准 baseline 如果只传入一个 `SCENARIO`，即使并行 128 个环境，也通常只是复制同一个场景。

### 3.2 单位类型

9 种单位定义在 [`src/tabx/units.py`](src/tabx/units.py)，ID 和场景名缩写定义在 [`src/tabx/constants.py`](src/tabx/constants.py)。

| ID | 缩写 | 单位 | HP | 速度 | 攻击/治疗 | 射程 | 冷却 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | F | Farmer | 60 | 1.1 | 14 | 2.5 | 2.5 |
| 2 | S | Assassin | 70 | 1.4 | 22 | 2.5 | 1.5 |
| 3 | K | TheKing | 346 | 1.2 | 46 | 3.2 | 2.5 |
| 4 | M | Mammoth | 685 | 1.2 | 20 | 3.0 | 6.5 |
| 5 | A | Archer | 40 | 1.0 | 28 | 27.0 | 8.0 |
| 6 | C | Cannon | 100 | 0.5 | 80 | 40.0 | 10.0 |
| 7 | D | Deadeye | 40 | 1.1 | 25 | 20.0 | 8.0 |
| 8 | H | Healer | 25 | 1.0 | -7 | 10.0 | 2.0 |
| 9 | P | Paladin | 220 | 1.2 | -6 | 7.5 | 2.0 |

负的 attack damage 表示治疗。除表中属性外，还有 body radius、body weight、sight angle、占用空间等。`get_all_unit_spec()` 返回所有单位模板。

当前状态：

- 单位类型：JSON 中可变，reset 时不随机。
- 双方单位数量：JSON 中可变，reset 时不随机。
- 阵营：`teams` 字段固定为 0/1。
- 单位属性：JSON 可覆盖；UED 只随机化 HP、速度、attack damage。
- 新增单位类型会影响编辑器、ID 映射和 heuristic 角色判断，成本高于只生成新编成。

### 3.3 初始布局

`VectorizedScenario` 中与布局有关的字段包括：

- `positions`
- `rotations`
- `pos_min` / `pos_max`
- `teams`
- `body_radiuss`
- `is_disabled`

这些值通常由 JSON 固定。

[`src/tabx/tabx.py`](src/tabx/tabx.py) 的 `TABX` 支持 `position_permutation`。开启后，reset 会打乱单位槽位与位置之间的对应关系。这是真随机，但要注意：

- 它不会创造新的空间点，只会在已有位置之间换单位。
- 如果交换的是不同单位类型，它可以改变战术关系。
- 如果只是同类单位之间交换，行为多样性几乎为零。
- 槽位顺序变化本身不应被当作新的 task。

目前没有程序化出生点采样器，也没有运行时的单位重叠、最小间距或双方可交战性检查。

### 3.4 Zone / 地形

`ZoneScenario` 定义在 [`src/tabx/scenarios/scenario.py`](src/tabx/scenarios/scenario.py)，包含：

- `zone_type`
- `position`
- `axes`
- `effect_value`
- `n_zone`

zone 行为在 [`src/tabx/tabx.py`](src/tabx/tabx.py) 的 `Zone.act()` 及其分支中：

| type | 类型 | 作用 |
|---:|---|---|
| 0 | empty | 无效果，也用于 padding |
| 1 | lava | 持续伤害 |
| 2 | bush | 改变可见性 |
| 3 | swamp | 减速 |

zone 是椭圆，不是墙体。当前环境不存在不可穿越障碍、路径连通性或资源点系统；地图主要是矩形边界、单位碰撞和效果区域。

### 3.5 地图和物理参数

战场尺寸与单位可活动边界来自场景 JSON。GUI 编辑器位于 [`src/tabx/scenario_editor.py`](src/tabx/scenario_editor.py)，可以编辑单位、zone 和地图。

物理参数由 [`src/tabx/physics/utils.py`](src/tabx/physics/utils.py) 加载：

| preset | `dt` | `percent` | `slop` | `restitution` |
|---|---:|---:|---:|---:|
| `default` | 0.5 | 0.5 | 0.01 | 0.8 |
| `fast` | 1.0 | 0.5 | 0.01 | 0.8 |

它们是可配置项，但当前 UED 不随机化物理参数。

### 3.6 Heuristic 对手

heuristic 主逻辑在 [`src/tabx/heuristic_policy/algorithm.py`](src/tabx/heuristic_policy/algorithm.py) 的 `heuristic_policy()`。参数结构位于 [`src/tabx/heuristic_policy/params.py`](src/tabx/heuristic_policy/params.py)：

- `epsilon`：随机动作概率
- `aggressive_threshold`
- `healer_aggressive_threshold`
- `assasin_speed`
- `ranger_attack_range`

预设位于 [`src/tabx/heuristic_policy/parameters/`](src/tabx/heuristic_policy/parameters/)：

```text
random / novice / medium / advanced / expert
```

[`src/tabx/wrappers/wrappers.py`](src/tabx/wrappers/wrappers.py) 的 `TABXEnemyHeuristicWrapper` 让敌方由 heuristic 控制，我方由外部策略控制。heuristic 中的 `epsilon` 会在每步引入真随机。

### 3.7 动作、目标和终局规则

动作定义在 [`src/tabx/constants.py`](src/tabx/constants.py)：

```text
UP, DOWN, LEFT, RIGHT, ATTACK, TURN_RIGHT, TURN_LEFT, IDLE
```

当前目标固定为击败对方：

- 一方全部死亡/disabled 时结束。
- 达到 `max_episode_steps` 时按团队剩余 HP ratio 判胜。
- HP ratio 平局偏向敌方。
- dense reward 使用双方 HP ratio 的变化。
- terminal reward 默认胜 `+1`、负 `-1`。

目前没有随机任务目标、护送、占点、守时、资源收集等机制。

### 3.8 UED 中已有的真随机

[`src/baseline/ued/level_generator.py`](src/baseline/ued/level_generator.py) 定义：

```python
FREE_PARAM_TYPES = {
    "zone": 0,
    "unit_spec": 1,
    "heuristic_config": 2,
}
```

#### Zone 随机化

`randomize_zone()` 当前范围：

| 字段 | 范围 |
|---|---|
| type | 整数 `[0, 3)` |
| position x | `[15, 45]` |
| position y | `[10, 30]` |
| axis x | `[2, 10]` |
| axis y | `[2, 5]` |
| effect value | `[2, 20]` 后乘类型系数 |

注意：JAX `randint` 的上界不包含在内，因此当前随机生成只会得到 0、1、2，不会得到 type 3 的 swamp。这一点与 `effect_value_coef` 中包含 4 个元素不一致，应在未来实现生成器前确认是否是预期行为。

另一个风险是 `replace()` 中新的 `zone_type` 与 `effect_value_coef[zone_scenario.zone_type]` 的读取时序：右侧读取的是替换前的 `zone_scenario.zone_type`。如果期望 effect coefficient 跟随新类型，需要显式先保存新类型再计算 effect。

#### 单位属性随机化

`randomize_unit_specs()` 当前独立均匀采样：

| 字段 | 范围 |
|---|---|
| health | `[25, 685]` |
| speed | `[0.5, 1.4]` |
| attack damage | `[-7, 80]` |

问题在于三个属性独立采样会破坏单位角色的相关结构。例如一个原本的 healer 可能采样到正伤害；一个高血量单位也可能同时得到最高速度和最高伤害。这会增加数值多样性，但不一定增加高质量的战术多样性。

#### Heuristic 参数随机化

`randomize_heuristic_config()` 随机化：

- `epsilon ∈ [0, 1]`
- `aggressive_threshold ∈ [0, 1]`

对离线题库而言，建议把 heuristic 参数当作**评测协议**固定下来，而不是默认算作 task 内容。否则同一战局只因裁判/对手变弱就会变成不同“难度”的 task，难以解释。

#### 变异和采样

- `mutate_zone()`：噪声、旋转、换类型等。
- `mutate_unit_spec()`：HP、速度、damage 各加绝对值 `[-0.1, 0.1]`。
- `mutate_heuristic_config()`：参数加 `[-0.1, 0.1]`。
- `LevelSampler`：在 [`src/baseline/ued/level_sampler.py`](src/baseline/ued/level_sampler.py) 中保存、评分和重放 level。
- SFL：[`src/baseline/ued/sfl_mappo_rnn.py`](src/baseline/ued/sfl_mappo_rnn.py) 的 `get_learnability_set()` 用 success rate 构造 learnability，并选择高分 level。

`mutate_unit_spec()` 的绝对步长不适合作为通用 buff：对 685 HP 几乎无影响，对 0.5 speed 则相对明显。临界实验应使用相对比例变化。

### 3.9 随机性分类总结

| 元素 | 当前真随机 | 配置可变但默认固定 | 尚未支持 |
|---|---|---|---|
| 场景选择 | UED/训练代码可采样 | 标准 baseline 常为单场景 | 独立 task registry/manifest |
| 单位类型/数量 | 否 | JSON 可变 | 程序化采样 |
| 初始位置/朝向 | 仅可选 permutation | JSON 可变 | 合法布局生成 |
| 单位属性 | UED 的 HP/speed/damage | JSON 可变 | 保持角色相关性的生成 |
| zone | UED 可随机部分字段 | JSON 可变 | 数量及战术关系生成 |
| 地图尺寸 | 否 | JSON/编辑器可变 | 运行时采样 |
| 物理参数 | 否 | `default`/`fast` | UED 随机 |
| heuristic 行为 | `epsilon` 每步随机 | 5 档 preset | 稳定的双方评测 runner |
| 随机目标 | 否 | 歼灭目标固定 | 占点/护送等 |
| 障碍/资源/天气 | 否 | 否 | 未实现 |

---

## 4. 怎样进一步提高 task diversity

“生成尽可能多”不能只看组合数。更重要的是让 task 在策略行为上覆盖不同能力，例如集火、拉扯、绕后、保护治疗、利用草丛、规避熔岩、近战接敌和远程阵地战。

### 4.1 第一层：不改环境 schema 的低成本扩展

> 实现状态：A/B/C/D 已由
> [`src/tabx/task_generators.py`](src/tabx/task_generators.py) 和
> [`src/tabx/sample_task.py`](src/tabx/sample_task.py) 接入离线 task-bank 生成流程。
> 默认使用固定 `max_n_ally=10`、`max_n_enemy=10`、`max_n_zone=4`，并将所选原型和
> 离散分桶记录到每个 task 的 `metadata`。设置 `programmatic_ratio=0` 可退回仅从既有
> JSON 资产采样的兼容模式。

#### A. 扩大单位编成

在固定 `max_n_ally/max_n_enemy` 下采样：

- 双方单位数量。
- 9 种单位的组合。
- 总价格差或总基准强度差。
- 近战/远程/治疗比例。
- 对称、轻度不对称、强 counter 等编成类型。

不建议对所有单位做完全独立均匀采样。可以先定义“编成原型”：

```text
前排 + 后排
前排 + 治疗
刺客群 + 脆皮后排
远程阵地
少量精英 vs 大量低价单位
高机动 vs 高射程
```

再在原型内部采样，这样能显著提高有效战术密度。

**代码位置：**

- 编成原型及角色池：[`src/tabx/task_generators.py`](src/tabx/task_generators.py) 的
  `COMPOSITION_ARCHETYPES`、`FRONTLINE`、`BACKLINE`、`HEALERS`、`ELITE` 等常量。
- 双方单位类型与数量采样：`sample_compositions()`。
- 单位属性模板：[`src/tabx/units.py`](src/tabx/units.py) 的 `get_all_unit_spec()`。
- 编成统计（单位计数、总价格、近战/远程/治疗比例）：`composition_features()`。
- 将编成接入完整 task：`generate_programmatic_task()`。

#### B. 程序化布局

在现有 `positions`、`rotations` 字段上增加生成器，不需要改变 observation/action schema。

建议采样的布局特征：

- 双方质心距离。
- 队内紧密程度。
- 横向展开宽度。
- 前后排层次。
- 包围角度。
- 是否有侧翼单位。
- 单位初始朝向及视野覆盖。
- 单位与 zone 的关系。

布局原型可以包括：

```text
正面对阵 / 交叉火力 / 包围 / 伏击 / 突围 / 狭长纵深 / 分兵 / 保护核心
```

必须配套最小距离和边界检查，避免单位重叠、Mammoth 卡住或单位出生在不合理区域。

**代码位置：**

- 8 种布局原型注册：[`src/tabx/task_generators.py`](src/tabx/task_generators.py) 的
  `LAYOUT_ARCHETYPES`。
- `compact/line/dispersed` 队形偏移：`_formation_offsets()`。
- 按布局原型、距离桶生成目标位置：`_layout_targets()`。
- 按单位半径和地图边界寻找无重叠位置：`_place_without_overlap()`。
- 填充 `positions`、`rotations`、`pos_min/max` 及完整 `VectorizedScenario`：
  `build_scenario()`。
- 最终边界与出生重叠校验：[`src/tabx/sample_task.py`](src/tabx/sample_task.py) 的
  `validate_task()`。

#### C. Zone 的关系式生成

比“独立随机坐标”更有效的方法是按战术关系生成：

- lava 位于双方最短接敌路径上。
- bush 位于侧翼、远程单位附近或视野边缘。
- swamp 覆盖中央争夺区或撤退路径。
- 一侧有掩护、另一侧有距离优势。
- zone 重叠、相切、形成通道或分割战场。

这会比均匀随机 `position/axes` 更容易产生真实策略差异。

**代码位置：**

- zone 原型、强度和空间关系分桶：
  [`src/tabx/task_generators.py`](src/tabx/task_generators.py) 的
  `ZONE_ARCHETYPES`、`ZONE_INTENSITY_BUCKETS`、`ZONE_RELATION_BUCKETS`。
- 计算双方质心、接敌中点、接敌主轴和侧翼方向：`_geometry()`。
- 按布局关系生成 lava/bush/swamp：`build_relational_zones()`。
- zone 强度映射：`_zone_effect()`。
- zone 边界约束与 lava 出生安全检查：`_fit_zone()`、
  `_lava_clear_of_spawns()`，以及 `validate_task()` 中的 programmatic-zone 检查。

#### D. 参数使用离散分桶

离线题库不宜直接从高维连续空间无限采样。可以先使用离散档位：

```text
地图：small / medium / large
队间距离：close / medium / far
展开：compact / line / dispersed
zone 强度：low / medium / high
属性倍率：0.9 / 1.0 / 1.1
```

优点是可解释、可复现、便于覆盖统计，也便于后续针对空缺桶补采样。

**代码位置：**

- CLI 与默认桶配置：[`src/tabx/sample_task.py`](src/tabx/sample_task.py) 的
  `SampleConfig`。
- `small/medium/large`：`SampleConfig.map_scales`，由 `build_scenario()` 应用。
- `close/medium/far`：`DISTANCE_BUCKETS` 和 `_layout_targets()`。
- `compact/line/dispersed`：`SPREAD_BUCKETS` 和 `_formation_offsets()`。
- `low/medium/high` zone 强度：`ZONE_INTENSITY_BUCKETS` 和 `_zone_effect()`。
- 友敌双方独立属性倍率：`SampleConfig.stat_scales` 和 `_scale_team_stats()`。
- task 抽样、组合所有桶、写入 `metadata`、去重：`sample_tasks()`。
- 单文件 task bank 保存/加载：`save_task_bank()`、`load_task_bank()`、
  `build_batched_env_params_from_task_file()`。

**采样命令：**

```bash
conda run -n tabx python -m src.tabx.sample_task \
  --output tasks.json \
  --n-tasks 1000 \
  --seed 0
```

### 4.2 第二层：扩展现有 level generator

可以为 `FREE_PARAM_TYPES` 增加：

```text
unit_comp
position
rotation
zone_count
map_size
physics
```

推荐仍然输出现有的 `VectorizedScenario` / `ZoneScenario`，并固定最大槽位。这样 baseline 网络输入维度不变。

[`src/tabx/scenarios/scenario.py`](src/tabx/scenarios/scenario.py) 中存在尚未接入主流程的 `UnitScenario`，其中已有 `ally_unit_comp`、`enemy_unit_comp`、`battle_field` 等字段。它可以作为参考，但不应在未梳理清楚语义前直接接入。

### 4.3 第三层：新增机制

下面这些会带来更大的语义多样性，但开发和验证成本明显更高：

- 新 zone 类型。
- 不可穿越墙体或障碍。
- 占点、护送、守住若干步等目标。
- 动态事件或增援。
- 第三阵营。
- 新单位与新动作。

其中新增目标最能提高 task 语义多样性，但会改变胜负、奖励、终止条件和评测方式；新增动作会直接改变 action schema；提高 `max_n_*` 会改变 observation 维度和 agent 数量，通常需要重训。

### 4.4 不要把对称变换误算为 diversity

以下变化有数据增强价值，但不应在题库多样性统计中按独立战术 task 全额计数：

- agent 槽位 permutation。
- 整张地图纯镜像。
- 整体平移或旋转，但相对几何完全相同。
- 同类单位之间交换。
- 浮点数只有极小且无行为影响的变化。

建议为 task 建立 canonical representation：

1. 单位按 team、type、位置排序。
2. 坐标归一化到地图尺寸。
3. 可选将镜像/旋转等价类映射到同一 canonical key。
4. 连续属性按容差量化后 hash。
5. 先做精确/近似去重，再做行为去重。

### 4.5 用特征覆盖来选最终题库

每个候选 task 可以提取一个特征向量：

```text
编成：
  双方单位数、每类单位计数、总价格、总 HP、总 DPS、治疗量、射程分布

几何：
  地图尺寸、质心距离、最近敌我距离、队内离散度、包围角、朝向覆盖

zone：
  类型计数、面积占比、与单位/接敌路径的距离、双方覆盖差

评测：
  胜率、HP margin、episode length、truncation rate、first-kill rate、
  attack success、damage/heal、seed 间方差
```

最终选择可采用：

1. 先按编成原型、布局原型、zone 原型和难度分桶。
2. 每个桶设最小/最大配额，防止某类组合淹没题库。
3. 桶内使用 farthest-point sampling、k-medoids 或基于距离的贪心选择。
4. 将行为指标加入距离，避免参数不同但战斗轨迹高度相似。

这比“随机生成 N 个后全部保留”更能保证固定容量题库的有效 diversity。

---

## 5. 怎样保证 task 质量

建议把质量控制分成四道门。前两道便宜，应在大规模 rollout 前执行。

当前实现由 [`src/tabx/sample_task.py`](src/tabx/sample_task.py) 负责生成阶段的静态门，
由 [`src/tabx/eval_task.py`](src/tabx/eval_task.py) 负责双方 heuristic 多 seed rollout
及指标汇总。评测器当前负责**计算指标和标记退化局面**，不会自动删除 task；最终阈值和
easy/medium/hard 分桶仍应根据题库统计分布确定。

### 5.1 第一门：静态合法性

每个候选必须满足：

- 双方至少各有一个有效、存活、非 disabled 单位。
- 数量不超过固定 `max_n_ally/max_n_enemy`。
- zone 数不超过 `max_n_zone`。
- 所有数值 finite，无 NaN/Inf。
- HP、半径、重量、速度、射程、冷却等满足合理范围。
- 非 healer 的 damage 为正；healer 的治疗符号和 attack type 一致。
- 单位在 `pos_min/pos_max` 内。
- 任意两个单位的初始距离大于半径之和加安全 margin。
- zone 的 axes 为正，中心和范围与地图相容。
- 初始时不能因边界或 zone 造成非预期秒杀。

这里的“合理范围”不应简单等于当前 UED 的 min/max。应按单位角色设置条件约束，例如治疗单位、远程单位和坦克分别使用不同分布。

**代码位置：**

- [`src/tabx/sample_task.py`](src/tabx/sample_task.py) 的 `validate_task()`：
  检查字段完整性、finite、双方有效单位、字段长度、地图边界、单位出生重叠、
  zone 数组长度和半轴。
- `validate_task()` 的 programmatic-zone 分支：检查 bush effect、swamp effect 范围，
  并禁止单位出生时与 lava 相交。
- [`src/tabx/task_generators.py`](src/tabx/task_generators.py) 的
  `_place_without_overlap()`、`_fit_zone()`、`_lava_clear_of_spawns()`：
  在生成阶段提前满足上述约束。
- `save_task_bank()` 和 `build_batched_env_params_from_tasks()`：
  检查固定 `max_n_ally/max_n_enemy/max_n_zone` 是否足以容纳整个 task bank。

### 5.2 第二门：交战性和退化局面

使用确定性或低随机性的 heuristic 做少量 smoke rollout，排除：

- 双方长时间完全不接敌。
- 全局 damage 和 heal 始终为 0。
- 几乎所有局都超时且双方 HP 很高。
- 开局极短时间内一方被秒杀。
- 单位持续撞边界、卡住或只原地转向。
- 胜负由出生在 lava 中等明显生成错误决定。

环境没有墙，因此不需要传统图搜索连通性；但仍需要“能否在时限内形成有效交战”的动态检查。

**代码位置：**

- [`src/tabx/eval_task.py`](src/tabx/eval_task.py) 的 `_make_episode_runner()`：
  双方所有单位逐步调用 `heuristic_policy()`，收集攻击、伤害、治疗、首杀和终局信息。
- `_aggregate_episode_metrics()`：输出 `no_interaction_rate`、
  `short_episode_rate`、`all_truncated`、`all_one_sided` 等 `quality_flags`。
- `EpisodeMetrics`：定义每次 rollout 保留的原始质量指标。

### 5.3 第三门：以 heuristic AI 为基准的难度

#### 固定评测协议

离线题库需要固定：

- heuristic 代码版本或 commit。
- heuristic preset，例如 `expert`。
- 是否将 `epsilon` 固定为 0。
- physics preset。
- `max_episode_steps`。
- seed 列表。
- 是否做阵营/地图翻转。

建议分两阶段：

1. **确定性筛选**：使用 `expert` 且令 `epsilon=0`，快速检查结构难度和可重复性。
2. **随机鲁棒性评估**：恢复目标 preset 的 epsilon，用 32～64 个固定 seed 估计统计量。

32～64 只是初始预算，不是理论保证。对胜率接近阈值的 task，应自适应追加 rollout，直到置信区间足够窄或达到预算上限。

#### 不只看胜率

胜率是第一指标，但以下 task 即使胜率相同，质量可能完全不同：

- 50% 的局面每次都是开局随机秒杀。
- 50% 的局面经过长期交战才分胜负。
- 50% 的局面大多超时，靠 tie-break 决定。

建议至少记录：

| 指标 | 用途 |
|---|---|
| ally win rate + Wilson interval | 主难度指标及不确定性 |
| episode length 的中位数和分位数 | 排除过短或长期僵持 |
| truncation rate | 识别靠超时判胜的 task |
| final HP margin | 识别碾压局和胶着局 |
| first-kill side/time | 识别先手偏差和秒杀 |
| attack success rate | 识别无效交战 |
| cumulative damage/heal | 识别无交互和治疗循环 |
| seed 间方差 | 识别高度随机的 task |

#### 难度分桶

因为双方可能是不对称编成，难度应相对于“目标方”定义。

不建议一开始硬编码永久阈值。推荐：

1. 生成一批较大的候选集。
2. 用固定协议评测。
3. 根据胜率和 HP margin 的联合分布设分位数桶。
4. 人工抽检每个桶。
5. 再固定本版本题库的阈值。

可作为首轮讨论起点，而非最终标准：

```text
过难：目标方胜率置信区间上界仍很低，且经常无法造成有效伤害
hard：胜率偏低，但有稳定交战和可观察的成功路径
medium：胜率接近中间区域，胜负不依赖大量 timeout
easy：胜率偏高，但不是极短碾压
过易：胜率置信区间下界仍很高，且 HP margin/时长显示明显碾压
```

如果题库要服务后续 RL 训练，建议保留 easy/medium/hard，而不是只保留 50% 胜率附近。只保留临界 task 会损失课程覆盖和能力诊断价值。

**代码位置：**

- [`src/tabx/eval_task.py`](src/tabx/eval_task.py) 的 `EvalConfig`：
  固定 heuristic、physics、seed 数、`epsilon_override` 和最大 episode 长度。
- `_make_episode_runner()`：双方共享同一份 heuristic preset；默认
  `epsilon_override=0.0`，即使用 expert 的其他参数但关闭随机动作。
- `evaluate_tasks()`：使用相同 seed 集合逐 task 执行多 seed 评测。
- `wilson_interval()`：计算 ally 胜率的 95% Wilson 区间。
- `_aggregate_episode_metrics()`：汇总胜率、episode length 分位数、truncation、
  HP margin、first kill、attack success、damage/heal 和 seed 方差。

### 5.4 第四门：公平性与侧偏

当前超时 HP ratio 平局偏向敌方，所以“同配置双方 AI 胜率 50%”不是天然成立的。

每个 task 建议做成对评测：

1. 原始阵营和布局。
2. 交换双方编成和位置。
3. 左右或上下镜像地图。
4. 使用相同 seed 集合，尽量采用共同随机数。

记录：

```text
side_bias = 原始目标方胜率 - 翻转后对应编成胜率
```

侧偏大不一定要删除。例如 ambush 本来就可以是不对称任务。但应明确标注为“情境优势”，不能误认为是纯单位平衡。

**代码位置：**

- [`src/tabx/eval_task.py`](src/tabx/eval_task.py) 的 `flip_task()`：
  交换双方单位与 team 身份，同时保留对应物理布局。
- `EvalConfig.evaluate_flipped`：控制是否用相同 seed 集合追加翻转评测。
- `main()`：将原始和翻转结果配对，并写入 `flipped` 和 `side_bias`。
- 注意：当前实现完成的是阵营/编成翻转对照，不会另外执行左右或上下地图镜像。

**评测命令：**

```bash
conda run -n tabx python -m src.tabx.eval_task \
  --task-file tasks.json \
  --output task_eval.json \
  --heuristic expert \
  --epsilon-override 0.0 \
  --num-seeds 32 \
  --max-episode-steps 512 \
  --evaluate-flipped
```

---

## 6. 对 buff/debuff 临界方案的评价

### 6.1 这个 idea 有什么价值

设目标方属性倍率为 `1 + δ` 和 `1 - δ`：

```text
若 +δ 后明显更容易获胜，
且 -δ 后明显更容易失败，
则局面处在一个对该属性敏感的临界区域。
```

它能帮助找到：

- 胜负边界附近的局面。
- 对 HP、速度、damage 等能力敏感的局面。
- 适合做鲁棒性或能力诊断的 task。

### 6.2 为什么不能把它直接等同于“质量”

1. **敏感不等于困难**：一个存在离散攻击阈值的简单局，也可能被 1% damage 翻转。
2. **单局噪声很大**：heuristic 的 epsilon、碰撞、朝向、非指向攻击和 cooldown 都会放大微扰。
3. **变化未必单调**：提高速度可能让远程单位更容易冲入危险区域；提高 HP 可能改变 heuristic 的目标选择。
4. **绝对微扰不公平**：现有 `±0.1` 对不同属性的相对影响差别巨大。
5. **双方全由同一 heuristic 控制并不保证公平**：角色逻辑、地图侧偏和超时规则仍会造成系统偏差。

因此应将它命名为 `sensitivity_score`，作为题库标签或辅助准入指标，而不是唯一的 difficulty score。

### 6.3 更可靠的实验协议

对每个候选 task 和每个 buff 轴：

```text
δ ∈ {1%, 2%, 5%, 10%}
variant ∈ {-δ, baseline, +δ}
seed 使用同一固定集合
```

推荐：

- 使用乘法 buff，而不是绝对加法。
- 一次只改变一个语义清楚的轴。
- 对目标方全队加 buff，或对明确角色组加 buff；两者分开记录。
- 对每个 variant 运行相同 seed。
- 比较胜率置信区间、HP margin 和时长，而非单次胜负。
- 检查整体单调趋势，不要求每个 seed 都翻转。

可以定义：

```text
sensitivity(δ) = win_rate(+δ) - win_rate(-δ)
margin_sensitivity(δ) = mean_hp_margin(+δ) - mean_hp_margin(-δ)
```

高质量临界 task 的建议特征：

- baseline 不是明显碾压或完全不可打。
- `sensitivity(δ)` 在多个相邻 δ 上方向一致。
- 差异不是完全由 timeout tie-break 产生。
- 结果在阵营/地图翻转后可解释。
- 随机 seed 方差不过分大。

### 6.4 比 buff 更稳定的补充方法

#### A. 对手强度扫描

固定 task，扫描：

```text
novice → medium → advanced → expert
```

如果目标方对 novice 稳定胜、对 expert 稳定负、在中间档发生过渡，这个 task 的难度结构通常比微小 stat 翻转更容易解释。

但 heuristic preset 之间未必严格单调，仍需实测。

#### B. 多 heuristic 配置集成

题库主难度仍以一个固定 preset 为准，同时用少量不同 epsilon/aggressive 配置做鲁棒性验证。这样可以防止 task 只针对 heuristic 的某个确定性漏洞。

#### C. 能力向量而非单一分数

为 task 标注：

```text
近战接敌难度
远程拉扯难度
治疗保护难度
zone 利用强度
布局敏感性
属性敏感性
随机性敏感性
```

这比把所有 task 压成一个 difficulty scalar 更适合高 diversity 题库。

---

## 7. 当前代码中需要先确认的两处风险

### 7.1 `TABXHeuristicWrapper` 可能已落后于当前接口

[`src/tabx/wrappers/wrappers.py`](src/tabx/wrappers/wrappers.py) 的 `TABXHeuristicWrapper` 设计上支持 `heuristic_units="all"`，很适合双方 AI 对战。

但当前实现与主接口存在明显不一致：

- `reset()` 接受 `senario: VectorizedScenario`，而当前 `TABX.reset()` 使用完整 `env_params`。
- 当前 `heuristic_policy()` 需要 `last_visible_target`、`num_agents`、`num_zones`、`heuristic_config`、`physics_params`，并返回 `(action, LastVisibleTarget)`。
- `TABXHeuristicWrapper.step()` 仍按较旧的参数形式调用，并直接把返回值赋给 action。

因此，不能假设 `TABXHeuristicWrapper("all")` 当前可直接作为离线评测 runner。正式实现题库生成前，应先对齐该 wrapper，或基于已更新的 `TABXEnemyHeuristicWrapper` 抽取一个双方通用版本。

### 7.2 UED zone 类型与 effect 的一致性

如 3.8 所述：

- `jax.random.randint(..., maxval=3)` 不会生成 swamp(type 3)。
- effect coefficient 可能按旧 type 计算。

如果直接大规模生成而不修正，题库会系统性缺少 swamp，并可能出现 zone 类型与强度不一致。

---

## 8. 推荐的离线题库落地方案

### 8.1 MVP：先验证质量协议

暂时不新增环境机制：

1. 选取现有 4 个 unit base 和 5 个 zone base。
2. 加入 challenge 中可复用的布局原型。
3. 对位置、朝向和 zone 做有限离散变体。
4. 固定单位原始属性，暂不做全范围独立随机。
5. 修正/实现双方 heuristic runner。
6. 对每个候选运行确定性筛选和 32 个固定 seed。
7. 做阵营翻转。
8. 按编成、布局、zone、难度分桶，去重后保存。

MVP 的目标不是最大数量，而是验证：

- 指标能否排除坏局。
- heuristic 难度是否稳定。
- 哪些参数最能产生行为 diversity。
- 单个 task 的评测成本。

### 8.2 第二阶段：扩大候选空间

增加：

- 程序化单位编成原型。
- 合法位置/朝向生成器。
- 关系式 zone 生成器。
- 地图尺寸和物理参数分桶。
- 角色保持的属性倍率。
- task feature extractor 和多样性选择器。

这一阶段可以先生成远大于目标题库容量的候选，再经过质量门和覆盖选择压缩。

### 8.3 第三阶段：临界性和鲁棒性标签

只对已经通过基础质量门的 task 执行：

- `±δ` buff/debuff 扫描。
- 多 heuristic 档位扫描。
- 更多 seed 的置信区间收敛。
- 行为轨迹特征提取。

这能把昂贵预算集中在有价值的候选上。

### 8.4 建议的 task manifest

每个固化 task 除环境参数外，至少保存：

```yaml
task_id: canonical hash
generator_version: ...
source_scenario: ...
generation_seed: ...
schema:
  max_n_ally: ...
  max_n_enemy: ...
  max_n_zone: ...
evaluation_protocol:
  heuristic_preset: expert
  epsilon_override: ...
  physics: default
  max_episode_steps: ...
  seeds: [...]
quality:
  valid: true
  win_rate: ...
  win_rate_interval: [...]
  episode_length_quantiles: [...]
  truncation_rate: ...
  hp_margin: ...
  attack_success_rate: ...
  side_bias: ...
  sensitivity: ...
diversity_features:
  composition_bucket: ...
  layout_bucket: ...
  zone_bucket: ...
  difficulty_bucket: ...
```

环境参数可继续使用现有 JSON schema；manifest 用于索引、质量统计、版本管理和复现实验。

---

## 9. 推荐优先级

按投入产出比排序：

1. **先建立双方 heuristic 多 seed 评测 runner 和质量指标。**
2. **再增加单位编成、位置、朝向和 zone 关系的程序化生成。**
3. **加入 canonical hash、近似去重和分层覆盖选择。**
4. **用相对 `±δ` 作为 sensitivity 标签，而不是唯一质量门。**
5. **最后再考虑新目标、新障碍、新单位等 schema/规则级扩展。**

最关键的设计原则是：

> 先让“什么是好 task”可测量、可复现，再扩大生成空间。否则生成器越强，只会更快地产生大量重复、退化或难度不可解释的战局。
