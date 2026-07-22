# TABX Offline Sequence Pipeline

This document summarizes the offline sequence data pipeline for TABX / OmniRL.

## Pipeline

Current pipeline:

```text
trained coach checkpoint
→ env-centered entity trajectory
→ FOV-based agent-centered continuous sequence
→ field-wise tokenized sequence
→ OmniRL flat token sequence chunks

Why FOV instead of KNN

TABX original observation uses fan-shaped FOV visibility, not KNN.
Therefore the main pipeline follows visible_matrix from the environment.

Each ego-agent sequence keeps fixed unit slots:

slot 0: self
slot 1..N-1: all other units in global unit order

Invisible units keep their slots but their features are zeroed.
During tokenization, invisible unit slots are encoded with UNSEEN.

KNN conversion is kept only as an optional compact representation / ablation, not the main pipeline.

Action tokenization

The action space is already discrete:

Discrete(8)
0 UP
1 DOWN
2 LEFT
3 RIGHT
4 ATTACK
5 TURN_RIGHT
6 TURN_LEFT
7 IDLE

So actions can be directly mapped to action tokens.

Observation tokenization

The observation contains many continuous values, such as:

relative position
distance
health ratio
cooldown
speed
attack range
attack damage
body radius
zone axes
zone effect value
reward

These are discretized by rule-based binning in configs/tabx_fov_tokenizer_v0.json.

Main scripts
scripts/generate_coach_trajectories_entities.py
  Generate env-centered trajectories with raw entity states.

scripts/convert_entity_to_fov_agent_sequence.py
  Convert env-centered entity trajectories into FOV-based agent-centered continuous sequences.

scripts/tokenize_fov_agent_sequence.py
  Convert continuous FOV sequences into field-wise token ids.

scripts/flatten_fov_tokens_to_omnirl_sequence.py
  Flatten field-wise tokens into OmniRL-style chunked token sequences.
Representative task output

Representative tasks have been tested with chunk size 8.

Each task produces:

input_ids:      (10240, 2642)
attention_mask: (10240, 2642)
token_type_ids: (10240, 2642)
step_ids:       (10240, 2642)

where:

10240 = 160 agent sequences × 64 chunks
2642 = BOS + 8 × (20×15 + 4×7 + action + reward) + EOS

Vocabulary size in v0:

vocab_size = 584

