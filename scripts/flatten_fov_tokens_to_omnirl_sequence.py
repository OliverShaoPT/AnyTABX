from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


TOKEN_TYPE_UNIT = 0
TOKEN_TYPE_ZONE = 1
TOKEN_TYPE_ACTION = 2
TOKEN_TYPE_REWARD = 3
TOKEN_TYPE_SPECIAL = 4


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def load_metadata(d):
    if "metadata" not in d.files:
        return {}
    try:
        return json.loads(str(d["metadata"]))
    except Exception:
        return {}


def flatten_one(src: Path, dst: Path, config_path: Path, chunk_steps: int, stride_steps: int):
    cfg = load_json(config_path)
    special = cfg["special_tokens"]

    pad_id = int(special["PAD"])
    bos_id = int(special["BOS"])
    eos_id = int(special["EOS"])

    d = np.load(src, allow_pickle=True)

    unit = d["unit_token_ids"]       # [S,T,U,Fu]
    zone = d["zone_token_ids"]       # [S,T,Z,Fz]
    action = d["action_token_ids"]   # [S,T]
    reward = d["reward_token_ids"]   # [S,T]

    S, T, U, Fu = unit.shape
    _, _, Z, Fz = zone.shape

    unit_per_step = U * Fu
    zone_per_step = Z * Fz
    block_len = unit_per_step + zone_per_step + 1 + 1
    max_len = 1 + chunk_steps * block_len + 1

    chunks = []
    masks = []
    token_types = []
    step_ids = []
    source_sequence_ids = []
    start_steps = []
    end_steps = []

    for s in range(S):
        for start in range(0, T, stride_steps):
            end = min(start + chunk_steps, T)
            if start >= end:
                continue

            ids = np.full((max_len,), pad_id, dtype=np.int32)
            attn = np.zeros((max_len,), dtype=bool)
            ttype = np.full((max_len,), TOKEN_TYPE_SPECIAL, dtype=np.int16)
            sid = np.full((max_len,), -1, dtype=np.int32)

            pos = 0
            ids[pos] = bos_id
            attn[pos] = True
            ttype[pos] = TOKEN_TYPE_SPECIAL
            sid[pos] = -1
            pos += 1

            for t in range(start, end):
                u_flat = unit[s, t].reshape(-1)
                z_flat = zone[s, t].reshape(-1)

                n = len(u_flat)
                ids[pos:pos+n] = u_flat
                attn[pos:pos+n] = True
                ttype[pos:pos+n] = TOKEN_TYPE_UNIT
                sid[pos:pos+n] = t
                pos += n

                n = len(z_flat)
                ids[pos:pos+n] = z_flat
                attn[pos:pos+n] = True
                ttype[pos:pos+n] = TOKEN_TYPE_ZONE
                sid[pos:pos+n] = t
                pos += n

                ids[pos] = action[s, t]
                attn[pos] = True
                ttype[pos] = TOKEN_TYPE_ACTION
                sid[pos] = t
                pos += 1

                ids[pos] = reward[s, t]
                attn[pos] = True
                ttype[pos] = TOKEN_TYPE_REWARD
                sid[pos] = t
                pos += 1

            ids[pos] = eos_id
            attn[pos] = True
            ttype[pos] = TOKEN_TYPE_SPECIAL
            sid[pos] = -1

            chunks.append(ids)
            masks.append(attn)
            token_types.append(ttype)
            step_ids.append(sid)
            source_sequence_ids.append(s)
            start_steps.append(start)
            end_steps.append(end)

    input_ids = np.stack(chunks, axis=0)
    attention_mask = np.stack(masks, axis=0)
    token_type_ids = np.stack(token_types, axis=0)
    step_ids = np.stack(step_ids, axis=0)

    source_sequence_ids = np.asarray(source_sequence_ids, dtype=np.int32)
    start_steps = np.asarray(start_steps, dtype=np.int32)
    end_steps = np.asarray(end_steps, dtype=np.int32)

    metadata = load_metadata(d)
    metadata.update({
        "source_file": str(src),
        "format": "omnirl-flat-token-sequence-v0",
        "tokenizer_config": str(config_path),
        "chunk_steps": chunk_steps,
        "stride_steps": stride_steps,
        "num_chunks": int(input_ids.shape[0]),
        "max_seq_len": int(input_ids.shape[1]),
        "block_len_per_step": int(block_len),
        "unit_tokens_per_step": int(unit_per_step),
        "zone_tokens_per_step": int(zone_per_step),
        "layout": "BOS + repeated([unit_tokens, zone_tokens, action_token, reward_token]) + EOS + PAD",
        "token_type_ids": {
            "UNIT": TOKEN_TYPE_UNIT,
            "ZONE": TOKEN_TYPE_ZONE,
            "ACTION": TOKEN_TYPE_ACTION,
            "REWARD": TOKEN_TYPE_REWARD,
            "SPECIAL": TOKEN_TYPE_SPECIAL
        }
    })

    dst.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        dst,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        step_ids=step_ids,
        source_sequence_ids=source_sequence_ids,
        start_steps=start_steps,
        end_steps=end_steps,
        metadata=json.dumps(metadata),
    )

    print("saved:", dst)
    print("  input_ids:", input_ids.shape)
    print("  attention_mask:", attention_mask.shape)
    print("  token_type_ids:", token_type_ids.shape)
    print("  step_ids:", step_ids.shape)
    print("  chunks:", input_ids.shape[0])
    print("  max_seq_len:", input_ids.shape[1])
    print("  token range:", int(input_ids.min()), int(input_ids.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/tabx_fov_tokenizer_v0.json")
    ap.add_argument("--chunk-steps", type=int, default=8)
    ap.add_argument("--stride-steps", type=int, default=8)
    args = ap.parse_args()

    flatten_one(
        src=Path(args.input),
        dst=Path(args.output),
        config_path=Path(args.config),
        chunk_steps=args.chunk_steps,
        stride_steps=args.stride_steps,
    )


if __name__ == "__main__":
    main()
