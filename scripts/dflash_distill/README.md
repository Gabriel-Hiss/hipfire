# DFlash drafter distillation

Fine-tunes a DFlash draft on the target's own outputs, so the draft predicts
what the served target actually writes. Measured on Ternary-Bonsai-2-27B
PTQ1_0 with the 6-layer Qwen3.5-27B draft: offline accepted length per
16-token block 5.18 -> 9.52 on held-out conversations, greedy DFlash on the
agentic fixture 187 -> 275 tok/s. See
`docs/perf-checkpoints/2026-09-26-ternary-bonsai-ptq1-gfx1100-dflash-agentic.md`.

Steps (GPU steps need the GPU to themselves):

1. Conversations. Agentic coding contexts (system prompt, tools, a
   `read_file` result from a real source file, a request) whose next turn the
   target will write:

       python scripts/dflash_distill/make_convs.py convs.jsonl --n 1600 --root <src tree> [--root ...]

2. Target outputs. Serves the target with the base draft and
   `HIPFIRE_SPEC_TOKEN_LOG`, which appends `{"prompt","output"}` token ids per
   DFlash turn. Resumable (`<log>.ids` lists finished conversations):

       python scripts/dflash_distill/gen_responses.py convs.jsonl tok.jsonl 0 1600 \
           --model <target> --speculation dflash --draft <base draft .hfq> \
           --kv q8 --thinking off --sampling greedy --max-tokens 1536 --max-seq 8192

3. Target tables and hidden states. `tables` writes the input embedding rows
   and the effective lm_head (as the draft's logits see it); `hidden` replays
   prompt ++ output through the target and stores the draft's context layers
   (the draft's `target_layer_ids`, in order). Resumable:

       cargo run --release -p hipfire-runtime --features deltanet --example dflash_distill_dump -- \
           tables --target <target> --out tables
       cargo run --release -p hipfire-runtime --features deltanet --example dflash_distill_dump -- \
           hidden --target <target> --tokens tok.jsonl --layers 1,10,18,27,35,44,52,61 --out hidden

   About 1.2 MB per position for 8 layers at hidden 5120 in f16 (400
   conversations: ~80 GB).

4. LoRA fine-tune (PyTorch with ROCm), then merge into a bf16 checkpoint:

       python scripts/dflash_distill/train_draft.py --base <HF draft dir> --tables tables \
           --data hidden --tokens tok.jsonl --out ft --epochs 3

   `--eval-only` prints the base draft's offline accepted length.

5. Convert and serve:

       cargo run --release -p hipfire-quantize --bin dflash_convert -- --input ft --output ft-mq4.hfq --mq4
