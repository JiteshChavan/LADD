export WANDB_API_KEY=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
python sample.py \
  --model_path /root/Grace/VideoX-Fun/models/Z-Image \
  --ckpt_path /root/Grace/checkpoints/checkpoint-20300 \
  --inference_nfe 4 \
  --sample_prompts "librarian with purple wavy hair, books, pixar, animated" \
  "fluffy koala bear happy" \
  "blue bird with rocket engines realistic" \
  "the empress tarot card painted in the style of Hilda af klint" \
  "crater lake shore lane, wide angle lens, overlook" \
  "starfield, deep sky, realistic, 8k" \
  "distant orbital spaceship science station," \
  "purple skin camels in house" \
  --sample_dir /root/Grace/checkpoints/checkpoint-5300/samples \
  --seed 42 \
  --run_name smoke_ckpt5000_nfe1