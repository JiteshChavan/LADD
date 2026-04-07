export WANDB_API_KEY=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
accelerate launch --num_processes 8 --mixed_precision bf16 train.py \
  --gradient_checkpointing \
  \
  --teacher_init \
  --use_8bit_adam \
  --use_ladd \
  --use_adv \
  --adv_weight 1.0 \
  --disc_lr 1e-5 \
  --disc_feature_layers_teacher 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 \
  --discrete_time_pdf ladd_paper \
  \
  --report_to wandb \
  --tracker_project_name "Distillation" \
  --tracker_run_name "FinalRun" \
  \
  --pretrained_model_name_or_path /root/Grace/VideoX-Fun/models/Z-Image \
  --train_data_dir /root/Grace/VideoX-Fun/datasets/8k/precomputed_pt \
  --output_dir /root/Grace/output/final_run \
  --train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --inference_nfe 4 \
  --max_train_steps 41000 \
  --dataloader_num_workers 0 \
  --mixed_precision bf16 \
  --learning_rate 1e-4 \
  --train_sampling_steps 1000 \
  --seed 110597 \
  --sample_every 100 \
  --checkpoint_every 100 \
  --checkpoints_total_limit 3 \
  --sample_prompts \
"cinematic, realistic, 1920s girls at college, girls studying, university, dark academia, ghost story, horror, haunted, creepy" \
"arts and crafts maker making an amazing dress, 40 year old woman with red hair and freckles, 3d cartoon" \
"a movie poster for The Vvitch in the style of Arthur Rackham" \
"a beautiful landscape inside a crystal shaped like a d20 floating in a field of pink magenta yellow clouds above an abyssal hole to the center of earth, very realistic" \
"Bunch of bananas jumping and fighting cats, kitchen, intricate detail" \
"black pharaoh panther god, crystal purple filigree, insanely detailed and intricate, hypermaximalist, elegant, ornate, hyper realistic, super detailed, 8K" \
"Doctor of the plague" \
"paper cut craft, paper illustration, demon and flames" \
"a beautiful dark forest, a witches house in the trees, creepy, dark wooded" \
"Valkyrie taking souls from the battlefield" \
"Exhibition hall realistic, super detailed" \
"High Fantasy, Fire Dragon" \
  #--resume_from_checkpoint latest \
