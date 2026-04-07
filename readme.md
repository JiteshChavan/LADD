# Setup:

```bash
mkdir -p /root/Grace
git clone https://github.com/JiteshChavan/LADD /root/Grace
cd /root/Grace

# setup environment
source env_setup.sh
# download training data
cd /root/Grace
source download_8k_split.sh
# debug data
source download_debug_split.sh
# download distilled ckpt
source download_ckpt.sh
```
---

### Start debug training expt on 200 images:
```bash
cd /root/Grace/VideoX-Fun/scripts/z_image
bash smoke.sh
```
### 8xA100 run
```bash
bash final_run.sh
```
### Inference
```bash
bash sample.sh
```

# 1. Modeling Intuition:

### 1. Flow Transport from gaussian to data
![image.png](attachment:image.png)

### 2. The issue with low NFE inference
![image-2.png](attachment:image-2.png)

### 3. X0 reparametrization of the Flow Field to predict X0 at any given time t
![image-3.png](attachment:image-3.png)

### 4. Adversarial alignment between generative features corresponding to noised(X0) and noised (x0_hat), from a pretrained backbone.

#### Where x0_hat is rendered using a flow field that is only ever trained on discrete time steps [1.0, 0.75, 0.5, 0.25]
#### Basically align X0 and X0_hat indirectly via adversarial alignment between features from a pretrained backbone, when subjected to sample X0~pt(.|x0) and X0_hat ~ pt(.|x0_hat)

![image-4.png](attachment:image-4.png)

### 5. Basically objective ends up some sort of KLD minimization between pt(.|x0) and pt(.|x0_hat)
upon convergence samples from both the distributions look the same, one is cheaper to evaluate.

---

# 2. Insights on engineering issues encountered and their resolutions:

1. Initializing the student model exactly the same size and weights as the pre-trained base model.
    - The base model is 6B parameters although both teacher and student (12B total) can sit on an A100 idle, 80G VRAM is insufficient to train the studen initialized from teacher even if the teacher is frozen and only student is trainable.
    - even casting the weights as torch.bf16 batch size of 1 does not allow end to end training thats faithful to the paper.
    - Notably, the bottleneck is adamW optimizer state for trainable 6B params + the generator (student model) requiring gradients through the massive 6B teacher backbone for generator loss, and 80G is not enough to hold activations from both student and teacher at the same time.
    - using 8bit adamw combined with gradient checkpointing on both student + teacher model allows for batch size of 8 (512x512 resolution, 64x64 latents, 32x32 = 1024 tokens) with a memory footprint ~78.7GB.
    - Drawback is that enabling gradient checkpointing does store forward activations for both the models and during backward passes (through both student and teacher(through discriminators)) multiple partial forward passes are required. Thereby increasing time required per step (roughly 9 sec/step)

2. Again the bottle neck is having to hold activations for both student and teacher model.
    - Another possible solution to not require gradient checkpointing on teacher, is to not tap all the 30 layers from teacher for adversarial dynamic and just tap first few layers and do early exit.
    - although that deviates from the paper, I have implemented the framework to early exit during forward pass of teacher.
    - That weakens the adversarial supervision from discriminators though.
    
3. Another solution to increase through put is:
    - cut student size down, and initialize student from teacher but in a sparse interpolated way say teacher [0, 29]-> student [0, 5]
    - Observed drawback: partial init student does not have the correct flow field internalized.
    - Because such student doesnt know the flow field, its learning reconstructions purely from adversarial dynamics
    - most of the training is spent trying to learn colors and texture soup.
    - as good as random init with no baseline starting representation of flow field, so harder task to learn.
    - init from teacher starts giving really good visuals as early as step 200 in training while random or sparse init smaller student struggles even with correct colors till step ~2000 and even beyond at same training setup.

4. Z_image transformer backbone runs self attention on concat(image, text) tokens just like flux kontext .1
    - Important to keep resolution -> latent size -> img_tok_seq len in mind to tap into the features from teacher, and reshape them for conv nets backbone for disc loss.
    - hardcoded to 512x512 right now but in future can be modified to slice the tensor appropriately depending on the resolution and aspect ratio.
    - conv nets for discriminator generalize well to different resolutions

5. Decided to keep discriminator heads in torch.fp32 as they are simpler and smaller and size and might lack capacity
    - the optimizer for discriminators is default AdamW as well instead of 8bit version as the overhead is not significant and I hypothesize that it should stabilize the training better than using 8bit adam for disc heads.
    - the paper requires discriminator heads to be conditioned on text + time embedding, I used simple additive modulation with mlp(pooled text and time embedding).

6. Offline precomputation of Latents before starting training runs helps with the VRAM disaster.



--- 
# 3. Data Pipeline

The paper emphasizes using sythetic data for adversarial alignment between vectorfields u and u_hat at x_t sampled from conditional distributions $p(.| X0)$ and $p(.|X0_hat)$.

Since $X0  \sim  p_{data}(.)$ generated by the teacher backbone yields better adversarial supervision across generative features throughout different layers, when subjected to renoised X0.


To support LADD distillation, I constructed a dataset that matches the Z-Image training format using synthetic teacher-generated data.

1. Data Source
    - Sampled a subset of prompts from JourneyDB
    - Generated corresponding images using the teacher model (40 NFE)
    - Total dataset size: ~8k image-text pairs

This follows the LADD setup where supervision is derived from teacher-generated samples rather than raw data.

2. Latent Precomputation

    - To make training efficient and match the Z-Image pipeline:

    Images (512×512) are encoded using the SDXL VAE
    (stored as 64×64 latent tensors)
    Text captions are encoded using Qwen3ForCausalLM
    (stored as embeddings (dim = 2560, variable sequence length))

All data is stored as precomputed shards on disk to avoid repeated encoding during training.

3. Dataset Structure

Each training sample contains:

- latent image tensor (x₀)
- corresponding text embedding

4. Debug vs Full Dataset
    - Debug split (50–200 samples)
    - - Used for fast iteration, overfitting tests, and verifying training stability
    - Full 8k dataset
    - - Used for multi-GPU training
    
5. Rationale
Using teacher-generated data ensures consistency between teacher and student distributions during distillation
Latent precomputation significantly reduces training cost and simplifies the training loop
Debug split enables rapid validation before scaling to full runs
