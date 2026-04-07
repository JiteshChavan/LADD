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
