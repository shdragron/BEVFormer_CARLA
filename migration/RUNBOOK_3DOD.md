# 3D OD full migration — pull-over-SSH runbook (school side)

Reproduce the full 6-model 3D Object Detection benchmark (train + eval) on the
school server by **rsync-pulling** from the source box. Mirror the absolute paths
→ near-zero code edits. Run everything inside `tmux` (rsync `-P` is resumable).

## 0. Decide the rail
- **rsync pull over SSH** (recommended): source box has an SSH ingress. Works iff
  the school network can reach that endpoint. Confirm:
  `ssh -p <PORT> hanyan_arch@<HOST> hostname`  → logs in ⇒ go.
- If unreachable → fallback to **croc** (relay on :9009, see `08_croc_send.sh`).

## 1. One-time prep
On the **source box** (add the school server's pubkey):
```bash
echo "<school id_ed25519.pub contents>" >> ~/.ssh/authorized_keys
```
On the **school server** (create mirror roots so paths match):
```bash
sudo mkdir -p /home/hanyan_arch/viewpoint \
              /NHNHOME/WORKSPACE/0526040099_A/jeongtae \
              /NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs
sudo chown -R $USER /home/hanyan_arch/viewpoint /NHNHOME/WORKSPACE/0526040099_A
SRC="hanyan_arch@<HOST>"; SSHC="ssh -p <PORT> -i ~/.ssh/<key>"     # set once
```

## 2. Pull — 3 tiers (value-first)

### Tier 1 — code + envs + eval ckpts  (~31 GB → all EVAL reproducible)
```bash
rsync -avP -e "$SSHC" $SRC:/NHNHOME/robogeo_migrate/ ~/robogeo_bundle/
# also the pretrained backbones the code tar omits (needed only to RE-train):
rsync -avP -e "$SSHC" --include='*/' --include='resnet50_msra*.pth' --exclude='*' \
  $SRC:/home/hanyan_arch/viewpoint/BEVFormer/ ~/robogeo_bundle/backbones/
```

### Tier 2 — data  (~2.4 TB → CTS + VP eval + training data)
```bash
J=/NHNHOME/WORKSPACE/0526040099_A/jeongtae
rsync -avP -e "$SSHC" $SRC:$J/carla_geobev/   $J/carla_geobev/      # 8.5G, symlinks kept
rsync -avP -e "$SSHC" $SRC:$J/simbev_compare/ $J/simbev_compare/    # 1.3T baseline imgs
rsync -avP -e "$SSHC" $SRC:$J/carla_VR/       $J/carla_VR/          # 1.1T VP variants
rsync -avP -e "$SSHC" $SRC:$J/simbev/ground-truth/ $J/simbev/ground-truth/
```
> 💡 You have `~/JG_workspace/SimBEV`. The 1.3 TB `simbev_compare` baseline images
> *could* be regenerated locally instead of pulled — but only the **pulled, original
> images guarantee the published eval numbers** (regen may drift). `carla_VR` (VP)
> must be pulled regardless. Recommendation: pull both for faithful reproduction.

### Tier 3 — full training work_dirs  (~141 GB → ONLY to resume exact training runs)
```bash
B=/home/hanyan_arch/viewpoint/BEVFormer
for d in work_dirs 3D-deformable-attention/BEVFormer_DFA3D/work_dirs \
         detr3d/work_dirs BEVDet/work_dirs BEVDepth/outputs; do
  rsync -avP -e "$SSHC" $SRC:$B/$d/ $B/$d/
done
```
> Skippable: eval is reproduced from the Tier-1 eval ckpts; **fresh** retraining
> needs only configs (in code) + backbones (Tier 1) + data (Tier 2), NOT these
> 141 GB of past epochs/optimizer states.

## 3. Restore (school server)
```bash
cd /home/hanyan_arch/viewpoint && tar --zstd -xf ~/robogeo_bundle/code_BEVFormer.tar.zst
for e in ~/robogeo_bundle/envs/*.tar.gz; do
  tar -xzf "$e" -C /NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs/; done
conda config --append envs_dirs /NHNHOME/WORKSPACE/0526040099_A/giyong/miniconda3/envs
ckptdir=~/robogeo_bundle/ckpts bash ~/robogeo_bundle/ckpts/restore_ckpts.sh
ln -sf /NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_geobev \
       /home/hanyan_arch/viewpoint/BEVFormer/data/nuscenes
# backbones (retrain): cp ~/robogeo_bundle/backbones/.../resnet50_msra*.pth into each repo's ckpts/
```

## 4. Verify
```bash
cd /home/hanyan_arch/viewpoint/BEVFormer && bash migration/06_verify.sh
# anchors: BEVFormer SUV-CAL CTS 71.9/72.5 ; BEVDepth BUS-CAL 16.6/0.4
```

## Sizes
| Tier | what | size | unlocks |
|---|---|---|---|
| 1 | code+envs+evalckpt(+backbones) | ~31 GB | all 6-model EVAL |
| 2 | data (geobev+simbev_compare+carla_VR) | ~2.4 TB | VP eval + training data |
| 3 | full work_dirs | ~141 GB | resume exact training |
| **total** | | **~2.57 TB** | full reproduction + retrain |
