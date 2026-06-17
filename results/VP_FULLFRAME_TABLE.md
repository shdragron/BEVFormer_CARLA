# VP full-frame mVRS (1/7) — consolidated source of truth

metric: **mVRS(1/7) = (6·mRRS_percam + RRSALL_allcam)/7**, raw = mVRS%×P_Normal.
All models use full-frame VP. (DETR3D/BEVFormer per-cam mRRS lives in working dirs; merged here. DETR3D P_Normal corrected 0.537→0.533 to match its own full-frame run.)

| Model | P_Normal | EXT | IMG | CAL | fps | source |
|---|---|---|---|---|---|---|
| BEVDet | 0.5166 | 89.1 | 83.0 | 87.5 | 79 | eval_vp.json |
| BEVDepth | 0.5354 | 90.2 | 82.8 | 87.4 | 79 | eval_vp.json |
| BEVFormer | 0.5037 | 83.2 | 83.9 | 93.6 | 4(percam)+79(allcam) | vp_tiny_sedan_fullval_percam |
| DFA3D | 0.4892 | 90.5 | 86.1 | 93.1 | 4 | vp_dfa3d_sedan_fullframe.json |
| DETR3D | 0.5333 | 84.3 | 84.1 | 96.0 | 4 | eval_vp.json |
| CAPE | 0.5547 | 94.0 | 83.9 | 88.6 | 1000 | vp_cape_sedan_fullframe.json |

## ready-to-paste LaTeX VP block (\rc{pct}{raw})
```latex
BEVDet     & 0.5166 & \rc{89.1}{0.461} & \rc{83.0}{0.429} & \rc{87.5}{0.452} \\
BEVDepth   & 0.5354 & \rc{90.2}{0.483} & \rc{82.8}{0.443} & \rc{87.4}{0.468} \\
BEVFormer  & 0.5037 & \rc{83.2}{0.419} & \rc{83.9}{0.422} & \rc{93.6}{0.471} \\
DFA3D      & 0.4892 & \rc{90.5}{0.443} & \rc{86.1}{0.421} & \rc{93.1}{0.455} \\
DETR3D     & 0.5333 & \rc{84.3}{0.450} & \rc{84.1}{0.449} & \rc{96.0}{0.512} \\
CAPE       & 0.5547 & \rc{94.0}{0.521} & \rc{83.9}{0.465} & \rc{88.6}{0.491} \\
```
