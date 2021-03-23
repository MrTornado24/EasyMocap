## Preprocessing to Aist
```bash
python3 scripts/preprocess/preprocess_aist.py
```
**the dataset should be like the following:**
```bash
aist_plusplus_final
└── cameras_dataset
    └── gBR_sBM_cAll_d04_mBR0_ch01
        ├── videos
        │   ├── gBR_sBM_c01_d04_mBR0_ch01.mp4
        │   ├── gBR_sBM_c02_d04_mBR0_ch01.mp4
        │   └── gBR_sBM_c03_d04_mBR0_ch01.mp4
        ├── extri.yml
        ├── intri.yml
        └── setting1.json
```
**Extracting videos**
```bash
python3 scripts/preprocess/extract_video.py
```
```bash
aist_plusplus_final
└── cameras_dataset
    └── gBR_sBM_cAll_d04_mBR0_ch01
        ├── videos
        │   ├── gBR_sBM_c01_d04_mBR0_ch01.mp4
        │   ├── gBR_sBM_c02_d04_mBR0_ch01.mp4
        │   └── gBR_sBM_c03_d04_mBR0_ch01.mp4
        ├── annots
        │   ├── gBR_sBM_c01_d04_mBR0_ch01
        │   ├── gBR_sBM_c02_d04_mBR0_ch01
        │   └── gBR_sBM_c03_d04_mBR0_ch01
        ├── images
        │   ├── gBR_sBM_c01_d04_mBR0_ch01
        │   ├── gBR_sBM_c02_d04_mBR0_ch01
        │   └── gBR_sBM_c03_d04_mBR0_ch01
        ├── openpose
        │   ├── gBR_sBM_c01_d04_mBR0_ch01.mp4
        │   ├── gBR_sBM_c02_d04_mBR0_ch01.mp4
        │   └── gBR_sBM_c03_d04_mBR0_ch01.mp4
        ├── extri.yml
        ├── intri.yml
        └── setting1.json  
```
