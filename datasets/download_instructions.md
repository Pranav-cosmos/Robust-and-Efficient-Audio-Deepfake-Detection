# Dataset Download Instructions

## Overview

This project requires the following datasets. Due to registration requirements and 
large file sizes, automated download is not possible for most datasets.

---

## 1. ASVspoof 2019 LA (Primary — Train/Dev/Eval)

**Source:** [Edinburgh DataShare](https://datashare.is.ed.ac.uk/handle/10283/3336)

**Size:** ~5 GB

**Steps:**
1. Visit https://datashare.is.ed.ac.uk/handle/10283/3336
2. Download `LA.zip`
3. Extract to `data/ASVspoof2019_LA/`

**Expected structure:**
```
data/ASVspoof2019_LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/
│   └── flac/
├── ASVspoof2019_LA_dev/
│   └── flac/
└── ASVspoof2019_LA_eval/
    └── flac/
```

---

## 2. ASVspoof 2021 LA (Test B)

**Source:** [Zenodo](https://zenodo.org/record/4837263)

**Keys:** https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz

**Steps:**
1. Download evaluation data from Zenodo
2. Download keys from ASVspoof website
3. Extract both to `data/ASVspoof2021_LA/`

**Expected structure:**
```
data/ASVspoof2021_LA/
├── ASVspoof2021_LA_eval/
│   └── flac/
└── keys/
    └── LA/
        └── CM/
            └── trial_metadata.txt
```

---

## 3. ASVspoof 2021 DF (Test B)

**Source:** [Zenodo](https://zenodo.org/record/4835108)

**Keys:** https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz

**Steps:** Same as 2021 LA but extract to `data/ASVspoof2021_DF/`

**Expected structure:**
```
data/ASVspoof2021_DF/
├── ASVspoof2021_DF_eval/
│   └── flac/
└── keys/
    └── DF/
        └── CM/
            └── trial_metadata.txt
```

---

## 4. WaveFake (Test C)

**Source:** [Zenodo #5642694](https://zenodo.org/record/5642694)

**Size:** ~4.5 GB

**Note:** WaveFake only contains generated (spoof) audio. You also need
LJSpeech as the bonafide reference.

**LJSpeech:** https://keithito.com/LJ-Speech-Dataset/

**Steps:**
1. Download WaveFake from Zenodo and extract to `data/WaveFake/`
2. Download LJSpeech-1.1 and extract to `data/LJSpeech-1.1/`

**Expected structure:**
```
data/WaveFake/
├── ljspeech_full_band_melgan/
├── ljspeech_hifiGAN/
├── ljspeech_melgan/
├── ljspeech_melgan_large/
├── ljspeech_multi_band_melgan/
├── ljspeech_parallel_wavegan/
└── ljspeech_waveglow/

data/LJSpeech-1.1/
└── wavs/
```

---

## 5. MUSAN (For Noise Perturbations)

**Source:** [OpenSLR #17](https://www.openslr.org/17/)

**Size:** ~11 GB (only noise subset needed)

**Steps:**
1. Download from https://www.openslr.org/17/
2. Extract to `data/musan/`
3. Only the `noise/` subfolder is used for background noise perturbations

**Expected structure:**
```
data/musan/
├── noise/
│   └── free-sound/
│       ├── noise-free-sound-0000.wav
│       └── ...
├── speech/  (not required)
└── music/   (not required)
```

---

## Verification

After downloading, run:
```bash
python scripts/prepare_datasets.py --verify
```

This will check all expected paths and report any missing files.
