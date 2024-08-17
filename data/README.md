# Sound Reconstruction Dataset

## Downloading dataset

First, please install [bdpy](https://github.com/KamitaniLab/bdpy) via pip (version >= 0.21).

```
$ pip install bdpy
```

Then you can download data with the following command.

```
$ python download.py <target>
```

Targets:

- `fmri`: fMRI dataset used in Horikawa et al., 2020 paper
- `features`: Category, affective, semantic, and vision scores
- `pycortex`: Pycortex surface data

### Data files

```
data
├── features/amt/mean_score_concat
│   ├── category
│   ├── dimension
│   ├── semantic
│   └── vision
├── fmri
│   ├── paper2020
│   │   ├── fmri_Subject1.h5
│   │   ├── fmri_Subject2.h5
│   │   ├── fmri_Subject3.h5
│   │   ├── fmri_Subject4.h5
│   │   └── fmri_Subject5.h5
│   └── general (We plan to make an addition later on)
│       ├── fmri_Subject1.h5
│       ├── fmri_Subject2.h5
│       ├── fmri_Subject3.h5
│       ├── fmri_Subject4.h5
│       └── fmri_Subject5.h5
└── pycortex
    ├── Subject1
    ├── Subject2
    ├── Subject3
    ├── Subject4
    └── Subject5
```

