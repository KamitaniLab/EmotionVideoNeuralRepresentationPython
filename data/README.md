# Emotional movie dataset

## Downloading dataset

First, please install [bdpy](https://github.com/KamitaniLab/bdpy) via pip.

```
$ pip install bdpy
```

Then you can download data with the following command.

```
$ python download.py <target>
```

Targets:

- `fmri`: All fMRI datasets used in Horikawa et al., 2020 paper
- `features`: Category, affective, semantic, and vision scores
- `pycortex`: All pycortex surface data

### Data files

```
data
├── features/amt/mean_score_concat
│   ├── category
│   ├── dimension
│   ├── semantic
│   └── vision
├── fmri
│   └── paper2020
│       ├── fmri_Subject1_for_tutorial.h5
│       ├── fmri_Subject1.h5
│       ├── fmri_Subject2.h5
│       ├── fmri_Subject3.h5
│       ├── fmri_Subject4.h5
│       └── fmri_Subject5.h5
└── pycortex
    ├── Subject1
    ├── Subject2
    ├── Subject3
    ├── Subject4
    └── Subject5
```

### Emotion category and dimension properties

Under `data/features/amt/mean_score_concat/category` and `data/features/amt/mean_score_concat/dimension`, the files `0001.mat`, `0002.mat`, ... contain the category/dimension scores for each movie clip, where **each file** (e.g., `0001.mat`) corresponds to **one movie clip**.

In `category/*.mat`, the columns correspond to the following emotion categories, in this order:

- admiration
- adoration
- aesthetic_appreciation
- amusement
- anger
- anxiety
- awe
- awkwardness
- boredom
- calmness
- confusion
- contempt
- craving
- disappointment
- disgust
- empathic_pain
- entrancement
- envy
- excitement
- fear
- guilt
- horror
- interest
- joy
- nostalgia
- pride
- relief
- romance
- sadness
- satisfaction
- sexual_desire
- surprise
- sympathy
- triumph

In `dimension/*.mat`, the columns correspond to the following affective dimensions:

- Approach
- Arousal
- Attention
- Certainty
- Commitment
- Control
- Dominance
- Effort
- Fairness
- Identity
- Obstruction
- Safety
- Upswing
- Valence
