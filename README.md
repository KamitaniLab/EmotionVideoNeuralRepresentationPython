# The neural representation of visually evoked emotion is high-dimensional, categorical, and distributed across transmodal brain regions

This repository is Python version of the following repositoy:
[KamitaniLab/EmotionVideoNeuralRepresentation](https://github.com/KamitaniLab/EmotionVideoNeuralRepresentation)

This repositoy contains the data and some code for reproducing results in our paper: [Horikawa, Cowen, Keltner, and Kamitani (2020) The neural representation of visually evoked emotion is high-dimensional, categorical, and distributed across transmodal brain regions. iScience](https://www.cell.com/iscience/fulltext/S2589-0042(20)30245-5?rss=yes).
We investigated the neural representation of visually evoked emotions using fMRI responses to 2185 emotionally evocative short videos generated by [Alan S. Cowen and Dacher Keltner (PNAS, 2017)](https://doi.org/10.1073/pnas.1702247114).


## Dataset

- Preprocessed fMRI data, DNN features extracted from sound clips: [figshare](https://figshare.com/articles/dataset/23633751)
- Trained transformer models: [figshare](https://figshare.com/articles/dataset/23633751)
- Stimulus sound clips: Refer to [data/README.md](data/README.md).

- The preprocessed fMRI data for five subjects and ratings/features (category, dimension, visual object, and semantic) are available at [figshare](https://doi.org/10.6084/m9.figshare.11988351).
- The raw fMRI data (bids format) is available at [OpenNeuro](https://openneuro.org/datasets/ds002425).

## Video stimuli

- We used 2185 emotion evocative short videos collected by Cowen and Keltner (2017).   
- You can request the videos with emotion ratings from the following URL (https://goo.gl/forms/XErJw9sBeyuOyp5Q2).


## Setup

### Prepare environment

1. Clone this `EmotionVideoNeuralRepresentationPython` repository to your local machine (GPU machine preferred).
```
git clone git@github.com:KamitaniLab/EmotionVideoNeuralRepresentationPython.git
```

2. Create conda environment using the `env.yaml`.
```
conda env create --name kmtnlabemotion -f env.yaml 
python -c "import torch; print(torch.cuda.is_available())"
# True
```

3. Clone the following repositoris under `EmotionVideoNeuralRepresentationPython` directory. 
```
git clone git@github.com:KamitaniLab/SpecVQGAN.git
git clone git@github.com:KamitaniLab/SpecVQGAN.git # あとでハッシュタグを足して固定する

```

### Download datasets and models

See [data/README.md](data/README.md).

## Usage

### Decoding analysis

We provide scripts that reproduce main results in the original paper.
Please execute the sh files in the following order.

1. Train feature decoders to predict the VGGishish features. 
```
cd EmotionVideoNeuralRepresentationPython
python feature-decoding
```

2. Using the decoders trained in step.1, perform feature predictions. (Perform the prediction for the attention task dataset at the same time.)
```
./2_test_batch.sh
```

3. Validate the prediction accuracy of predicted features.
```
./3_eval_batch.sh
```
Visualize the prediction accuracy with the following notebook. This notebook draws Fig.3D and Fig.3E of the original paper.
```
feature_decoding/makefigures_featdec_eval.ipynb
```

4. Reconstruct sound clips using predicted features.
```
./4_recon_batch.sh
```

5. Validate the quality of reconstructed sound.
```
./5_recon_eval_batch.sh 
```
Visualize the reconstruction quality with the following notebooks. These notebooks draws Fig.4C and Fig.8C of the original paper.
```
reconstruction/makefigures_recon_eval.ipynb
reconstruction/makefigures_recon_eval_attention.ipynb
```

### Encoding analysis
