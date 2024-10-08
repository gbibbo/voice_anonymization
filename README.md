# The Sounds of Home Dataset

The complete dataset (197 GB) can be downloaded directly from the CVSSP server:

[![Download](https://img.shields.io/badge/download-197GB-green.svg)](https://www.cvssp.org/data/ai4s/sounds_of_home/VITALISE_data_light.zip)

Alternatively, the dataset is available through mirrors across four Zenodo repositories:

- **Folders 1-7**: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12737915.svg)](https://doi.org/10.5281/zenodo.12737915)
- **Folders 8-10**: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12800789.svg)](https://doi.org/10.5281/zenodo.12800789)
- **Folders 11-13**:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12802632.svg)](https://doi.org/10.5281/zenodo.12802632)
- **Folder 14**:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12802716.svg)](https://doi.org/10.5281/zenodo.12802716)

The dataset consists of 1,344 one-hour recordings obtained in the homes of 8 participants in Belgium during the year 2023. In each participant's home, an [AudioMoth](https://www.openacousticdevices.info/audiomoth) was installed in the Living Room and another one in the Kitchen, recording simultaneously. Subsequently, the anonymization sorfware presented below was used to identify the spoken voice and remove it. In addition, the 7 best estimates of sound events calculated using PANNs (the model [Cnn14_DecisionLevelAtt_mAP=0.425](https://zenodo.org/records/3987831)) for each audio fame of 0.021 seconds are provided.

The dataset is segmented into 14 folders, with each set of two consecutive folders representing recordings from the same house. The dataset has been distributed across three separate Zenodo repositories due to its size.

# Voice Anonymization for Sound Event Detection

This repository contains the resources and Jupyter Notebook developed as part of the research paper: "The Sounds of Home: A Speech-Removed Residential Audio dataset for Sound Event Detection," authored by Thomas Deacon, Gabriel Bibbó, Arshdeep Singh, and Mark D. Plumbley from the Center for Vision, Speech and Signal Processing, University of Surrey, UK.

## Overview

The project focuses on developing a privacy-compliant audio dataset to support sound event detection research aimed at promoting wellbeing for older adults in smart home environments. The core of this research involves an automated speech removal pipeline that detects and eliminates spoken voice segments while preserving other sound events within residential audio recordings.

## Repository Content

- **Anonymize.ipynb**: An interactive Jupyter Notebook that demonstrates the speech removal process used in our research. This notebook allows users to upload their audio files and apply our pre-trained models to anonymize the audio by removing speech segments.

- **Scripts**: Includes all the Python scripts used for audio processing and sound event detection as detailed in our research.

- **Data**: Sample data files used for demonstration in the notebook.

## Visualization of Speech Removal

The image below provides a visualization of the speech removal process applied to a sample audio recording. This visualization consists of three main components:

1. **Audio Spectrogram**: The top panel shows the spectrogram of the original audio recording. Here, you can observe various sound events, including speech, as indicated by the brighter areas which represent higher energy or intensity at various frequencies.

2. **Sound Event Predictions**: The middle panel displays the probabilities of different sound events detected by the model. Each color represents a different type of sound, such as male speech, narration, or other noises like clicking. This graph helps in identifying the temporal occurrence and likelihood of different sound events.

3. **Spectrogram with Speech Removed**: The bottom panel shows the spectrogram of the audio after the speech has been removed. Notice the significant reduction in intensity in areas where speech was previously detected, indicating the effectiveness of the speech removal process.

![Visualization of Speech Removal](https://github.com/gbibbo/voice_anonymization/blob/master/metadata/black_list/speech_removed.png)

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.6+
- Jupyter Notebook or JupyterLab
- Required Python libraries: `librosa`, `numpy`, `matplotlib`, and others as listed in `requirements.txt`.

### Installation

Clone this repository to your local machine using:

```bash
git clone https://github.com/gbibbo/voice_anonymization.git
```

Navigate to the cloned directory and install the necessary Python packages:
```python
pip install -r requirements.txt
```

### Using the Notebook

To use the Anonymize.ipynb notebook:

Open the notebook in Jupyter Notebook or JupyterLab.
Follow the instructions within the notebook to upload your audio file and run the cells to process the audio.
You can access the notebook directly in Google Colab for an interactive experience without any local setup:

Anonymize Audio Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gbibbo/voice_anonymization/blob/master/Anonymize.ipynb)

### Citation

If you use the data or the anonymization pipeline in your research, please cite our paper:

@inproceedings{bibbo2024sounds,
  title={The Sounds of Home: A Speech-Removed Residential Audio dataset for Sound Event Detection},
  author={Bibbò, Gabriel and Deacon, Thomas and Singh, Arshdeep and Plumbley, Mark D.},
  booktitle={INTERSPEECH 2024},
  year={2024},
  organization={University of Surrey}
}

### Acknowledgements

This work was supported by the Engineering and Physical Sciences Research Council (EPSRC) under Grant EP/T019751/1 "AI for Sound (AI4S)."
