[Anonymize Audio Notebook](https://github.com/gbibbo/voice_anonymization/blob/master/Anonymize.ipynb)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gbibbo/voice_anonymization/blob/master/Anonymize.ipynb)


# Voice Anonymization for Sound Event Detection

This repository contains the resources and Jupyter Notebook developed as part of the research paper: "The Sounds of Home: A Speech-Removed Residential Audio dataset for Sound Event Detection," authored by Thomas Deacon, Gabriel Bibbó, Arshdeep Singh, and Mark D. Plumbley from the Center for Vision, Speech and Signal Processing, University of Surrey, UK.

## Overview

The project focuses on developing a privacy-compliant audio dataset to support sound event detection research aimed at promoting wellbeing for older adults in smart home environments. The core of this research involves an automated speech removal pipeline that detects and eliminates spoken voice segments while preserving other sound events within residential audio recordings.

## Repository Content

- **Anonymize.ipynb**: An interactive Jupyter Notebook that demonstrates the speech removal process used in our research. This notebook allows users to upload their audio files and apply our pre-trained models to anonymize the audio by removing speech segments.

- **Scripts**: Includes all the Python scripts used for audio processing and sound event detection as detailed in our research.

- **Data**: Sample data files used for demonstration in the notebook.

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


## Cite
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "Panns: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

## Reference
[2] Gemmeke, J.F., Ellis, D.P., Freedman, D., Jansen, A., Lawrence, W., Moore, R.C., Plakal, M. and Ritter, M., 2017, March. Audio set: An ontology and human-labeled dataset for audio events. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 776-780, 2017

[3] Hershey, S., Chaudhuri, S., Ellis, D.P., Gemmeke, J.F., Jansen, A., Moore, R.C., Plakal, M., Platt, D., Saurous, R.A., Seybold, B. and Slaney, M., 2017, March. CNN architectures for large-scale audio classification. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 131-135, 2017

## External links
Other work on music transfer learning includes: <br>
https://github.com/jordipons/sklearn-audio-transfer-learning <br>
https://github.com/keunwoochoi/transfer_learning_music
