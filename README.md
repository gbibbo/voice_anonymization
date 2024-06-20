[Anonymize Audio Notebook](https://github.com/gbibbo/voice_anonymization/blob/master/Anonymize.ipynb)





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
```

```python
Navigate to the cloned directory and install the necessary Python packages:
```

### Using the Notebook

To use the Anonymize.ipynb notebook:

Open the notebook in Jupyter Notebook or JupyterLab.
Follow the instructions within the notebook to upload your audio file and run the cells to process the audio.
You can access the notebook directly in Google Colab for an interactive experience without any local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gbibbo/voice_anonymization/blob/master/Anonymize.ipynb)

### Citation

If you use the data or the anonymization pipeline in your research, please cite our paper:

@inproceedings{deacon2022sounds,
  title={The Sounds of Home: A Speech-Removed Residential Audio dataset for Sound Event Detection},
  author={Deacon, Thomas and Bibbò, Gabriel and Singh, Arshdeep and Plumbley, Mark D.},
  booktitle={INTERSPEECH 2022},
  year={2022},
  organization={University of Surrey}
}

### Acknowledgements

This work was supported by the Engineering and Physical Sciences Research Council (EPSRC) under Grant EP/T019751/1 "AI for Sound (AI4S)."
