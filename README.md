# Pattern recognition for HGCAL reconstructions

This repository documents the AI approach to trackster smoothing for CMS HGCAL reconstructions, developed in the scope of a master's thesis in [Pattern recognition for particle shower reconstruction](https://www.merlin.uzh.ch/publication/show/23612) by [Eduard Cuba](mailto:eduard.cuba@uzh.ch). The content is distilled from the [original repository](https://github.com/edcuba/TICLPatternReco), covering data preprocessing and model training.

## Models and CMSSW integration

Neural network components can be integrated into CMSSW via the [ONNX format](https://onnx.ai/).
This work compared standard feed-forward neural networks (MLP) applied to trackster pairs and graph neural networks (GNNs) applied to entire events or calorimeter regions (typically a cylinder defined around a selected trackster). GNNs were shown to outperform the MLP-based models by a small margin; however, due to lower complexity, only the MLP architecture was used to demonstrate [the CMSSW integration](https://github.com/edcuba/cmssw). The integration of GNNs is possible, yet the dynamic nearest neighbor selection used in dynamic graph convolutional networks (DGCNN) implemented via a `torch-cluster` module is not currently supported by ONNX and needs to be implemented manually in CMSSW.

## Pipeline

The pipeline includes:
- preprocessing ntuplized ROOT files into a PyTorch dataset
- training the model
    - evaluation
- export to ONNX
- integration into CMSSW

## Technical guide

This work is organized as a collection of Jupyter Notebooks supported by a small Python library.
Training configuration, such as dataset routes and model parameters are adjusted directly in the Jupyter Notebooks.

### Environment setup

The project is using a `virtualenv` setup and requires Python 3.7 or higher.
Several python libraries are required, including `torch`, `scikit-learn`, and `uproot`. To avoid compatibility issues, use a clean virtual environment and install the requirements defined in `requirements.txt` as follows:
```
virtualenv ve       # initialize the environment
. ve/bin/activate   # enter the environment
pip install -r requirements.txt     # install the project requirements
```

### Model training
