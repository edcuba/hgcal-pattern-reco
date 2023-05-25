# Pattern recognition for HGCAL reconstructions

This repository documents the AI approach to trackster smoothing for CMS HGCAL reconstructions, developed in the scope of a master's thesis in [Pattern recognition for particle shower reconstruction](https://www.merlin.uzh.ch/publication/show/23612) by [Eduard Cuba](mailto:eduard.cuba@uzh.ch). The content is distilled from the [original repository](https://github.com/edcuba/TICLPatternReco), covering data preprocessing and model training.

## Pipeline

The pipeline includes:
- Preprocessing ntuplized ROOT files into a PyTorch dataset
    - selecting tracksters of interest
    - finding tracksters in their neighborhood
    - feature extraction
- Training the model
    - training
    - validation
    - evaluation
- Export to ONNX (MLP model)
- Integration into CMSSW (MLP model)

## Setup

This work is organized as a collection of Jupyter Notebooks, one for each model, supported by a small Python library (`reco`).
The training configuration, such as dataset routes and model parameters, are adjusted directly in the Jupyter Notebooks.

The project uses a `virtualenv` setup and requires Python 3.7 or higher.
Several Python libraries are required, including `torch`, `scikit-learn`, and `uproot`. To avoid compatibility issues, use a clean virtual environment and install the requirements defined in `requirements.txt` as follows:
```
virtualenv ve                     # initialize the environment
. ve/bin/activate                 # enter the environment
pip install torch                 # torch needs to be installed first
pip install -r requirements.txt   # install the remaining project requirements
```

## Data preprocessing

The Notebooks can work directly with n-tuplized ROOT files, which need to be put into a folder defined in the notebooks (`raw_dir`).
Before the model training, the events are extracted and stored in PyTorch dataset files. This process uses the wrapper classes `reco.dataset_pair.TracksterPairs` and `reco.dataset_graph.TracksterGraph`.
The dataset wrappers will process a specified number of ROOT files and save the content into a PyTorch dataset in the `data_root` folder defined in the notebook.
The configuration of each dataset is embedded in the filename, and if a preprocessed version of the given dataset configuration is already available, it will be loaded directly.
If preprocessed datasets are available, the preprocessing can be skipped by placing them in the `data_root` folder.

### Feature extraction

During preprocessing, trackster pairs or trackster graphs are described by features.
In the pairwise case, each pair is described by a total of 63 values [detailed here](https://github.com/edcuba/cmssw/blob/CMSSW_12_6_0_pre3_MLP_smoothing_with_ntuplizer/RecoHGCal/TICL/plugins/SmoothingAlgoByMLP.cc#L196). This includes most of the trackster features available in the ROOT files (for both tracksters), plus their pairwise distance, and z-coordinates of the first and last layer-cluster along the z-axis.
In the graph case, the feature vector is extended by a set of structural features describing the trackster composition; see `reco.features`.

### Batching

The datasets are randomly split into a training and validation set with a ratio of 90:10. A `DataLoader` object handles batching. In the graph case, it is crucial to consider that the graphs in the batches are flattened into one large graph, and all operations on the set of nodes in the graph (such as aggregations or nearest-neighbor search) need to respect the `data.batch` node assignment.


## Models and CMSSW integration

Neural network components can be integrated into CMSSW via the [ONNX format](https://onnx.ai/).
This work compared standard feed-forward neural networks (MLP) applied to trackster pairs and graph neural networks (GNNs) applied to entire events or calorimeter regions (typically a cylinder defined around a selected trackster).
GNNs were shown to outperform the MLP-based models by a small margin; however, due to lower complexity, only the MLP architecture was used to demonstrate [the CMSSW integration](https://github.com/edcuba/cmssw).
The integration of GNNs is possible, yet the dynamic nearest neighbor selection used in dynamic graph convolutional networks (DGCNN) implemented via a `torch-cluster` module is not currently supported by ONNX and needs to be implemented manually in CMSSW.

### Pretrained artifacts

The directory `models` contains serialized pre-trained models for each case. Using the default configuration, you can skip the training cell in the notebooks and directly load the pre-trained models.
