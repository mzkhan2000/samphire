# RelSifter: Scoring Triples for Type-Like Relations 

## Team: **samphire** at WSDM Cup 2017 held at Cambridge, UK.

## Pre-requisites
Download a .zip file from [wsdm-cup-2017-models](https://iu.box.com/s/impsevvpr3xsmcpdm8n36ame8msq8iew) containing all the following:
* knowledge graph
* machine learned models for profession and nationality

## Getting started
1. Clone this repository and cd into it.
2. Under the directory ``relsifter``, place the uncompressed directory ``wsdm-cup-2017-models`` and rename it to ``model``.

## Installing RelSifter
Navigate to the root directory and run the following command. This may take a while.
```bash
    python setup.py install
```

## Using RelSifter
Once installed, the following command can be used to run RelSifter. This will create an output file with the same name in the directory specified by the output flag.
```bash
    relsifter -i input.txt -o ./
```

## Development mode
Start with installing RelSifter in development mode to experiment with extracting features and building models for predicting relevance scores for type-like relations.
```bash
    python setup.py develop
```

### Generating features for learning
1. TF-IDF features: Navigate to ``relsifter/characterization`` and use the ``compute_pertinence.py`` module to compute combined pertinence. 
2. Text based features: Navigate to ``relsifter/textprofile`` and use the ``feature_extraction.py`` module to compute Wikipedia abstracts-based features. 

### Building machine learning models
1. TF-IDF based model: Navigate to ``relsifter/characterization`` and use the ``model_building.py`` module to train RandomForest, Adaboost and/or Ordinal Logistic Regression.
2. Text based model: Navigate to ``relsifter/textprofile`` and use the ``model_building`` module to build RandomForest, Adaboost and/or Ordinal Logistic Regression.

## Acknowledgments

Fabian Pedregosa-Izquierdo. Feature extraction and supervised learning on fMRI : from practice to theory. Medical Imaging. Université Pierre et Marie Curie - Paris VI, 2015. English. Github repository: [mord](https://github.com/fabianp/mord)