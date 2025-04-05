<h1>"Interpretable approach to detecting semantic changes based on generated definitions"</h1>

This is the repository for the paper "Interpretable approach to detecting semantic changes based on generated definitions"

## Components

The repository consists of the following components:

* [mas_parser](mas_parser/README.md) - a module for parsing the [MAS](Small Academic Dictionary) dataset.
* [model](model/README.md) - a module for training, evaluation and inference of the FRED-T5-1.7B model.
  * [Model training](model/docs/Model%20training.md) - split the dataset and train the model.
  * [Inference](model/docs/Inference.md) - generate definitions for the dataset.
  * [Evaluation](model/docs/Evaluation.md) - evaluate the model.
* [rushifteval](rushifteval/README.md) - a module for evaluating the model on the RuShiftEval dataset.
* [ruscorpora](ruscorpora/README.md) - a module for qualitative testing of the project.
* [vizvector](vizvector/README.md) - a module for visualizing the semantic changes.

## Steps to reproduce

To reproduce the results, follow the steps below. Steps can be skipped if you download the corresponding data.

1. The instructions for parsing MAS dataset are in the
[README](mas_parser/README.md) of the `mas_parser` module.
2. The instructions for training the model, initial testing and inference are in the
[README](model/docs/Model%20training.md) of the `model` module.
3. The instructions for evaluating the model with `rushifteval` are in the
[README](rushifteval/README.md) of the `rushifteval` module.
4. The instructions for working with RNC (visualizing the words' meanings) are in the
[README](ruscorpora/README.md) of the `ruscorpora` module.
