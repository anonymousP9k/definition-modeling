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
