# Semi-Supervised Learning of Deep Neural Networks

Implementation of the Master's thesis at FIT CTU.

## Dependencies
* numpy
* tensorflow
* keras
* scikit-learn
* matplotlib

## Usage
`moons.py`

Implemented methods can be tested on a simple two moons dataset using `moons.py` script
The resulting model and the decision border plot is saved by default to the 'output' folder.

`malware.py`

Example of usage with a real-world malware dataset. It requires setting command line arguments
as listed in `lib/utils.py`. More importantly it needs the input malware dataset, that can not be included
with the thesis.