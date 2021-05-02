# (Caffe and) Sugar : Python library and tools for training Caffe models.

### Description

Put sugar in your Caffe !

*Sugar* is a Python library based on Pycaffe for training Caffe models. It provides :

* a trainer class (Trainer) for training generic Caffe models and replicating the Caffe standard training tools.

* a solver class (Solver) for reading Caffe solver prototxt and configuring the training in the Caffe standard way.

* a generic class for evaluating networks (Evaluator class) and for monitoring evaluation metrics during the training.

### Installation

Sugar and its dependencies can be installed directly with pip3 as it is structured as a Python package. Don't forget to use -e option to incorporate any Sugar code modification without desinstalling/reinstalling the package.

<pre>
pip3 install -e path_to_Sugar
</pre>

Other dependencies :

* datum : see https://github.com/clacroi/Datum for installing Datum
* Caffe and PyCaffe, its Python API. See https://github.com/BVLC/caffe for more informations about installing Caffe..
** Sugar was tested with Segnet family models developped with a specific version of Caffe (https://github.com/alexgkendall/caffe-segnet). An updated version of this repository was used to install Caffe for the Segnet architecture :  https://github.com/navganti/caffe-segnet-cudnn7.

### Examples

For use examples of Sugar library, you can check :

* sugar/notebooks/sugar_tester.py (notebook)