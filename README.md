# DeepCrossCancer: a Deep Learning Framework for Discovering Cross-cancer Patients with Similar Transcriptomic Profiles
DeepCrossCancer provides a deep-learning method for discovering cross-cancer patients with similar transcriptomic profiles. It is implemented by deep learning library Keras and Tensorflow-GPU. DeepCrossCancer provides clustering of cancer patients, prediction of cross-cancer patients, and statistical tests to analyze cross-cancer patients in terms of other genomic data. The clustering part is inspired from https://github.com/runpuchen/DeepType which is a deep learning framework for prediction breast cancer subgroups.
# Installation
•	Download DeepCrossCancer by
```
git clone https://github.com/dyguay/DeepCrossCancer
```
•	Since the package is written in python 3.5, [python 3.5](https://www.python.org/downloads/) with the pip tool must be installed first. DeepCrossCancer uses the following dependencies: numpy, scipy, pandas, talos, sklearn, keras=2.1.6, tensorflow-gpu=1.12.0, shap, statsmodels. You can install these packages first, by the following commands:
```python
pip install pandas
pip install numpy
pip install scipy
pip install git+https://github.com/autonomio/talos
pip install -U scikit-learn
pip install -v keras==2.1.6
pip install -v tensorflow==1.12.0
pip install shap
pip install statsmodels
```
•	For the visualization, it uses the following dependencies:
```python
pip install matplotlib
pip install seaborn
pip install plotly
pip install holoviews
pip install bokeh
```
