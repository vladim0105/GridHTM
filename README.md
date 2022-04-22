# GridHTM
This repo depends on htm.core (https://github.com/htm-community/htm.core)
### Cloning and Setup
To clone the repo and htm.core:
```commandline
git clone https://github.com/vladim0105/GridHTM
git submodule update --init --recursive
```
To build and install htm.core, which requires cmake and more (see https://github.com/htm-community/htm.core):

(default)
```commandline
cd htm.core/
python setup.py install --user --force
```
(Anaconda)
```commandline
cd htm.core/
python setup.py install
```
### Running
To run:
```commandline
python main.py segmentedVideoFile settingsFile -o outputName
```
Help:
```commandline
python main.py -h
```