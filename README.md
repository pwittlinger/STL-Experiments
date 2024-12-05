# About
This repository contains experiments done on the data set from Wibisono et. al. [Science Direct](https://www.sciencedirect.com/science/article/pii/S2352340920308180)
using the library [STLCG](https://github.com/StanfordASL/stlcg).

## Declare2STL Parser




## Usage

If you want to reproduce the exact results, you can use the provided Dockerfile, or run the Image under https://hub.docker.com/r/pwittlinger/declarestlexperiments
Be sure to save the files which are being output by mounting a local folder to the container. It contains all necessary dependencies and the relevant data.

```
docker -run -v <path/to/local>:/outputs/ declarestlexperiments
```

If you would instead like to try some other constraints you can Edit the file _Constraints.txt_. 
However, you will also need to edit the Code in _stlExperiments.py_ to match the corresponding input to the Constraints (cf. line 190)
