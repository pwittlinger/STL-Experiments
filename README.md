This repository contains experiments done on the data set from Wibisono et. al. https://www.sciencedirect.com/science/article/pii/S2352340920308180
using the library STLCG https://github.com/StanfordASL/stlcg

If you want to reproduce the results, you can use the provided Dockerfile.  (Link to Container pending)
Just be sure to save the files which are being output (e.g by mounting a local folder to the container)

docker -run -v <path/to/local>:/outputs/ <containerName>
