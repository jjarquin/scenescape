# Robot Vision

Algorithms for sensor fusion, environment perception, object detection and tracking in C++ with Python interface


# Build Dependencies
```bash
sudo apt install build-essential pkg-config python3 python3-setuptools python3-wheel cmake python3-dev googletest libpython3-dev
```

# Install Dependencies
```bash
sudo apt install pybind11-dev libopencv-dev libeigen3-dev libpcl-dev libtbb-dev libomp-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev
```

# Installation

```bash
python3 setup.py bdist_wheel
pip3 install dist/robot_vision-X.X.X-cpXX-cpXXm-linux_x86_64.whl
```

## Documentation

### Html documentation
Install necessary packages
```bash
pip3 install sphinx, sphinx-rtd-theme
```


then build the documentation with:
```bash
make docshtml
```

after that you can launch the doc server with
```bash
make docserve
```

To access the remote server from your windows machine, activate the port forwarding via ssh
```bash
ssh -N -f -L localhost:8000:localhost:8000 userid@machine
```

Access the documentation at http://localhost:8000

### PDF documentation

Install latexmk


```bash
sudo apt install latexmk
```
Call the make command

```bash
make docspdf
```

The pdf will be generated in docs/_build/latex/robot_vision.pdf


## How to test Robot vision
We'll start with cloning the scenescape branch from the robot_vision repo.

Clone robot_vision repo

```bash

```

Create base docker
We will use the ubuntu 22.04 docker image as base and install the required dependencies

```bash
touch Dockerfile
```

Copy dockerfile contents

```bash
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update && apt-get install --no-install-recommends -y git wget\
                        python3-dev python3-distutils python3-pip build-essential cmake libboost-all-dev\
                        googletest pybind11-dev libpython3-dev libopencv-dev libeigen3-dev && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade setuptools==58.2.0 wheel

COPY robot-vision /home/robot-vision
WORKDIR /home/robot-vision

RUN python3 -m pip install -r requirements.txt

CMD ["/bin/bash"]
```

Build base docker image

```bash
docker build -t robot_vision:latest . -f Dockerfile
docker run -it robot_vision
```

The docker container starts in interactive mode at the /home/robot-vision folder

Run Python tests

```bash
python3 setup.py test
```
A valid output will look like this:

[100%] Built target tracking

==========test session starts ==============

platform linux -- Python 3.10.12, pytest-7.4.3, pluggy-1.3.0
rootdir: /home/robot-vision
collected 8 items                                                                                                                                                                                                                         
python/test/tracking_test.py ........                                                                                                                                                                                                      [100%]


========== 8 passed in 0.13s ==============
Run CMake tests
```bash
mkdir build_cmake && cd build_cmake
cmake ../ -DBUILD_TESTING=ON
make
```

Execute tests

```bash
./test/RobotVisionTests
```

A Valid output will look like this:

[----------] 8 tests from MultipleObjectTrackerTest (6626 ms total)

[----------] Global test environment tear-down
[==========] 8 tests from 1 test suite ran. (6626 ms total)
[  PASSED  ] 8 tests.
