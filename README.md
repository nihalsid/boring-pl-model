# boring-pl-model

Simple pytorch lightning boring model. Runs a resnet model on random junk data and optimizes a loss. Batches are cached to the GPU so that there's no data loading overhead. 


# Installation

### With docker
Build the image

```
docker build --build-arg USER_ID=<Your unix id> --build-arg USER_NAME=<unix username> --platform=linux/x86_64 --tag <image tag> .
```

### Without docker

Install the requirements through pip. Additionally, you'd need to install `torch` and `torchvision`.
 ```
 pip install -r requirements.txt
 ```
 
# Running

### With docker

Standard docker run to execute the benchmark
```
docker run --user <username> --privileged --gpus all -it --rm <image tag>
```

### Without docker
From the root folder execute

`python trainer/train_boring_model.py`

# Performance

Iterations per second on the tqdm progress should indicate the speed. Further, the profiling logs from pytorch-lightning are dumped in the `<repo_root>/logs/lightning_logs/version_0/fit-run.pprof.txt` file. Not the `version_0` is the log folder for the first run, and each subsequent run increments the version number.

The four important lines to look out in the profiler are:

```
|  Action                                                 |  Mean duration (s)	|  Num calls      	|  Total time (s) 	|  Percentage %   	|
---------------------------------------------------------------------------------------------------------------------------------------------
|  [LightningModule]BoringModel.optimizer_step            |  0.047357       	|  4096           	|  193.98         	|  95.762         	|
|  [Strategy]SingleDeviceStrategy.backward                |  0.0214         	|  4096           	|  87.655         	|  43.274         	|
|  [Strategy]SingleDeviceStrategy.training_step           |  0.019809       	|  4096           	|  81.137         	|  40.056         	|
|  [LightningModule]BoringModel.optimizer_zero_grad       |  0.0025156      	|  4096           	|  10.304         	|  5.0868         	|
```

The above output shows the numbers on a machine with the following specs:

```
AMD Ryzen Threadripper PRO 3975WX 32-Cores
128G Memory
NVIDIA 3080Ti GPU
```