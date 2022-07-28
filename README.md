# boring-pl-model

Simple pytorch lightning boring model. Runs a resnet model on random junk data and optimizes a loss. Batches are cached to the GPU so that there's no data loading overhead. 

# Installation

 `pip install -r requirements.txt`
 
 # Running
 
 From the root folder execute

 `python trainer/train_boring_model.py`

Iterations per second on the tqdm progress should indicate the speed.
