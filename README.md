# profile-torch-inference
Profiling inference latency and memory usage for inference jobs run on PyTorch  

## How to use
- for batch sweep tests, run `. batch-sweep.sh gpu` in terminal to run with GPU, and `. batch-sweep.sh cpu` to run on CPU.
- for single runs, run `python main.py --model_name resnet18 --num_inference 10 --batch_size 8 --gpu` to run the model RESNET18 with batch size of 8 on GPU with 10 inferences in total.

## Statistics
- Statistics are printed in terminal (`stdout`) if you're doing single runs, and in files under `profile-results/` if you're running sweep tests.
- Note that number of inferences should be greater than or equal to 10, since the first ~10% of runs will be omitted in the statistics.


Copyright © Jinha Chung, KAIST School of Electrical Engineering
