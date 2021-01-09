#!/bin/bash
echo "==================================================================="
echo "==== STARTING: PyTorch Inference Profiling - Batch Size Sweep ====="
echo 
# to take in user input, simply use $1, $2, etc.
# but it's not necessary here :)

outfiledir="profile-results/"
num_inference=10
outfilename=$outfiledir"batch-sweep-"$num_inference"-inferences.csv"

# erase previously existing file
rm -f $outfilename
# add number number order: '>' is just redirecting to file, '>>' is to APPEND TO END OF FILE!
echo "MODEL NAME, BATCH SIZE, AVG LATENCY (ms), AVG MEM USAGE (MiB)" >> $outfilename

for model in "resnet" "mobilenet" "bert" "gpt2"
do
    for batch_size in 1 2 4 8 16 32 64
    do
        python main.py --model_name $model --num_inference $num_inference --batch_size $batch_size >> $outfilename
    done
    echo ">>>>>>>>>>>>>>>>>>>>>>>>> STATUS UPDATE: MODEL "$model" FINISHED"
done

echo 
echo "==== FINISHING: PyTorch Inference Profiling - Batch Size Sweep ===="
echo "==================================================================="