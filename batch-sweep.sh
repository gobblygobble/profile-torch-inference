#!/bin/bash
echo "==================================================================="
echo "==== STARTING: PyTorch Inference Profiling - Batch Size Sweep ====="
echo 

# user input with $1, $2, etc.
if [ "${1}" == "gpu" ]; then
    dev="gpu"
else
    dev="cpu"
fi

outfiledir="profile-results"
num_inference=1000
outfilename=$outfiledir"/"$dev"-batch-sweep-"$num_inference"-inferences.csv"

# erase previously existing file
rm -f $outfilename
# add number number order: '>' is just redirecting to file, '>>' is to APPEND TO END OF FILE!
echo "MODEL NAME, BATCH SIZE, AVG LATENCY (ms), AVG MEM USAGE (MiB)" >> $outfilename

for model in "resnet18" "wide_resnet101_2" "mobilenet" "bert" "gpt2" #"dlrm"
do
    for batch_size in 1 2 4 8 16 32 64
    do
        if [ "${dev}" == "gpu" ]; then
            python main.py --model_name $model --num_inference $num_inference --batch_size $batch_size --gpu >> $outfilename
            #echo "$outfilename $dev $model_name $num_inference $batch_size"
        else
            python main.py --model_name $model --num_inference $num_inference --batch_size $batch_size >> $outfilename
            #echo "$outfilename $dev $model_name $num_inference $batch_size"
        fi
    done
    echo ">>>>>>>>>>>>>>>>>>>>>>>>> STATUS UPDATE: MODEL "$model" FINISHED"
done

echo 
echo "==== FINISHING: PyTorch Inference Profiling - Batch Size Sweep ===="
echo "==================================================================="