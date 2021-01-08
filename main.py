import torch
import sys
import time
import argparse

from torchvision import datasets, models, transforms
from transformers import AutoTokenizer, AutoModel # BERT
from transformers import GPT2Tokenizer, GPT2Model # GPT2

def average_90_percent(l):
    # average over iterable
    # check only last ~90% of elements (warmup)
    if (len(l) < 10):
        print("average(): I need at least 10 items to work with!")
        return 0
    better_l = l[int(len(l)/10):]
    return sum(better_l) / len(better_l)

def sec_to_ms(sec):
    # a.bcdefghijk... --> "abcd.efg"
    return str(int(sec * 10**6) / 10**3)

def main():
    print("MODEL NAME, \tBATCH SIZE,\tAVG LATENCY (ms),\tAVG MEM USAGE (MB)")
    #parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_inference', type=int)
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()
    model_name = args.model_name
    num_inference = args.num_inference
    batch_size = args.batch_size
    # stores inference latency values (unit: sec)
    l_inference_latency = list()
    # call corresponding DNN model...
    # TODO: ADD RECSYS MODEL!
    if (model_name == "resnet"):
        with torch.no_grad():
            model = models.resnet18(True, True)
            # inference
            for i in range(num_inference):
                # input
                empty_tensor = torch.zeros(batch_size, 3, 224, 224)
                start_time = time.time()
                model(empty_tensor)
                torch.cuda.synchronize()
                end_time = time.time()
                l_inference_latency.append(end_time - start_time)
            str_avg_inf_time = sec_to_ms(average_90_percent(l_inference_latency))
            print(",".join(["RESNET18", str(batch_size), str_avg_inf_time]))

    elif (model_name == "mobilenet"):
        with torch.no_grad():
            model = models.mobilenet_v2(True, True)
            # warmup
            for i in range(num_inference):
                empty_tensor = torch.zeros(batch_size, 3, 224, 224)
                start_time = time.time()
                model(empty_tensor)
                torch.cuda.synchronize()
                end_time = time.time()
                l_inference_latency.append(end_time - start_time)
            str_avg_inf_time = sec_to_ms(average_90_percent(l_inference_latency))
            print(",".join(["MOBILENET_V2", str(batch_size), str_avg_inf_time]))

    elif (model_name == "bert"):
        with torch.no_grad():
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased")
            # inference
            for i in range(num_inference):
                texts = ["inference tokens!"] * batch_size
                inputs = tokenizer(texts, return_tensors="pt")
                start_time = time.time()
                outputs = model(**inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                l_inference_latency.append(end_time - start_time)
            str_avg_inf_time = sec_to_ms(average_90_percent(l_inference_latency))
            print(",".join(["BERT-BASE-UNCASED", str(batch_size), str_avg_inf_time]))

    elif (model_name == "gpt2"):
        with torch.no_grad():
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2Model.from_pretrained("gpt2")
            # inference
            for i in range(num_inference):
                texts = ["these are some pretty long inference tokens!"] * batch_size
                inputs = tokenizer(texts, return_tensors="pt")
                start_time = time.time()
                outputs = model(**inputs)
                torch.cuda.synchronize()
                end_time = time.time()
                l_inference_latency.append(end_time - start_time)
            str_avg_inf_time = sec_to_ms(average_90_percent(l_inference_latency))
            print(",".join(["GPT2", str(batch_size), str_avg_inf_time]))
    else:
        print("Unidentified model name: {}".format(model_name))
        return
# cf) allocated memory: torch.cuda.memory_allocated() but this keeps returning 0


if __name__ == "__main__":
    main()

