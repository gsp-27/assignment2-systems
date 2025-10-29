import os
from collections import defaultdict
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
import torch
import timeit
import pandas as pd
import numpy as np

def train_one_step(model, x, y, inference_only, warm_up=False):
    times = {}
    if inference_only:
        with torch.no_grad():
            start = timeit.default_timer()
            out = model(x)
            end = timeit.default_timer()
            times["inference_forward"] = end - start
    else:
        start = timeit.default_timer()
        out = model(x)
        end = timeit.default_timer()
        times["train_forward"] = end - start
        loss = cross_entropy(out, y)
        start = timeit.default_timer()
        loss.backward()
        end = timeit.default_timer()
        times["train_backward"] = end - start
    
    return times


def benchmark():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--num-layers", type=int, default=12, help="number of transformer blocks")
    # parser.add_argument("--d-model", type=int, default=768, help="dimensionality of model")
    # parser.add_argument("--d-ff", type=int, default=3072, help="dimensionality of feedforward block")
    # parser.add_argument("--num-heads", type=int, default=12, help="number of heads")
    # parser.add_argument("--vocab-size", type=int, default=10_000, help="vocabulary size")
    # parser.add_argument("--context-len", type=int, default=512, help="seq length of each example")
    # parser.add_argument("--warmup-steps", type=int, default=10, help="number of warmup steps to perform")
    # parser.add_argument("--n-train-steps", type=int, default=100, help="number of fwd-bwd steps to perform")
    # parser.add_argument("--inference-only", action="store_true", default=False, help="do inference only")

    # args = parser.parse_args()

    vocab_size = 10000
    context_len = 512
    batch_size = 4
    # d_models = [768, 1024, 1280, 1600, 2560]
    d_models = [768, 1024, 1280]
    # d_ffs = [3072, 4096, 5120, 6400, 10240]
    d_ffs = [3072, 4096, 5120]
    # num_layers = [12, 24, 36, 48, 32]
    num_layers = [12, 24, 36]
    # num_heads = [12, 16, 20, 25, 32]
    num_heads = [12, 16, 20]
    warmup_steps = 5
    training_steps = 10
    inference_only = False
    batch_size = 4

    # make random batch of data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, context_len), device=device)
    y = torch.cat([x[:, 1:], torch.randint(low=0, high=vocab_size, size=(batch_size, 1))], dim=1, device=device)

    column_names = ["d_model", "d_ff", "num_layers", "num_heads",
        "training_forward_time_mean", "training_forward_time_std",
        "training_backward_time_mean", "training_backward_time_std",
        "inference_forward_time_mean", "inference_forward_time_std"]
    df = pd.DataFrame(columns=column_names)


    # initialize the model with the arguments
    for d_model, d_ff, num_layer, num_head in zip(d_models, d_ffs, num_layers, num_heads, strict=True):
        timings = defaultdict(list)

        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_len,
            d_model=d_model,
            num_layers=num_layer,
            num_heads=num_head,
            d_ff=d_ff,
            rope_theta=1e4,
        )
        model.to(device)
        print(f"initialization complete")


        if inference_only:
            model = model.eval()
        else:
            model = model.train()

        # perform warm-up
        if warmup_steps > 0:
            for _ in range(warmup_steps):
                if inference_only:
                    with torch.no_grad():
                        out = model(x)
                else:
                    out = model(x)
                    loss = cross_entropy(out, y)
                    loss.backward()
        if torch.cuda.device_count() > 0:
            torch.cuda.synchronize()
        
        print(f"finished warmup")
        
        # now we do the actual training
        for _ in range(training_steps):
            times = train_one_step(model, x, y, inference_only)
            if torch.cuda.device_count() > 0:
                torch.cuda.synchronize()
            
            if inference_only:
                timings["infer_fwd"].append(times["infer_forward"])
            
            else:
                timings["train_fwd"].append(times["train_forward"])
                timings["train_bwd"].append(times["train_backward"])

        if inference_only:
            ifwd = np.array(timings["infer_fwd"])
            ifwd_mean = ifwd.mean()
            ifwd_std = ifwd.std()
            tfwd_mean = None
            tfwd_std = None
            tbwd_mean = None
            tbwd_std = None
        else:
            tfwd = np.array(timings["train_fwd"])
            tbwd = np.array(timings["train_bwd"])
            tfwd_mean = tfwd.mean()
            tfwd_std = tfwd.std()
            tbwd_mean = tbwd.mean()
            tbwd_std = tbwd.std()
            ifwd_mean = None
            ifwd_std = None

        row_data = {
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layer,
            "num_heads": num_head,
            "training_forward_time_mean": tfwd_mean,
            "training_forward_time_std": tfwd_std,
            "training_backward_time_mean": tbwd_mean,
            "training_backward_time_std": tbwd_std,
            "inference_forward_time_mean": ifwd_mean,
            "inference_forward_time_std": ifwd_std,
        }

        df = pd.concat([df, pd.DataFrame(row_data, index=[0])], ignore_index=True)
    return df


if __name__ == "__main__":
    timing_df = benchmark()
    if not os.path.exists("results"):
        os.makedirs("results")
    timing_df.to_csv("results/timing_results.csv")
