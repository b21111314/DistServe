import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

target = '(400.0, 100.0)'


def parse_args(args_=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=target)
    parser.add_argument("--max-machine", type=int, default=4)
    parser.add_argument("--max-gpu-per-node", type=int, default=8)
    args = parser.parse_args(args_)
    return args


args = parse_args()
target = eval(args.target)
max_machine = args.max_machine
max_gpu_per_node = args.max_gpu_per_node

Path("figure").mkdir(exist_ok=True)
Path("visual").mkdir(exist_ok=True)

# Get all files with format '*.latency.csv' from root_dir
# root_dir = Path("fig11-abalation-log")
root_dir = Path("result")
latency_file_paths = sorted(list(root_dir.glob("*.latency.csv")))
columns = ['backend', 'rate', 'target', 'attainment', 'latency']

dfs = []
namespaces = []
for latency_file_path in latency_file_paths:
    try:
        df = pd.read_csv(latency_file_path)
        dfs.append(df)
    except pd.errors.EmptyDataError:
        pass

big_df = pd.concat(dfs, ignore_index=True)
big_df['ngpu'] = big_df['tp_prefill'] * big_df['pp_prefill'] + big_df['tp_decode'] * big_df['pp_decode']
big_df['per_gpu_rate'] = big_df['rate'] / big_df['ngpu']
big_df['goodput@90'] = big_df.apply(
    lambda x: x['rate'] / x['ngpu'] if x['attainment'] >= 90 else 0,
    axis=1,
)

model_type = big_df['model_type'].unique()[0]


def can_fit_low_affinity_distserve(x):
    a, b, c, d = x['tp_prefill'], x['pp_prefill'], x['tp_decode'], x['pp_decode']
    for pp_common in range(1, max_machine + 1):
        bp = b / pp_common
        dp = d / pp_common
        # If either bp or dp is not int, skip
        if int(bp) != bp or int(dp) != dp:
            continue
        # Check if the segment can be placed inside a node
        if a * bp + c * dp <= max_gpu_per_node:
            return True
        pass
    return False


def can_fit_low_affinity(x):
    if x['backend'] == 'distserve':
        return can_fit_low_affinity_distserve(x)
    else:
        return True
    pass


big_df['low_affin'] = big_df.apply(can_fit_low_affinity, axis=1)

big_df.sort_values(by=['backend', 'per_gpu_rate', 'tp_prefill', 'pp_prefill', 'tp_decode', 'pp_decode'])

big_df['target_evaled'] = big_df['target'].apply(eval)
figure_11_left_df = big_df[
    (big_df['pd'] == 'both')
    & (big_df['target_evaled'] == target)
    ].copy()

figure_11_left_df = figure_11_left_df.sort_values(by=[
    'backend', 'tp_prefill', 'pp_prefill', 'tp_decode', 'pp_decode',
    'rate'
])
# Choose the config with the best goodput in each group
figure_11_distserve_high = figure_11_left_df[
    (figure_11_left_df['backend'] == 'distserve')
]
figure_11_distserve_low = figure_11_left_df[
    (figure_11_left_df['backend'] == 'distserve')
    & (figure_11_left_df['low_affin'])
    ]
figure_11_vllm_high = figure_11_left_df[
    (figure_11_left_df['backend'] == 'vllm')
]

figure_11_vllm_low = figure_11_left_df[
    (figure_11_left_df['backend'] == 'vllm')
    & (figure_11_left_df['pp_prefill'] == 1)
    # & (figure_11_left_df['tp_prefill'] == __tp)
    ]


# Find the best config that has the highest goodput@90 and attainment
def get_top_config(df):
    max_per_gpu_rate = max(df['per_gpu_rate'].unique())
    df2 = df[df['per_gpu_rate'] == max_per_gpu_rate]
    df3 = df2.sort_values(by=['goodput@90', 'attainment'], ascending=False, )
    r = df3.iloc[0][[
        "tp_prefill",
        "pp_prefill",
        "tp_decode",
        "pp_decode",
    ]]
    return r


def add_matplotlib_trace(fig, df: 'DataFrame', trace: str):
    tp_prefill, pp_prefill, tp_decode, pp_decode = get_top_config(df)
    config_df = df[
        (df['tp_prefill'] == tp_prefill) & (df['pp_prefill'] == pp_prefill) &
        (df['tp_decode'] == tp_decode) & (df['pp_decode'] == pp_decode)
        ]
    name = trace
    fig.plot(
        config_df['per_gpu_rate'], config_df['attainment'],
        label=name,
        marker='o',
    )
    return config_df['attainment'].tolist()


# Plot a line chart with 4 curves
# x-axis: per_gpu_rate
# y-axis: attainment

fig, ax = plt.subplots()
a = add_matplotlib_trace(ax, figure_11_distserve_high, "disthigh")
b = add_matplotlib_trace(ax, figure_11_distserve_low, "distlow")
c = add_matplotlib_trace(ax, figure_11_vllm_high, "vllm++")
d = add_matplotlib_trace(ax, figure_11_vllm_low, "vllm")

data_points = {
    "dist++": a,
    "dist": b,
    "vllm++": c,
    "vllm": d,
}
with open("figure/figure_11a.json", "w") as f:
    import json

    json.dump(data_points, f)
