# %%
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "notebook"
import plotly.graph_objects as go
import numpy as np
import torch

# %%
num_points = 20
exponent_base = 3
vals_fp32 = [0 for i in range(num_points)]
means_fp32 = [0 for i in range(num_points)]
for i in range(num_points):
    big = exponent_base**i
    length = 1000
    dtype = torch.float32
    rot = torch.randn(length, length).svd()[0].cuda()

    def make_vec(big, length, dtype=torch.float32):
        v = torch.ones(length).to(dtype)
        v[0] *= big
        return v

    def tn(tens):
        return tens.detach().cpu().numpy()

    v = make_vec(big, length, dtype).cuda()
    v2 = v @ rot @ rot.T
    print(i, big, v2[1:].std())
    vals_fp32[i] = v2[1:].std().item()
    means_fp32[i] = v2[1:].mean().item()
    # px.line(tn(v2[1:])).show()
vals_fp16 = [0 for i in range(num_points)]
means_fp16 = [0 for i in range(num_points)]
for i in range(num_points):
    big = exponent_base**i
    length = 1000
    dtype = torch.float16
    rot = torch.randn(length, length).svd()[0].cuda().to(dtype)

    def make_vec(big, length, dtype=torch.float32):
        v = torch.ones(length).to(dtype)
        v[0] *= big
        return v

    def tn(tens):
        return tens.detach().cpu().numpy()

    v = make_vec(big, length, dtype).cuda()
    v2 = v @ rot @ rot.T
    print(i, big, v2[1:].std())
    vals_fp16[i] = v2[1:].std().item()
    means_fp16[i] = v2[1:].mean().item()
    # px.line(tn(v2[1:])).show()
vals_bf16 = [0 for i in range(num_points)]
means_bf16 = [0 for i in range(num_points)]
for i in range(num_points):
    big = exponent_base**i
    length = 1000
    dtype = torch.bfloat16
    rot = torch.randn(length, length).svd()[0].cuda().to(dtype)

    def make_vec(big, length, dtype=torch.float32):
        v = torch.ones(length).to(dtype)
        v[0] *= big
        return v

    def tn(tens):
        return tens.detach().cpu().numpy()

    v = make_vec(big, length, dtype).cuda()
    v2 = v @ rot @ rot.T
    print(i, big, v2[1:].std())
    vals_bf16[i] = v2[1:].std().item()
    means_bf16[i] = v2[1:].mean().item()
arr = np.array([vals_fp32, vals_fp16, vals_bf16])
# px.line(arr.T).show()
# %%
def lines(
    lines_list,
    x=None,
    mode="lines",
    labels=None,
    xaxis="",
    yaxis="",
    title="",
    log_y=False,
    log_x=False,
    hover=None,
    **kwargs,
):
    # Helper function to plot multiple lines
    if type(lines_list) == torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x = np.arange(len(lines_list[0]))
    fig = go.Figure(layout={"title": title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line) == torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(
            go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs)
        )
    if log_y:
        fig.update_layout(yaxis_type="log")
    if log_x:
        fig.update_layout(xaxis_type="log")
    fig.show()


# %%
lines(
    [vals_fp32, vals_fp16, vals_bf16],
    x=[3**i for i in range(num_points)],
    labels=["fp32", "fp16", "bf16"],
    xaxis=f"{exponent_base}^x",
    yaxis="std",
    title="std of vector after rotation",
    log_y=True,
    log_x=True,
)
lines(
    [means_fp32, means_fp16, means_bf16],
    x=[3**i for i in range(num_points)],
    labels=["fp32", "fp16", "bf16"],
    xaxis=f"{exponent_base}^x",
    yaxis="std",
    title="mean of vector after rotation",
    log_y=True,
    log_x=True,
)
lines(
    [means_fp32, means_fp16, means_bf16],
    x=[3**i for i in range(num_points)],
    labels=["fp32", "fp16", "bf16"],
    xaxis=f"{exponent_base}^x",
    yaxis="std",
    title="mean of vector after rotation",
    log_y=False,
    log_x=True,
)
