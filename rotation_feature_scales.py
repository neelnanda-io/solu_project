# %%
import plotly.express as px 
import plotly.io as pio
pio.renderers.default = "notebook"
import plotly.graph_objects as go
import numpy as np
import torch
import tqdm.auto as tqdm
# %%
dtypes = ['fp32', 'bf16', 'fp16']
dtype_map = {'fp32':torch.float32, 'fp16':torch.float16, 'bf16':torch.bfloat16}

# %%
num_points = 20
exponent_base = 3
arrays = {}
for length in tqdm.tqdm([2, 5, 10, 100, 500, 1000]):
    vals_fp32 = [0 for i in range(num_points)] 
    means_fp32 = [0 for i in range(num_points)] 
    for i in range(num_points):
        big = exponent_base**i
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
    vals_fp64 = [0 for i in range(num_points)]
    means_fp64 = [0 for i in range(num_points)]
    for i in range(num_points):
        big = exponent_base**i
        dtype = torch.float64
        rot = torch.randn(length, length).to(torch.float64).svd()[0].cuda().to(dtype)
        def make_vec(big, length, dtype=torch.float32):
            v = torch.ones(length).to(dtype)
            v[0] *= big
            return v
        def tn(tens):
            return tens.detach().cpu().numpy()
        v = make_vec(big, length, dtype).cuda()
        v2 = v @ rot @ rot.T
        print(i, big, v2[1:].std())
        vals_fp64[i] = v2[1:].std().item()
        means_fp64[i] = v2[1:].mean().item()
    arrays[length] = np.array([vals_fp32, vals_fp16, vals_bf16, vals_fp64])
# px.line(arr.T).show()
# %%
def to_numpy(tensor):
    try:
        return tensor.detach().cpu().numpy()
    except:
        return tensor
def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, log_x=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    if log_x:
        fig.update_layout(xaxis_type="log")
    fig.show()

def imshow(tensor, **kwargs):
    px.imshow(to_numpy(tensor), color_continuous_scale='RdBu', color_continuous_midpoint=0.0, **kwargs).show()

def line(tensor, **kwargs):
    lines([tensor], **kwargs)
# %%
lines([vals_fp32, vals_fp16, vals_bf16,  vals_fp64], x=[3**i for i in range(num_points)], labels=['fp32', 'fp16', 'bf16', 'fp64'], xaxis=f'{exponent_base}^x', yaxis='std', title='std of vector after rotation', log_y=True, log_x=True)
lines([means_fp32, means_fp16, means_bf16,  means_fp64], x=[3**i for i in range(num_points)], labels=['fp32', 'fp16', 'bf16', 'fp64'], xaxis=f'{exponent_base}^x', yaxis='mean', title='mean of vector after rotation', log_y=True, log_x=True)
lines([means_fp32, means_fp16, means_bf16,  means_fp64], x=[3**i for i in range(num_points)], labels=['fp32', 'fp16', 'bf16', 'fp64'], xaxis=f'{exponent_base}^x', yaxis='mean', title='mean of vector after rotation', log_y=False, log_x=True)

# %%
num_points = 10
exponent_base = 2
lengths = [2, 5, 10, 100, 500, 1000]
mult_in_64 = False
stds_big = {}
for length in tqdm.tqdm(lengths):
    stds = {}
    for dtype_name in tqdm.tqdm(dtypes):
        dtype = dtype_map[dtype_name]
        std_list = [0 for i in range(num_points)] 
        for i in range(num_points):
            big = exponent_base**i
            
            def make_vec(big, length, dtype=torch.float32):
                v = torch.ones(length).to(dtype)
                v[0] *= big
                return v
            def tn(tens):
                return tens.detach().cpu().numpy()
            v = make_vec(big, length, dtype).cuda()
            if mult_in_64:
                rot = torch.randn(length, length).to(torch.float64).svd()[0].cuda().to(torch.float64)
                v2 = ((v.to(torch.float64) @ rot).to(dtype).to(torch.float64) @ rot.T).to(dtype)
            else:
                rot = torch.randn(length, length).to(torch.float64).svd()[0].cuda().to(dtype)
                v2 = ((v @ rot) @ rot.T).to(dtype)
            # print(i, big, v2[1:].std())
            std_list[i] = v2[1:].std().item()
        stds[dtype_name] = std_list
    stds_big[length] = stds
# %%
#Percentage vs Absolute error
stds = {}
num_points = 8
exponent_base = 3
length = 500
for dtype_name in tqdm.tqdm(dtypes):
    dtype = dtype_map[dtype_name]
    error_list = [0 for i in range(num_points)] 
    perc_error_list = [0 for i in range(num_points)] 
    for i in range(num_points):
        big = exponent_base**i
        
        def make_vec_2(big, length, dtype=torch.float32):
            v = torch.randn(length).to(dtype)
            v[0] = big
            return v
        def tn(tens):
            return tens.detach().cpu().numpy()
        v = make_vec_2(big, length, dtype).cuda()

        rot = torch.randn(length, length).to(torch.float64).svd()[0].cuda().to(dtype)
        v2 = ((v @ rot) @ rot.T).to(dtype)
        # print(i, big, v2[1:].std())
        error_list[i] =  (v2[1:]-v[1:]).std().item()
        perc_error_list[i] = (v2[1:].abs()/(v[1:].abs()+1e-6)).std().item()
        print(i, error_list[i], perc_error_list[i])
    stds[dtype_name] = (error_list, perc_error_list)
vecs = []
l  = []
for dtype_name in dtypes:
    vecs.append(stds[dtype_name][0])
    vecs.append(stds[dtype_name][1])
    l.append(f"{dtype_name} abs")
    l.append(f"{dtype_name} perc")
lines(vecs, labels=l, x=exponent_base**np.arange(num_points), log_y=True, log_x=True, title='Percentage vs Absolute error, Gaussian noise + big')
# %%
stds = {}
num_points = 8
exponent_base = 3
length = 500
for dtype_name in tqdm.tqdm(dtypes):
    dtype = dtype_map[dtype_name]
    error_list = [0 for i in range(num_points)] 
    perc_error_list = [0 for i in range(num_points)]
    perc_error_list_big_eps = [0 for i in range(num_points)]
    perc_error_list_clamp = [0 for i in range(num_points)]
    const_error_list = [0 for i in range(num_points)]
    for i in range(num_points):
        big = exponent_base**i
        
        def make_vec_const(big, length, dtype=torch.float32):
            v = torch.ones(length).to(dtype)
            v[0] = big
            return v
        def make_vec_gauss(big, length, dtype=torch.float32):
            v = torch.randn(length).to(dtype)
            v[0] = big
            return v
        def tn(tens):
            return tens.detach().cpu().numpy()
        v = make_vec_gauss(big, length, dtype).cuda()
        rot = torch.randn(length, length).to(torch.float64).svd()[0].cuda().to(dtype)
        v2 = ((v @ rot) @ rot.T).to(dtype)
        # print(i, big, v2[1:].std())
        error_list[i] =  (v2[1:]-v[1:]).std().item()
        perc_error_list[i] = (v2[1:].abs()/(v[1:].abs()+1e-6)).std().item()
        perc_error_list_clamp[i] = (v2[1:].abs()/(v[1:].abs()+1e-9)).clamp(0, 10).std().item()
        perc_error_list_big_eps[i] = (v2[1:].abs()/((v[1:].abs()+v2[1:].abs())/2 + 1e-6)).std().item()
        
        v3 = make_vec_const(big, length, dtype).cuda()
        rot = torch.randn(length, length).to(torch.float64).svd()[0].cuda().to(dtype)
        v4 = ((v3 @ rot) @ rot.T).to(dtype)
        # print(i, big, v2[1:].std())
        const_error_list[i] =  (v4[1:]-v3[1:]).std().item()
        print(i, error_list[i], perc_error_list[i], perc_error_list_big_eps[i], const_error_list[i])
    stds[dtype_name] = (error_list, perc_error_list, perc_error_list_big_eps,perc_error_list_clamp, const_error_list)
vecs = []
l  = []
for dtype_name in dtypes:
    vecs.append(stds[dtype_name][0])
    l.append(f"{dtype_name} abs Gauss")
    # vecs.append(stds[dtype_name][1])
    # l.append(f"{dtype_name} perc")
    # vecs.append(stds[dtype_name][2])
    # l.append(f"{dtype_name} perc big eps")
    vecs.append(stds[dtype_name][3])
    l.append(f"{dtype_name} percentage Gauss")
    vecs.append(stds[dtype_name][4])
    l.append(f"{dtype_name} abs const")
lines(vecs, labels=l, x=exponent_base**np.arange(num_points), log_y=True, log_x=True, title='Percentage vs Absolute error on Gaussian noise vs absolute error on constant')
# %%
from easy_transformer import EasyTransformer
# %%
model = EasyTransformer('facebook/opt-6.7b')


torch.set_grad_enabled(False)
cache = {}
model.cache_all(cache)
text = "From that, I learned that quantization research is like printers. Nobody cares about printers. Nobody likes printers. But everybody is happy if printers do their job."
logits = model(model.to_tokens(text))
# %%

resids = torch.stack([cache[f'blocks.{l}.hook_resid_pre'][0] for l in range(model.cfg['n_layers'])], axis=0)
np_resids = to_numpy(resids)
max_pos = resids.shape[1]
max_resids = resids.abs().max(0).values.max(0).values
px.histogram(to_numpy(max_resids)).show()
# %%
import random
layer = random.randint(0, model.cfg['n_layers']-1)
pos = random.randint(0, max_pos-1)
px.line(to_numpy(resids[layer, pos, :]), title=f'Res stream at layer {layer} and position {pos}').show()
imshow(np_resids[:, pos], title=f'Rest stream at pos {pos}', aspect='auto')
imshow(np_resids[layer, :], title=f'Rest stream at layer {layer}', aspect='auto')
# %%
line_l = []
label_l = []
for i in range(5):
    
    layer = random.randint(5, model.cfg['n_layers']-6)
    pos = random.randint(1, max_pos-1)
    line_l.append(np_resids[layer, pos, :])
    label_l.append(f'layer {layer}, pos {pos}')
lines(line_l, labels=label_l, title='Residual streams at random positions & layers', opacity=0.5)
# %%

px.imshow((np.abs(np_resids).max(axis=1)), labels={'y':'Layer', 'x':'Index'}, aspect='auto', color_continuous_scale='RdBu', color_continuous_midpoint=0.0, title='Max along pos').show()
px.imshow((np.abs(np_resids).max(axis=0)), labels={'y':'Layer', 'x':'Index'}, aspect='auto', color_continuous_scale='RdBu', color_continuous_midpoint=0.0, title='Max along layer').show()
px.imshow((np.abs(np_resids).max(axis=2)), labels={'y':'Layer', 'x':'Index'}, aspect='auto', color_continuous_scale='RdBu', color_continuous_midpoint=0.0, title='Max along index').show()
# %%
px.imshow((np.log(np.abs(np_resids).max(axis=1))), labels={'y':'Layer', 'x':'Index'}, aspect='auto', color_continuous_scale='RdBu', color_continuous_midpoint=0.0, title='Max along pos').show()
px.imshow((np.log(np.abs(np_resids).max(axis=0))), labels={'y':'Layer', 'x':'Index'}, aspect='auto', color_continuous_scale='RdBu', color_continuous_midpoint=0.0, title='Max along layer').show()
px.imshow((np.log(np.abs(np_resids).max(axis=2))), labels={'y':'Layer', 'x':'Index'}, aspect='auto', color_continuous_scale='RdBu', color_continuous_midpoint=0.0, title='Max along index').show()
# %%
px.imshow((np.log(np.abs(np_resids).mean(axis=1))), labels={'y':'Layer', 'x':'Index'}, aspect='auto', color_continuous_scale='RdBu', color_continuous_midpoint=0.0, title='Max along pos').show()
px.imshow((np.log(np.abs(np_resids).mean(axis=0))), labels={'y':'Layer', 'x':'Index'}, aspect='auto', color_continuous_scale='RdBu', color_continuous_midpoint=0.0, title='Max along layer').show()
px.imshow((np.log(np.abs(np_resids).mean(axis=2))), labels={'y':'Layer', 'x':'Index'}, aspect='auto', color_continuous_scale='RdBu', color_continuous_midpoint=0.0, title='Max along index').show()
# %%
# Analysing a specific position
length = model.cfg.d_model
