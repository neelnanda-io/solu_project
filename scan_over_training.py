# %%
from solu_utils import *
line(np.arange(5))
CHECKPOINT_ROOT = Path('/workspace/solu_project/solu_checkpoints/v9_1L1024W/')
checkpoint_names = os.listdir(CHECKPOINT_ROOT)[:-3]
# print(checkpoint_names)
model = Transformer(cfg, tokenizer)
model.to('cuda')
# %%
import re
# sds = {}
def name_to_step(checkpoint_name):
    s = re.split('_|\.', checkpoint_name)
    return int(s[-2])
# for checkpoint_name in checkpoint_names:
#     step = name_to_step(checkpoint_name)
#     sds[step] = torch.load(CHECKPOINT_ROOT/checkpoint_name, map_location='cuda')
#     # print(checkpoint_name, model(dataset[0]['text'].cuda().unsqueeze(0), return_type='loss'))

# # %%
# model.reset_hooks()
# torch.set_grad_enabled(False)
# neuron_caches = {}
# interesting_neurons = [2, 5, 15, 18]
# neuron_index = interesting_neurons[0]
# text_indices = stores_dict['0post'].index[neuron_index]
# tokens = dataset[text_indices]['text'].cuda()
# print(tokens)
# names = ['blocks.0.mlp.hook_pre', 'blocks.0.mlp.hook_post', 'blocks.0.mlp.hook_post_ln']
# cache = {names: [] for names in names}
# def caching_hook(act, hook):
#     cache[hook.name].append(act[:, :, neuron_index].detach())
# model.reset_hooks()
# model.blocks[0].mlp.hook_pre.add_hook(caching_hook)
# model.blocks[0].mlp.hook_post.add_hook(caching_hook)
# model.blocks[0].mlp.hook_post_ln.add_hook(caching_hook)
# for checkpoint_name in tqdm.tqdm(checkpoint_names):
#     model.load_state_dict(torch.load(CHECKPOINT_ROOT/checkpoint_name, map_location='cuda'))
#     model(tokens, calc_logits=False)

# cache_2 = {}
# for name in names:
#     cache_2[name] = torch.stack(cache[name], dim=0).detach().cpu().numpy()
# neuron_caches[neuron_index] = cache_2
# # %%
# norms = []
# norms_folded = []
# norms_ln = []
# for checkpoint_name in checkpoint_names:
#     sd = torch.load(CHECKPOINT_ROOT/checkpoint_name, map_location='cpu')
#     W_in = sd['blocks.0.mlp.W_in']
#     w_ln = sd['blocks.0.norm2.w']
#     norms.append(W_in[neuron_index].norm().item())
#     norms_ln.append(w_ln.norm().item())
#     norms_folded.append((W_in[neuron_index] * w_ln).norm().item())
# lines([norms, norms_folded, norms_ln])
# %%
# norms = np.array(norms)
# norms_folded = np.array(norms_folded)
# norms_ln = np.array(norms_ln)

# %%
model.load_state_dict(torch.load('/workspace/solu_project/solu_checkpoints/SoLU_1L_v9_final.pth'))
W_U = model.unembed.W_U
W_out = model.blocks[0].mlp.W_out
# [d_vocab, d_mlp]
if cfg['normalization'] == 'RMS':
    print("Folded!")
    W_logit = W_U @ (model.norm.w[:, None] * W_out)
else:
    print("Not folded!")
    W_logit = W_U @ W_out
def print_neuron_logits(neuron_index, top_k=5):
    l = []
    l.append(f"<h1>Neuron {neuron_index}</h1>")
    l.append(f"<b>Top {top_k} logits for Neuron: {neuron_index}</b>")
    logit_vec, logit_indices = W_logit[:, neuron_index].sort()
    for i in range(top_k):
        l.append(f"<p style='color: blue; font-family: \"Courier New\"'>+{logit_vec[-i-1].item():.6f} <b>|{tokenizer.decode([logit_indices[-i-1].item()], clean_up_tokenization_spaces=False)}|</b></p>")
    l.append('...')
    for i in range(top_k):
        l.append(f"<p style='color: red; font-family: \"Courier New\"'>{logit_vec[top_k-i-1].item():.6f} <b>|{tokenizer.decode([logit_indices[top_k-i-1].item()], clean_up_tokenization_spaces=False)}|</b></p>")
    return pysvelte.Html(html="".join(l))

# %%
def text_to_str_tokens(text, tokenizer=tokenizer):
    if text.startswith('<|endoftext|>'):
        return tokenizer.batch_decode(tokenizer.encode(text))
    else:
        return tokenizer.batch_decode(tokenizer.encode("<|endoftext|>"+text))

def vis_activations(str_tokens, activations, name="", incl_bos=True, plot=True):
    if type(str_tokens)==str:
        str_tokens = text_to_str_tokens(str_tokens)
    if incl_bos:
        html = pysvelte.TextSingle(tokens=str_tokens, activations=activations[:], neuron_name=name)
    else:
        html = pysvelte.TextSingle(tokens=str_tokens[1:], activations=activations[1:], neuron_name=name)
    if plot:
        html.show()
    else:
        return html


# %%
print("Loading the stores")
stores_dict = load_stores_dict("/workspace/solu_project/max_act_solu_v2_1662711752.9306793/4328980.pt", cfg['d_mlp'], dtype=torch.float32)
# %%
model.reset_hooks()
cache = {}
model.to(torch.float32)
model.cache_all(cache, remove_batch_dim=True)
neuron_index = 0
prefix = 'blocks.0.mlp.hook_'
name = 'post'
names = ['post', 'post_ln', 'pre']
def logit_str_to_html(logit_str):
    l = logit_str.split('\n')
    return pysvelte.Html(html="".join([f"<p>{i}</p>" for i in l]))
for neuron_index in tqdm.tqdm(range(1)):
# for neuron_index in tqdm.tqdm(range(cfg['d_mlp'])):
    f = open(f'/workspace/solu_project/html_pages/{neuron_index}.html', 'w')
    logit_str = print_neuron_logits(neuron_index)
    htmls = [logit_str]
    for example_index in range(10):
        text_index = stores_dict['0'+name].index[neuron_index, example_index]
        tokens = dataset[text_index.item()]['text'].unsqueeze(0)
        model(tokens, calc_logits=False)
        text = tokenizer.decode(tokens[0])
        html = vis_activations(str_tokens=tokenizer.batch_decode(tokens[0]), activations=cache[prefix+name][:, neuron_index], name=f"Act_{name}_N{neuron_index}_#{example_index}", plot=False)
        htmls.append(html)
    f.write(sum(htmls).html_page_str())
    f.close()
# %%
'''
+2.118806 |.’|

+1.900040 |?’|

+1.839577 |.’”|

+1.800300 |,’|

+1.551048 |!’|

...
-3.012367 | ("|

-3.037094 |)"|

-3.049691 |"),|

-3.265025 |")|

-3.369297 | **(|
'''

# Testing W_logit is legit
batch = dataset[:10]['text'].cuda()
cache = {}
model.cache_all(cache)
logits, loss = model(batch)
print(loss)
model.reset_hooks()
# %%
print(cache.keys())
# %%
W_logit.shape
# %%
def is_close(tens1, tens2, **kwargs):
    print(torch.isclose(tens1, tens2, **kwargs).sum()/tens1.numel())

W_out = model.blocks[0].mlp.W_out
b_out = model.blocks[0].mlp.b_out
neur_post = cache['blocks.0.mlp.hook_post']
neur_post_ln = cache['blocks.0.mlp.hook_post_ln']
resid_mid = cache['blocks.0.hook_resid_mid']
mlp_out = cache['blocks.0.hook_mlp_out']
resid_post = cache['blocks.0.hook_resid_post']
is_close(mlp_out, torch.einsum('mn,bpn->bpm', W_out, neur_post_ln) + b_out)
# %%
neur_post_ave = neur_post.mean(-1, keepdim=True)
neur_post_std = ((neur_post - neur_post_ave).pow(2).mean(-1, keepdim=True)+1e-5).sqrt()
is_close(neur_post_std, cache['blocks.0.mlp.ln.hook_scale'].sqrt())
# %%
is_close(neur_post_ln, (((neur_post - neur_post_ave)/neur_post_std)*model.blocks[0].mlp.ln.w + model.blocks[0].mlp.ln.b))

# %%
is_close(resid_post, resid_mid + torch.einsum('mn,bpn->bpm', W_out, ((neur_post - neur_post_ave)/neur_post_std)*model.blocks[0].mlp.ln.w + model.blocks[0].mlp.ln.b) + b_out, rtol=1e-4)

# %%
# resid_post_ave = resid_post - resid_post.mean(-1, keepdim=True)
resid_post_scale = ((resid_post).pow(2).mean(-1, keepdim=True)+1e-5).sqrt()
is_close(resid_post_scale, cache['norm.hook_scale'])

# %%
W_U_eff = W_U * model.norm.w
is_close(logits, torch.einsum('vm,bpm->bpv', W_U_eff, (resid_post/resid_post_scale)), rtol=1e-4, atol=1e-6)
# %%
approx_logits = logits - torch.einsum('vm,bpm->bpv', W_U_eff, (resid_mid/resid_post_scale)) - (W_U_eff @ b_out)/resid_post_scale
W_U_eff_2 = W_U_eff @ W_out
is_close(approx_logits, torch.einsum('vn,bpn->bpv', W_U_eff_2, neur_post_ln)/resid_post_scale, rtol=1e-4, atol=1e-6)

# %%

# get_corner(torch.randn(5, 5, 5, 5, 6))



# lines(arr, labels=['fp32', 'fp16', 'bf16'], xaxis=f'{exponent_base}^x', yaxis='std', title='std of vector after rotation', log_y=False)
# %%
