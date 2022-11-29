# %%
# Setup
from neel.imports import *
from solu.microscope.microscope import *

pio.renderers.default = "vscode"
torch.set_grad_enabled(False)
import gradio as gr
from easy_transformer import EasyTransformer
from easy_transformer.utils import to_numpy
from IPython.display import HTML
import argparse
import solu.utils as utils

# %%
# Variables
debug = IN_IPYTHON
DEFAULT_CFG = dict(
    layer=0,
    model_name="solu-1l",
    # data_name = "c4-code",
    version=3,
    top_k=20,
    top_logits=10,
    bottom_logits=5,
    website_version=2,
    debug=False,
    use_logits=False,
)
if not IN_IPYTHON:
    cfg = utils.arg_parse_update_cfg(DEFAULT_CFG)
else:
    cfg = DEFAULT_CFG
layer = cfg["layer"]
model_name = cfg["model_name"]
# data_name = cfg['data_name']
version = cfg["version"]
top_k = cfg["top_k"]
top_logits = cfg["top_logits"]
bottom_logits = cfg["bottom_logits"]
website_version = cfg["website_version"]
debug = cfg["debug"] or IN_IPYTHON
use_logits = cfg["use_logits"]

data_name = "pile" if "old" in model_name else "c4-code"
print("Data name:", data_name)


WEBSITE_DIR = Path(
    f"/workspace/solu_project/lexoscope{'_debug' if debug else ''}/v{website_version}/{model_name}/{layer}"
)
WEBSITE_DIR.mkdir(parents=True, exist_ok=True)
fancy_data_names = {
    "c4-code": "80% C4 (Web Text) and 20% Python Code",
    "pile": "The Pile",
}
fancy_model_names = {
    "solu-1l": "SoLU Model 1 Layer Width 512",
    "solu-2l": "SoLU Model 2 Layer Width 512",
    "solu-3l": "SoLU Model 3 Layer Width 512",
    "solu-4l": "SoLU Model 4 Layer Width 512",
    "solu-1l-old": "SoLU Model 1 Layer Width 1024",
    "solu-2l-old": "SoLU Model 2 Layer Width 736",
    "solu-4l-old": "SoLU Model 4 Layer Width 512",
    "solu-6l-old": "SoLU Model 6 Layer Width 768",
    "solu-8l-old": "SoLU Model 8 Layer Width 1024",
    "solu-10l-old": "SoLU Model 10 Layer Width 1280",
}
# %%
# Loading
if data_name == "c4-code":
    data = get_c4_code()
elif data_name == "pile":
    data = get_pile()
else:
    raise ValueError(f"Invalid data name {data_name}")

model = EasyTransformer.from_pretrained(model_name)
if "old" not in model_name:
    max_act_store = MaxActStore.load(f"{model_name}-{data_name}_v{version}L{layer}")
else:
    max_act_store = MaxActStore.load(f"{model_name}_v{version}L{layer}")
max_act_store.max_acts = max_act_store.max_acts.cpu()
max_act_store.index = max_act_store.index.cpu()
max_act_store.max_acts, indices = max_act_store.max_acts.sort(descending=True)
max_act_store.index = max_act_store.index.gather(-1, indices)
# %%
W_U = model.W_U
# if not isinstance(model.ln_final, easy_transformer.components.LayerNormPre):
#     print("Folding in Layer Norm")
#     W_U = model.ln_final.w[:, None] * W_U
W_logit = model.blocks[layer].mlp.W_out @ W_U
print("W_logit:", W_logit.shape)
# %%
def get_neuron_acts(text, neuron_index):
    """Hacky way to get out state from a single hook - we have a single element list and edit that list within the hook."""
    cache = {}

    def caching_hook(act, hook):
        cache["activation"] = act[0, :, neuron_index]

    model.run_with_hooks(
        text,
        fwd_hooks=[(f"blocks.{layer}.mlp.hook_mid", caching_hook)],
        return_type=None,
    )
    return to_numpy(cache["activation"])


def get_batch_neuron_acts(tokens, neuron_index):
    """Hacky way to get out state from a single hook - we have a single element list and edit that list within the hook.

    We feed in a batch x pos batch of tokens, and get out a batch x pos tensor of activations.
    """
    cache = {}

    def caching_hook(act, hook):
        cache["activation"] = act[:, :, neuron_index]

    # Data already comes with bos prepended
    model.run_with_hooks(
        tokens,
        fwd_hooks=[(f"blocks.{layer}.mlp.hook_mid", caching_hook)],
        return_type=None,
    )
    return cache["activation"].cpu()


# Test
# For some reason, there's slight differences in the activations, but doesn't matter lol. Confusing though! Also not in a consistent direction. I wonder if it's downstream of how the tensor is stored or smth?
if "tokens" in data[0]:
    tokens_name = "tokens"
else:
    tokens_name = "text"
print("Tokens name:", tokens_name)

if debug:
    out = get_batch_neuron_acts(data[max_act_store.index[5]][tokens_name], 5)
    print(out.shape)
    print(out.max(1).values)
    print(max_act_store.max_acts[5])
    print(
        torch.isclose(
            out.max(1).values, max_act_store.max_acts[5], rtol=1e-3, atol=1e-5
        )
    )
# %%
# Make HTML
# This is some CSS (tells us what style )to give each token a thin gray border, to make it easy to see token separation
grey_color = 180
style_string = f"""<style> 
    span.token {{
        border: 1px solid rgb({grey_color}, {grey_color}, {grey_color});
        white-space: pre;
        color: rgb(0, 0, 0);
        }} 
    div.token-text {{
        word-wrap: normal;
        }} 
    </style>"""
#  display: flex; flex-wrap: wrap;
if debug:
    print(style_string)
if debug and IN_IPYTHON:
    display(
        HTML(
            style_string
            + "<span class='token'>Text!</span><span class='token'>Tixt</span>"
        )
    )
# %%
def calculate_color(val, max_val, min_val):
    # Hacky code that takes in a value val in range [min_val, max_val], normalizes it to [0, 1] and returns a color which interpolates between slightly off-white and red (0 = white, 1 = red)
    # We return a string of the form "rgb(240, 240, 240)" which is a color CSS knows
    normalized_val = (val - min_val) / max_val
    return f"rgb(250, {round(250*(1-normalized_val))}, {round(250*(1-normalized_val))})"


def make_single_token_text(str_token, act, max_val, min_val):
    return f"<span class='token' style='background-color:{calculate_color(act, max_val, min_val)}' >{str_token}</span>"


def make_header(neuron_index):
    htmls = []
    htmls.append(f"<div style='font-size:medium;'>")
    if neuron_index > 0:
        htmls.append(f"< <a href='./{neuron_index-1}.html'>Prev</a> | ")
    htmls.append(f"<a href='../../../index.html'>Home</a> | ")
    htmls.append(f"<a href='../../model.html'>Model</a> | ")
    htmls.append(
        f"<a href='./{random.randint(0, model.cfg.d_mlp-1)}.html'>Random</a> | "
    )
    if neuron_index < model.cfg.d_mlp:
        htmls.append(f"<a href='./{neuron_index+1}.html'>Next</a> >")
    htmls.append(f"</div>")
    htmls.append(f"<h1>Model: {fancy_model_names[model_name]}</h1>")
    htmls.append(f"<h1>Dataset: {fancy_data_names[data_name]}</h1>")
    htmls.append(f"<h2>Neuron {neuron_index} in Layer {layer} </h2>")
    htmls.append(
        f"<h3>Easy Transformer Loading: <span style='font-family: \"Courier New\"'>EasyTransformer.from_pretrained('{model_name}')</span></h3>"
    )
    return "".join(htmls)


def make_logits(neuron_index):
    if not use_logits:
        return ""
    htmls = []
    htmls.append("<h3>Direct Logit Effect</h3>")
    logit_vec, logit_indices = W_logit[neuron_index].sort(descending=True)
    for i in range(top_logits):
        htmls.append(
            f"<p style='color: blue; font-family: \"Courier New\"'>#{i} +{logit_vec[i].item():.4f} <span class='token'>{model.to_string([logit_indices[i].item()])}</span></p>"
        )
    htmls.append("<p>...</p>")
    for i in range(bottom_logits):
        htmls.append(
            f"<p style='color: red; font-family: \"Courier New\"'>#{i} {logit_vec[-(i+1)].item():.4f} <span class='token'>{model.to_string([logit_indices[-(i+1)].item()])}</span></p>"
        )
    return "".join(htmls)


def make_token_text(
    tokens: np.ndarray, acts: np.ndarray, max_val: float, min_val: float
):
    str_tokens = model.to_str_tokens(tokens)
    act_max = acts.max()
    act_min = acts.min()
    htmls = [style_string]
    htmls.append(
        f"<h4>Max Range: <b>{max_val:.4f}</b>. Min Range: <b>{min_val:.4f}</b></h4>"
    )
    htmls.append(
        f"<h4>Max Act: <b>{act_max:.4f}</b>. Min Act: <b>{act_min:.4f}</b></h4>"
    )
    htmls.append("<div class='token-text'>")
    for str_token, act in zip(str_tokens, acts):
        # A span is an HTML element that lets us style a part of a string (and remains on the same line by default)
        # We set the background color of the span to be the color we calculated from the activation
        # We set the contents of the span to be the token
        htmls.append(make_single_token_text(str_token, act, max_val, min_val))
    htmls.append("</div>")
    return "".join(htmls)


def make_token_texts(tokens: np.ndarray, acts: np.ndarray, neuron_index: int):
    max_val = acts.max()
    min_val = 0.0
    return "<br><br>".join(
        [
            f"<h3>Text #{i}</h3>"
            + make_token_text(tokens[i], acts[i], max_val, min_val)
            for i in range(top_k)
        ]
    )


def make_html(neuron_index):
    data_indices = max_act_store.index[neuron_index]
    tokens = data[data_indices][tokens_name]
    acts = get_batch_neuron_acts(tokens, neuron_index)
    acts = to_numpy(acts)
    tokens = to_numpy(tokens)
    htmls = [
        make_header(neuron_index),
        make_logits(neuron_index),
        make_token_texts(tokens, acts, neuron_index),
    ]
    return "<hr>".join(htmls)


if debug and IN_IPYTHON:
    display(HTML(make_html(5)))
# %%
if debug:
    num_pages = 5
else:
    num_pages = model.cfg.d_mlp
for neuron_index in tqdm.tqdm(range(num_pages)):
    with open(WEBSITE_DIR / f"{neuron_index}.html", "w") as f:
        f.write(make_html(neuron_index))
# %%
def gen_index_page(model_names):
    htmls = []
    htmls.append(
        "<h1>Lexoscope: A Website for Mechanistic Interpretability of Language Models</h1>"
    )
    htmls.append(
        "<div>Each model has a page per neuron, displaying the top 20 maximum activating dataset examples.</div>"
    )
    htmls.append("<h2>Supported models</h2>")
    htmls.append("<ul>")
    for name in model_names:
        fancy_name = fancy_model_names[name]
        htmls.append(
            f"<li><b><a href='./{name}/model.html'>{name}</a>:</b> {fancy_name}</li>"
        )
    htmls.append("</ul>")
    return "".join(htmls)


def gen_model_page(model_name):
    layer_num = int(re.match("solu-(\d+)l.*", model_name).group(1))
    htmls = []
    htmls.append(f"<div style='font-size:medium;'>")
    if neuron_index > 0:
        htmls.append(f"< <a href='./0/{neuron_index-1}.html'>First Neuron</a> | ")
    htmls.append(f"<a href='../index.html'>Home</a> | ")
    htmls.append(
        f"<a href='./{random.randint(0, layer_num-1)}/{random.randint(0, model.cfg.d_mlp-1)}.html'>Random</a> | "
    )
    if neuron_index < model.cfg.d_mlp:
        htmls.append(
            f"<a href='./{layer_num-1}/{neuron_index+1}.html'>Final Neuron</a> >"
        )
    htmls.append(f"</div>")
    htmls.append(f"<h1>Model Index Page: {fancy_model_names[model_name]}</h1>")
    if "old" in model_name:
        htmls.append(f"<h1>Dataset: {fancy_data_names['pile']}</h1>")
    else:
        htmls.append(f"<h1>Dataset: {fancy_data_names['c4-code']}</h1>")
    htmls.append(
        f"<h2>Easy Transformer Loading: <span style='font-family: \"Courier New\"'>EasyTransformer.from_pretrained('{model_name}')</span></h2>"
    )
    htmls.append(f"<h2>Layers:</h2>")
    htmls.append("<ul>")
    for l in range(layer_num):
        htmls.append(f"<li><b><a href='./{l}/0.html'>Layer #{l}</b></a></li>")
    htmls.append("</ul>")
    return "".join(htmls)


if IN_IPYTHON:
    REAL_DIR = Path(f"/workspace/solu_project/lexoscope/v{website_version}")
    REAL_DIR.mkdir(exist_ok=True)
    model_names = [
        "solu-1l",
        "solu-2l",
        "solu-2l-old",
        "solu-3l",
        "solu-4l",
        "solu-8l-old",
    ]
    print("Index:")
    index_html = gen_index_page(model_names)
    display(HTML(index_html))
    (REAL_DIR / "index.html").write_text(index_html)
    for name in model_names:
        print(f"Model: {name}")
        model_html = gen_model_page(name)
        display(HTML(model_html))
        folder = REAL_DIR / name
        folder.mkdir(exist_ok=True)
        (folder / "model.html").write_text(model_html)
command = "scp /workspace/solu_project/lexoscope/v2/index.html neelnanda_lexoscope@ssh.phx.nearlyfreespeech.net:/home/public/index.html"


# %%
