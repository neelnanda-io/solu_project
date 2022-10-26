"""
# torch.save({k:(v.counter, v.max_acts, v.index) for k, v in stores.items()}, "max_acts_128W_1_7m_v2.pth")
# # %%
# torch.set_grad_enabled(False)
# do_logging = False
# model.reset_hooks()

# def update_store_hook(act, hook, store):
#     act_max = act.max(-2).values
#     for i in range(act_max.size(0)):
#         store.update(act_max[i])
# stores = {}
# for hook in [model.blocks[0].mlp.hook_pre, model.blocks[0].mlp.hook_post, model.blocks[0].mlp.hook_post_ln]:
#     s = hook.name.split('.')
#     name = f"{s[1]}{s[2][5:]}"
#     stores[hook.name] = MaxActStore(cfg['d_mlp'], name=hook.name, log=do_logging)
#     hook.add_hook(partial(update_store_hook, store=stores[hook.name]))
# print(stores)

     
# if do_logging:
#     wandb.init(config=cfg, project='max_act_solu')
# data_loader = DataLoader(dataset, batch_size=20, pin_memory=True, num_workers=10)
# data_iter = iter(data_loader)
# max_c = 10**5
# with torch.inference_mode():
#     for c, batch in enumerate(tqdm.tqdm(data_iter)):
#         tokens = batch['text'].cuda()
#         model(tokens, calc_logits=False)
# if do_logging:
#     wandb.finish()

# torch.set_grad_enabled(False)
do_logging = True
metric_cache = {}
for batch_size in [5, 10, 20, 100, 200]:
    model.reset_hooks()
    def update_store_hook(act, hook, store):
        act_max = act.max(-2).values
        store.batch_update(act_max)

    def cache_act(act, hook):
        hook.ctx['act'] = act.detach()

    def get_attr_hook(grad, hook, store):
        store.batch_update((grad * hook.ctx['act']).max(-2).values)

    stores = {}
    for hook in [model.blocks[0].mlp.hook_pre, model.blocks[0].mlp.hook_post, model.blocks[0].mlp.hook_post_ln]:
        s = hook.name.split('.')
        name = f"{s[1]}{s[2][5:]}"
        stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
        hook.add_hook(partial(update_store_hook, store=stores[name]))
        name = f"{s[1]}{s[2][5:]}_attr"
        hook.add_hook(cache_act)
        stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
        hook.add_hook(partial(get_attr_hook, store=stores[name]), dir='bwd')
    W_U = model.unembed.W_U
    W_out = model.blocks[0].mlp.W_out
    # [d_vocab, d_mlp]
    W_logit = W_U @ W_out

    def direct_logit_attr_hook(act, hook, store):
        # Shape batch, pos, d_mlp
        # act has shape batch, pos, d_mlp
        store.batch_update((act * W_logit[tokens]).max(-2).values)
    name = 'logit_attr'
    stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
    model.blocks[0].mlp.hook_post_ln.add_hook(partial(update_store_hook, store=stores[name]))

    # data_loader = DataLoader(dataset, batch_size=20, pin_memory=True)
    # batch = next(iter(data_loader))
    # tokens = batch['text'].cuda()
    # loss = model(tokens, return_type='loss')
    # loss.backward()
    # del loss
    # print(stores)
    if do_logging:
        wandb.init(config=cfg, project='max_act_solu')
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    data_iter = iter(data_loader)
    max_c = 2*10**4
    start_time = time.time()
    # with torch.inference_mode():
    for c, batch in enumerate(tqdm.tqdm(data_iter)):
        tokens = batch['text'].cuda()
        loss = model(tokens, calc_logits=True, return_type='loss')
        loss.backward()
        del loss
        if (c * batch_size) >= max_c:
            output = (time.time() - start_time)
            print(batch_size, output)
            metric_cache[batch_size]=output
            break
    if do_logging:
        wandb.finish()
print(metric_cache)

div_by_4_cache = {
    1: 17.842707633972168,
    2: 12.305243015289307,
    5: 9.121133089065552,
    10: 8.626007556915283,
    20: 7.983260631561279,
    50: 7.892266035079956,
    100: 7.862484455108643
}

old_metric_cache = {
    10: 33.940598011016846,
    20: 32.53163957595825,
    50: 32.89558291435242,
    100: 32.4731719493866,
    200: 32.48869609832764,
    500: 33.245705366134644,
    1000: 34.51264953613281
}
batched_metric_cache = {
    10: 12.485169172286987,
    20: 9.546191453933716,
    50: 17.380213022232056,
    100: 8.173691272735596,
    200: 7.056461572647095,
    500: 6.916813373565674,
    1000: 6.849961042404175
}

batched_bfloat16_cache = {
    10: 9.195661544799805,
    20: 7.18474817276001,
    50: 15.344133138656616,
    100: 8.605464696884155,
    200: 5.258362531661987,
    500: 5.256837368011475,
    1000: 5.376178026199341
}

calc_logits_bfloat16_cache = {10: 55.207096099853516, 20: 52.917214155197144, 100: 63.41084957122803, 200: 73.57792639732361}

float32_calc_logits = {10: 50.02782344818115,
 20: 49.28316593170166,
 50: 57.536864280700684,
 100: 66.34676790237427}
"""
# bfloat16, calc logits
# 10 27.067036151885986
# 20 26.091500997543335
# 100 36.60754323005676
# 200 44.59227895736694
"""
print(old_metric_cache)

model.reset_hooks()
def update_store_hook(act, hook, store):
    act_max = act.max(-2).values
    store.batch_update(act_max)

def cache_act(act, hook):
    hook.ctx['act'] = act.detach()

def get_attr_hook(grad, hook, store):
    store.batch_update((grad * hook.ctx['act']).max(-2).values)

stores = {}
for hook in [model.blocks[0].mlp.hook_pre, model.blocks[0].mlp.hook_post, model.blocks[0].mlp.hook_post_ln]:
    s = hook.name.split('.')
    name = f"{s[1]}{s[2][5:]}"
    stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
    hook.add_hook(partial(update_store_hook, store=stores[name]))
    name = f"{s[1]}{s[2][5:]}_attr"
    hook.add_hook(cache_act)
    stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
    hook.add_hook(partial(get_attr_hook, store=stores[name]), dir='bwd')
W_U = model.unembed.W_U
W_out = model.blocks[0].mlp.W_out
# [d_vocab, d_mlp]
W_logit = W_U @ W_out

def direct_logit_attr_hook(act, hook, store):
    # Shape batch, pos, d_mlp
    # act has shape batch, pos, d_mlp
    store.batch_update((act * W_logit[tokens]).max(-2).values)
name = 'logit_attr'
stores[name] = MaxActStore(cfg['d_mlp'], name=name, log=do_logging)
model.blocks[0].mlp.hook_post_ln.add_hook(partial(update_store_hook, store=stores[name]))

data_loader = DataLoader(dataset, batch_size=20, pin_memory=True)
batch = next(iter(data_loader))
tokens = batch['text'].cuda()
loss = model(tokens, return_type='loss')
loss.backward()
del loss

"""

print("Finish")

# example_text = "Avery Brundage (1887–1975) was the fifth president of the International Olympic Committee (IOC), the only American to hold that office. In 1912, he competed in the Summer Olympics, contesting the pentathlon and decathlon; both events were won by Jim Thorpe. Brundage became a sports administrator, rising rapidly through the ranks in U.S. sports groups. He fought zealously against a boycott of the 1936 Summer Olympics in Berlin, Nazi Germany. Although Brundage was successful, the U.S. participation was controversial, and has remained so. Brundage was elected to the IOC that year, and quickly became a major figure in the Olympic movement. Elected IOC president in 1952, Brundage fought strongly for amateurism."
# cache = {}
# model.cache_all(cache, remove_batch_dim=True)
# logits, loss = model(example_text)
# logits = logits.squeeze()

# def text_to_str_tokens(text, tokenizer=tokenizer):
#     return tokenizer.batch_decode(tokenizer.encode("<|endoftext|>"+text))



# for neuron_index in range(1):
#     print(neuron_index)
#     v = torch.zeros_like(cache['blocks.0.mlp.hook_pre'][:, neuron_index])
#     v[0]=1.
#     vis_activations(example_text, v, name=f'Pre Activation for neuron {neuron_index}')
#     vis_activations(example_text, cache['blocks.0.mlp.hook_pre'][:, neuron_index], name=f'Pre Activation for neuron {neuron_index}')
    
    # vis_activations(example_text, cache['blocks.0.mlp.hook_post'][:, neuron_index], name=f'Post Activation for neuron {neuron_index}')
    # vis_activations(example_text, cache['blocks.0.mlp.hook_post'][:, neuron_index], name=f'Post LN for neuron {neuron_index}')
    

# logits, loss = model(example_text)
# # %%
# start_time = time.time()
# if cfg['shuffled_data']:
# randperm = np.random.permutation(28)
# print('Permutation of PILE URLs', randperm)
# pile_urls = [f"https://mystic.the-eye.eu/public/AI/pile/train/{i:0>2}.jsonl.zst" for i in randperm]
# dataset = load_dataset('json', data_files="https://mystic.the-eye.eu/public/AI/pile/train/29.jsonl.zst", split='train', cache_dir='cache', streaming=True)
#     # else:
#     #     dataset = load_dataset(cfg['dataset_name'], streaming=True, split='train')
# print('Loaded!', time.time()-start_time)
# start_time = time.time()
# try:
#     dataset = dataset.remove_columns('meta')
# except:
#     print('Meta not in dataset')

# start_time = time.time()
# dataset = dataset.with_format(type='torch')
# print('dataset.set_format', time.time()-start_time)
# start_time = time.time()
# # dataset = dataset.shuffle(seed=cfg['seed'], buffer_size=30000)
# print('dataset.shuffle', time.time()-start_time)
# start_time = time.time()
# train_data_loader = DataLoader(dataset, batch_size=cfg['batch_size'])



# print('Loaded!', time.time()-start_time)
# start_time = time.time()
# dataset = dataset.map(tokenize, batched=True, num_proc=20)
# print('dataset.map', time.time()-start_time)




# # %%
# logits, loss = model(example_text)
# logits2, loss2 = model(x['text'].cuda())
# example_text = " a"*1023
# example_tokens = model.to_tokens(example_text)
# rand_mat = torch.randn(10**9)
# # for b in [True, False]:
# # dl = DataLoader(dataset, batch_size=20, pin_memory=b)
# # tokens = next(iter(dl))['text']
# tokens = dataset[:80]['text'].cuda()
# tokens2 = dataset[80:160]['text'].cuda()
# modelfp32 = Transformer(cfg, tokenizer).cuda()
# modelfp32.load_state_dict(torch.load(Path.home()/'solu_project/solu_checkpoints/SoLU_1L_v11_final.pth'))
# modelbf16 = Transformer(cfg, tokenizer).cuda()
# modelbf16.load_state_dict(torch.load(Path.home()/'solu_project/solu_checkpoints/SoLU_1L_v11_final.pth'))
# modelbf16.to(torch.bfloat16)
# modelfp16 = Transformer(cfg, tokenizer).cuda()
# modelfp16.load_state_dict(torch.load(Path.home()/'solu_project/solu_checkpoints/SoLU_1L_v11_final.pth'))
# modelfp16.to(torch.float16)
# # %%
# from torch.profiler import profile, record_function, ProfilerActivity
# with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=False) as prof:
#     with record_function("Overall inference"):
#         # with record_function('fp32'):
#         #     loss1 = modelfp32(tokens, return_type='loss').detach()
#         #     loss2 = modelfp32(tokens2, return_type='loss').detach()
#         # with record_function('bf16'):
#         #     loss1 = modelbf16(tokens, return_type='loss').detach()
#         #     loss2 = modelbf16(tokens2, return_type='loss').detach()
#         with record_function('fp16'):
#             loss1 = modelfp16(tokens, return_type='loss').detach()
#             loss2 = modelfp16(tokens2, return_type='loss').detach()
    
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))