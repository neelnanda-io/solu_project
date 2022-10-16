#     cfg = create_cfg({'n_layers':n, 'd_model':512, 'attn_only':False, 'act_fn':'solu_ln', 'seed':n*1234})
#     model = Transformer(cfg)
#     init_weights(model, cfg)

#     torch.save(model.state_dict(), INITIALIZATION_DIR/f"{n}L512W.pth")

# cfg = create_cfg({'n_layers':2, 'd_model':256, 'attn_only':True, 'act_fn':'solu_ln', 'seed':n*1234})
# model = Transformer(cfg)
# init_weights(model, cfg)

# torch.save(model.state_dict(), INITIALIZATION_DIR/f"{n}L256W_attn_only.pth")