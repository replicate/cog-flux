# %% [markdown]
# ___

# %%
available_layers_to_optimize = [
    "transformer.transformer_blocks.0.norm1.linear",
    "transformer.transformer_blocks.0.norm1_context.linear",
    "transformer.transformer_blocks.0.attn.to_q",
    "transformer.transformer_blocks.0.attn.to_k",
    "transformer.transformer_blocks.0.attn.to_v",
    "transformer.transformer_blocks.0.attn.add_k_proj",
    "transformer.transformer_blocks.0.attn.add_v_proj",
    "transformer.transformer_blocks.0.attn.add_q_proj",
    "transformer.transformer_blocks.0.attn.to_out.0",
    "transformer.transformer_blocks.0.attn.to_add_out",
    "transformer.transformer_blocks.0.ff.net.0.proj",
    "transformer.transformer_blocks.0.ff.net.2",
    "transformer.transformer_blocks.0.ff_context.net.0.proj",
    "transformer.transformer_blocks.0.ff_context.net.2",
    "transformer.transformer_blocks.1.norm1.linear",
    "transformer.transformer_blocks.1.norm1_context.linear",
    "transformer.transformer_blocks.1.attn.to_q",
    "transformer.transformer_blocks.1.attn.to_k",
    "transformer.transformer_blocks.1.attn.to_v",
    "transformer.transformer_blocks.1.attn.add_k_proj",
    "transformer.transformer_blocks.1.attn.add_v_proj",
    "transformer.transformer_blocks.1.attn.add_q_proj",
    "transformer.transformer_blocks.1.attn.to_out.0",
    "transformer.transformer_blocks.1.attn.to_add_out",
    "transformer.transformer_blocks.1.ff.net.0.proj",
    "transformer.transformer_blocks.1.ff.net.2",
    "transformer.transformer_blocks.1.ff_context.net.0.proj",
    "transformer.transformer_blocks.1.ff_context.net.2",
    "transformer.transformer_blocks.2.norm1.linear",
    "transformer.transformer_blocks.2.norm1_context.linear",
    "transformer.transformer_blocks.2.attn.to_q",
    "transformer.transformer_blocks.2.attn.to_k",
    "transformer.transformer_blocks.2.attn.to_v",
    "transformer.transformer_blocks.2.attn.add_k_proj",
    "transformer.transformer_blocks.2.attn.add_v_proj",
    "transformer.transformer_blocks.2.attn.add_q_proj",
    "transformer.transformer_blocks.2.attn.to_out.0",
    "transformer.transformer_blocks.2.attn.to_add_out",
    "transformer.transformer_blocks.2.ff.net.0.proj",
    "transformer.transformer_blocks.2.ff.net.2",
    "transformer.transformer_blocks.2.ff_context.net.0.proj",
    "transformer.transformer_blocks.2.ff_context.net.2",
    "transformer.transformer_blocks.3.norm1.linear",
    "transformer.transformer_blocks.3.norm1_context.linear",
    "transformer.transformer_blocks.3.attn.to_q",
    "transformer.transformer_blocks.3.attn.to_k",
    "transformer.transformer_blocks.3.attn.to_v",
    "transformer.transformer_blocks.3.attn.add_k_proj",
    "transformer.transformer_blocks.3.attn.add_v_proj",
    "transformer.transformer_blocks.3.attn.add_q_proj",
    "transformer.transformer_blocks.3.attn.to_out.0",
    "transformer.transformer_blocks.3.attn.to_add_out",
    "transformer.transformer_blocks.3.ff.net.0.proj",
    "transformer.transformer_blocks.3.ff.net.2",
    "transformer.transformer_blocks.3.ff_context.net.0.proj",
    "transformer.transformer_blocks.3.ff_context.net.2",
    "transformer.transformer_blocks.4.norm1.linear",
    "transformer.transformer_blocks.4.norm1_context.linear",
    "transformer.transformer_blocks.4.attn.to_q",
    "transformer.transformer_blocks.4.attn.to_k",
    "transformer.transformer_blocks.4.attn.to_v",
    "transformer.transformer_blocks.4.attn.add_k_proj",
    "transformer.transformer_blocks.4.attn.add_v_proj",
    "transformer.transformer_blocks.4.attn.add_q_proj",
    "transformer.transformer_blocks.4.attn.to_out.0",
    "transformer.transformer_blocks.4.attn.to_add_out",
    "transformer.transformer_blocks.4.ff.net.0.proj",
    "transformer.transformer_blocks.4.ff.net.2",
    "transformer.transformer_blocks.4.ff_context.net.0.proj",
    "transformer.transformer_blocks.4.ff_context.net.2",
    "transformer.transformer_blocks.5.norm1.linear",
    "transformer.transformer_blocks.5.norm1_context.linear",
    "transformer.transformer_blocks.5.attn.to_q",
    "transformer.transformer_blocks.5.attn.to_k",
    "transformer.transformer_blocks.5.attn.to_v",
    "transformer.transformer_blocks.5.attn.add_k_proj",
    "transformer.transformer_blocks.5.attn.add_v_proj",
    "transformer.transformer_blocks.5.attn.add_q_proj",
    "transformer.transformer_blocks.5.attn.to_out.0",
    "transformer.transformer_blocks.5.attn.to_add_out",
    "transformer.transformer_blocks.5.ff.net.0.proj",
    "transformer.transformer_blocks.5.ff.net.2",
    "transformer.transformer_blocks.5.ff_context.net.0.proj",
    "transformer.transformer_blocks.5.ff_context.net.2",
    "transformer.transformer_blocks.6.norm1.linear",
    "transformer.transformer_blocks.6.norm1_context.linear",
    "transformer.transformer_blocks.6.attn.to_q",
    "transformer.transformer_blocks.6.attn.to_k",
    "transformer.transformer_blocks.6.attn.to_v",
    "transformer.transformer_blocks.6.attn.add_k_proj",
    "transformer.transformer_blocks.6.attn.add_v_proj",
    "transformer.transformer_blocks.6.attn.add_q_proj",
    "transformer.transformer_blocks.6.attn.to_out.0",
    "transformer.transformer_blocks.6.attn.to_add_out",
    "transformer.transformer_blocks.6.ff.net.0.proj",
    "transformer.transformer_blocks.6.ff.net.2",
    "transformer.transformer_blocks.6.ff_context.net.0.proj",
    "transformer.transformer_blocks.6.ff_context.net.2",
    "transformer.transformer_blocks.7.norm1.linear",
    "transformer.transformer_blocks.7.norm1_context.linear",
    "transformer.transformer_blocks.7.attn.to_q",
    "transformer.transformer_blocks.7.attn.to_k",
    "transformer.transformer_blocks.7.attn.to_v",
    "transformer.transformer_blocks.7.attn.add_k_proj",
    "transformer.transformer_blocks.7.attn.add_v_proj",
    "transformer.transformer_blocks.7.attn.add_q_proj",
    "transformer.transformer_blocks.7.attn.to_out.0",
    "transformer.transformer_blocks.7.attn.to_add_out",
    "transformer.transformer_blocks.7.ff.net.0.proj",
    "transformer.transformer_blocks.7.ff.net.2",
    "transformer.transformer_blocks.7.ff_context.net.0.proj",
    "transformer.transformer_blocks.7.ff_context.net.2",
    "transformer.transformer_blocks.8.norm1.linear",
    "transformer.transformer_blocks.8.norm1_context.linear",
    "transformer.transformer_blocks.8.attn.to_q",
    "transformer.transformer_blocks.8.attn.to_k",
    "transformer.transformer_blocks.8.attn.to_v",
    "transformer.transformer_blocks.8.attn.add_k_proj",
    "transformer.transformer_blocks.8.attn.add_v_proj",
    "transformer.transformer_blocks.8.attn.add_q_proj",
    "transformer.transformer_blocks.8.attn.to_out.0",
    "transformer.transformer_blocks.8.attn.to_add_out",
    "transformer.transformer_blocks.8.ff.net.0.proj",
    "transformer.transformer_blocks.8.ff.net.2",
    "transformer.transformer_blocks.8.ff_context.net.0.proj",
    "transformer.transformer_blocks.8.ff_context.net.2",
    "transformer.transformer_blocks.9.norm1.linear",
    "transformer.transformer_blocks.9.norm1_context.linear",
    "transformer.transformer_blocks.9.attn.to_q",
    "transformer.transformer_blocks.9.attn.to_k",
    "transformer.transformer_blocks.9.attn.to_v",
    "transformer.transformer_blocks.9.attn.add_k_proj",
    "transformer.transformer_blocks.9.attn.add_v_proj",
    "transformer.transformer_blocks.9.attn.add_q_proj",
    "transformer.transformer_blocks.9.attn.to_out.0",
    "transformer.transformer_blocks.9.attn.to_add_out",
    "transformer.transformer_blocks.9.ff.net.0.proj",
    "transformer.transformer_blocks.9.ff.net.2",
    "transformer.transformer_blocks.9.ff_context.net.0.proj",
    "transformer.transformer_blocks.9.ff_context.net.2",
    "transformer.transformer_blocks.10.norm1.linear",
    "transformer.transformer_blocks.10.norm1_context.linear",
    "transformer.transformer_blocks.10.attn.to_q",
    "transformer.transformer_blocks.10.attn.to_k",
    "transformer.transformer_blocks.10.attn.to_v",
    "transformer.transformer_blocks.10.attn.add_k_proj",
    "transformer.transformer_blocks.10.attn.add_v_proj",
    "transformer.transformer_blocks.10.attn.add_q_proj",
    "transformer.transformer_blocks.10.attn.to_out.0",
    "transformer.transformer_blocks.10.attn.to_add_out",
    "transformer.transformer_blocks.10.ff.net.0.proj",
    "transformer.transformer_blocks.10.ff.net.2",
    "transformer.transformer_blocks.10.ff_context.net.0.proj",
    "transformer.transformer_blocks.10.ff_context.net.2",
    "transformer.transformer_blocks.11.norm1.linear",
    "transformer.transformer_blocks.11.norm1_context.linear",
    "transformer.transformer_blocks.11.attn.to_q",
    "transformer.transformer_blocks.11.attn.to_k",
    "transformer.transformer_blocks.11.attn.to_v",
    "transformer.transformer_blocks.11.attn.add_k_proj",
    "transformer.transformer_blocks.11.attn.add_v_proj",
    "transformer.transformer_blocks.11.attn.add_q_proj",
    "transformer.transformer_blocks.11.attn.to_out.0",
    "transformer.transformer_blocks.11.attn.to_add_out",
    "transformer.transformer_blocks.11.ff.net.0.proj",
    "transformer.transformer_blocks.11.ff.net.2",
    "transformer.transformer_blocks.11.ff_context.net.0.proj",
    "transformer.transformer_blocks.11.ff_context.net.2",
    "transformer.transformer_blocks.12.norm1.linear",
    "transformer.transformer_blocks.12.norm1_context.linear",
    "transformer.transformer_blocks.12.attn.to_q",
    "transformer.transformer_blocks.12.attn.to_k",
    "transformer.transformer_blocks.12.attn.to_v",
    "transformer.transformer_blocks.12.attn.add_k_proj",
    "transformer.transformer_blocks.12.attn.add_v_proj",
    "transformer.transformer_blocks.12.attn.add_q_proj",
    "transformer.transformer_blocks.12.attn.to_out.0",
    "transformer.transformer_blocks.12.attn.to_add_out",
    "transformer.transformer_blocks.12.ff.net.0.proj",
    "transformer.transformer_blocks.12.ff.net.2",
    "transformer.transformer_blocks.12.ff_context.net.0.proj",
    "transformer.transformer_blocks.12.ff_context.net.2",
    "transformer.transformer_blocks.13.norm1.linear",
    "transformer.transformer_blocks.13.norm1_context.linear",
    "transformer.transformer_blocks.13.attn.to_q",
    "transformer.transformer_blocks.13.attn.to_k",
    "transformer.transformer_blocks.13.attn.to_v",
    "transformer.transformer_blocks.13.attn.add_k_proj",
    "transformer.transformer_blocks.13.attn.add_v_proj",
    "transformer.transformer_blocks.13.attn.add_q_proj",
    "transformer.transformer_blocks.13.attn.to_out.0",
    "transformer.transformer_blocks.13.attn.to_add_out",
    "transformer.transformer_blocks.13.ff.net.0.proj",
    "transformer.transformer_blocks.13.ff.net.2",
    "transformer.transformer_blocks.13.ff_context.net.0.proj",
    "transformer.transformer_blocks.13.ff_context.net.2",
    "transformer.transformer_blocks.14.norm1.linear",
    "transformer.transformer_blocks.14.norm1_context.linear",
    "transformer.transformer_blocks.14.attn.to_q",
    "transformer.transformer_blocks.14.attn.to_k",
    "transformer.transformer_blocks.14.attn.to_v",
    "transformer.transformer_blocks.14.attn.add_k_proj",
    "transformer.transformer_blocks.14.attn.add_v_proj",
    "transformer.transformer_blocks.14.attn.add_q_proj",
    "transformer.transformer_blocks.14.attn.to_out.0",
    "transformer.transformer_blocks.14.attn.to_add_out",
    "transformer.transformer_blocks.14.ff.net.0.proj",
    "transformer.transformer_blocks.14.ff.net.2",
    "transformer.transformer_blocks.14.ff_context.net.0.proj",
    "transformer.transformer_blocks.14.ff_context.net.2",
    "transformer.transformer_blocks.15.norm1.linear",
    "transformer.transformer_blocks.15.norm1_context.linear",
    "transformer.transformer_blocks.15.attn.to_q",
    "transformer.transformer_blocks.15.attn.to_k",
    "transformer.transformer_blocks.15.attn.to_v",
    "transformer.transformer_blocks.15.attn.add_k_proj",
    "transformer.transformer_blocks.15.attn.add_v_proj",
    "transformer.transformer_blocks.15.attn.add_q_proj",
    "transformer.transformer_blocks.15.attn.to_out.0",
    "transformer.transformer_blocks.15.attn.to_add_out",
    "transformer.transformer_blocks.15.ff.net.0.proj",
    "transformer.transformer_blocks.15.ff.net.2",
    "transformer.transformer_blocks.15.ff_context.net.0.proj",
    "transformer.transformer_blocks.15.ff_context.net.2",
    "transformer.transformer_blocks.16.norm1.linear",
    "transformer.transformer_blocks.16.norm1_context.linear",
    "transformer.transformer_blocks.16.attn.to_q",
    "transformer.transformer_blocks.16.attn.to_k",
    "transformer.transformer_blocks.16.attn.to_v",
    "transformer.transformer_blocks.16.attn.add_k_proj",
    "transformer.transformer_blocks.16.attn.add_v_proj",
    "transformer.transformer_blocks.16.attn.add_q_proj",
    "transformer.transformer_blocks.16.attn.to_out.0",
    "transformer.transformer_blocks.16.attn.to_add_out",
    "transformer.transformer_blocks.16.ff.net.0.proj",
    "transformer.transformer_blocks.16.ff.net.2",
    "transformer.transformer_blocks.16.ff_context.net.0.proj",
    "transformer.transformer_blocks.16.ff_context.net.2",
    "transformer.transformer_blocks.17.norm1.linear",
    "transformer.transformer_blocks.17.norm1_context.linear",
    "transformer.transformer_blocks.17.attn.to_q",
    "transformer.transformer_blocks.17.attn.to_k",
    "transformer.transformer_blocks.17.attn.to_v",
    "transformer.transformer_blocks.17.attn.add_k_proj",
    "transformer.transformer_blocks.17.attn.add_v_proj",
    "transformer.transformer_blocks.17.attn.add_q_proj",
    "transformer.transformer_blocks.17.attn.to_out.0",
    "transformer.transformer_blocks.17.attn.to_add_out",
    "transformer.transformer_blocks.17.ff.net.0.proj",
    "transformer.transformer_blocks.17.ff.net.2",
    "transformer.transformer_blocks.17.ff_context.net.0.proj",
    "transformer.transformer_blocks.17.ff_context.net.2",
    "transformer.transformer_blocks.18.norm1.linear",
    "transformer.transformer_blocks.18.norm1_context.linear",
    "transformer.transformer_blocks.18.attn.to_q",
    "transformer.transformer_blocks.18.attn.to_k",
    "transformer.transformer_blocks.18.attn.to_v",
    "transformer.transformer_blocks.18.attn.add_k_proj",
    "transformer.transformer_blocks.18.attn.add_v_proj",
    "transformer.transformer_blocks.18.attn.add_q_proj",
    "transformer.transformer_blocks.18.attn.to_out.0",
    "transformer.transformer_blocks.18.attn.to_add_out",
    "transformer.transformer_blocks.18.ff.net.0.proj",
    "transformer.transformer_blocks.18.ff.net.2",
    "transformer.transformer_blocks.18.ff_context.net.0.proj",
    "transformer.transformer_blocks.18.ff_context.net.2",
    "transformer.single_transformer_blocks.0.norm.linear",
    "transformer.single_transformer_blocks.0.proj_mlp",
    "transformer.single_transformer_blocks.0.proj_out",
    "transformer.single_transformer_blocks.0.attn.to_q",
    "transformer.single_transformer_blocks.0.attn.to_k",
    "transformer.single_transformer_blocks.0.attn.to_v",
    "transformer.single_transformer_blocks.1.norm.linear",
    "transformer.single_transformer_blocks.1.proj_mlp",
    "transformer.single_transformer_blocks.1.proj_out",
    "transformer.single_transformer_blocks.1.attn.to_q",
    "transformer.single_transformer_blocks.1.attn.to_k",
    "transformer.single_transformer_blocks.1.attn.to_v",
    "transformer.single_transformer_blocks.2.norm.linear",
    "transformer.single_transformer_blocks.2.proj_mlp",
    "transformer.single_transformer_blocks.2.proj_out",
    "transformer.single_transformer_blocks.2.attn.to_q",
    "transformer.single_transformer_blocks.2.attn.to_k",
    "transformer.single_transformer_blocks.2.attn.to_v",
    "transformer.single_transformer_blocks.3.norm.linear",
    "transformer.single_transformer_blocks.3.proj_mlp",
    "transformer.single_transformer_blocks.3.proj_out",
    "transformer.single_transformer_blocks.3.attn.to_q",
    "transformer.single_transformer_blocks.3.attn.to_k",
    "transformer.single_transformer_blocks.3.attn.to_v",
    "transformer.single_transformer_blocks.4.norm.linear",
    "transformer.single_transformer_blocks.4.proj_mlp",
    "transformer.single_transformer_blocks.4.proj_out",
    "transformer.single_transformer_blocks.4.attn.to_q",
    "transformer.single_transformer_blocks.4.attn.to_k",
    "transformer.single_transformer_blocks.4.attn.to_v",
    "transformer.single_transformer_blocks.5.norm.linear",
    "transformer.single_transformer_blocks.5.proj_mlp",
    "transformer.single_transformer_blocks.5.proj_out",
    "transformer.single_transformer_blocks.5.attn.to_q",
    "transformer.single_transformer_blocks.5.attn.to_k",
    "transformer.single_transformer_blocks.5.attn.to_v",
    "transformer.single_transformer_blocks.6.norm.linear",
    "transformer.single_transformer_blocks.6.proj_mlp",
    "transformer.single_transformer_blocks.6.proj_out",
    "transformer.single_transformer_blocks.6.attn.to_q",
    "transformer.single_transformer_blocks.6.attn.to_k",
    "transformer.single_transformer_blocks.6.attn.to_v",
    "transformer.single_transformer_blocks.7.norm.linear",
    "transformer.single_transformer_blocks.7.proj_mlp",
    "transformer.single_transformer_blocks.7.proj_out",
    "transformer.single_transformer_blocks.7.attn.to_q",
    "transformer.single_transformer_blocks.7.attn.to_k",
    "transformer.single_transformer_blocks.7.attn.to_v",
    "transformer.single_transformer_blocks.8.norm.linear",
    "transformer.single_transformer_blocks.8.proj_mlp",
    "transformer.single_transformer_blocks.8.proj_out",
    "transformer.single_transformer_blocks.8.attn.to_q",
    "transformer.single_transformer_blocks.8.attn.to_k",
    "transformer.single_transformer_blocks.8.attn.to_v",
    "transformer.single_transformer_blocks.9.norm.linear",
    "transformer.single_transformer_blocks.9.proj_mlp",
    "transformer.single_transformer_blocks.9.proj_out",
    "transformer.single_transformer_blocks.9.attn.to_q",
    "transformer.single_transformer_blocks.9.attn.to_k",
    "transformer.single_transformer_blocks.9.attn.to_v",
    "transformer.single_transformer_blocks.10.norm.linear",
    "transformer.single_transformer_blocks.10.proj_mlp",
    "transformer.single_transformer_blocks.10.proj_out",
    "transformer.single_transformer_blocks.10.attn.to_q",
    "transformer.single_transformer_blocks.10.attn.to_k",
    "transformer.single_transformer_blocks.10.attn.to_v",
    "transformer.single_transformer_blocks.11.norm.linear",
    "transformer.single_transformer_blocks.11.proj_mlp",
    "transformer.single_transformer_blocks.11.proj_out",
    "transformer.single_transformer_blocks.11.attn.to_q",
    "transformer.single_transformer_blocks.11.attn.to_k",
    "transformer.single_transformer_blocks.11.attn.to_v",
    "transformer.single_transformer_blocks.12.norm.linear",
    "transformer.single_transformer_blocks.12.proj_mlp",
    "transformer.single_transformer_blocks.12.proj_out",
    "transformer.single_transformer_blocks.12.attn.to_q",
    "transformer.single_transformer_blocks.12.attn.to_k",
    "transformer.single_transformer_blocks.12.attn.to_v",
    "transformer.single_transformer_blocks.13.norm.linear",
    "transformer.single_transformer_blocks.13.proj_mlp",
    "transformer.single_transformer_blocks.13.proj_out",
    "transformer.single_transformer_blocks.13.attn.to_q",
    "transformer.single_transformer_blocks.13.attn.to_k",
    "transformer.single_transformer_blocks.13.attn.to_v",
    "transformer.single_transformer_blocks.14.norm.linear",
    "transformer.single_transformer_blocks.14.proj_mlp",
    "transformer.single_transformer_blocks.14.proj_out",
    "transformer.single_transformer_blocks.14.attn.to_q",
    "transformer.single_transformer_blocks.14.attn.to_k",
    "transformer.single_transformer_blocks.14.attn.to_v",
    "transformer.single_transformer_blocks.15.norm.linear",
    "transformer.single_transformer_blocks.15.proj_mlp",
    "transformer.single_transformer_blocks.15.proj_out",
    "transformer.single_transformer_blocks.15.attn.to_q",
    "transformer.single_transformer_blocks.15.attn.to_k",
    "transformer.single_transformer_blocks.15.attn.to_v",
    "transformer.single_transformer_blocks.16.norm.linear",
    "transformer.single_transformer_blocks.16.proj_mlp",
    "transformer.single_transformer_blocks.16.proj_out",
    "transformer.single_transformer_blocks.16.attn.to_q",
    "transformer.single_transformer_blocks.16.attn.to_k",
    "transformer.single_transformer_blocks.16.attn.to_v",
    "transformer.single_transformer_blocks.17.norm.linear",
    "transformer.single_transformer_blocks.17.proj_mlp",
    "transformer.single_transformer_blocks.17.proj_out",
    "transformer.single_transformer_blocks.17.attn.to_q",
    "transformer.single_transformer_blocks.17.attn.to_k",
    "transformer.single_transformer_blocks.17.attn.to_v",
    "transformer.single_transformer_blocks.18.norm.linear",
    "transformer.single_transformer_blocks.18.proj_mlp",
    "transformer.single_transformer_blocks.18.proj_out",
    "transformer.single_transformer_blocks.18.attn.to_q",
    "transformer.single_transformer_blocks.18.attn.to_k",
    "transformer.single_transformer_blocks.18.attn.to_v",
    "transformer.single_transformer_blocks.19.norm.linear",
    "transformer.single_transformer_blocks.19.proj_mlp",
    "transformer.single_transformer_blocks.19.proj_out",
    "transformer.single_transformer_blocks.19.attn.to_q",
    "transformer.single_transformer_blocks.19.attn.to_k",
    "transformer.single_transformer_blocks.19.attn.to_v",
    "transformer.single_transformer_blocks.20.norm.linear",
    "transformer.single_transformer_blocks.20.proj_mlp",
    "transformer.single_transformer_blocks.20.proj_out",
    "transformer.single_transformer_blocks.20.attn.to_q",
    "transformer.single_transformer_blocks.20.attn.to_k",
    "transformer.single_transformer_blocks.20.attn.to_v",
    "transformer.single_transformer_blocks.21.norm.linear",
    "transformer.single_transformer_blocks.21.proj_mlp",
    "transformer.single_transformer_blocks.21.proj_out",
    "transformer.single_transformer_blocks.21.attn.to_q",
    "transformer.single_transformer_blocks.21.attn.to_k",
    "transformer.single_transformer_blocks.21.attn.to_v",
    "transformer.single_transformer_blocks.22.norm.linear",
    "transformer.single_transformer_blocks.22.proj_mlp",
    "transformer.single_transformer_blocks.22.proj_out",
    "transformer.single_transformer_blocks.22.attn.to_q",
    "transformer.single_transformer_blocks.22.attn.to_k",
    "transformer.single_transformer_blocks.22.attn.to_v",
    "transformer.single_transformer_blocks.23.norm.linear",
    "transformer.single_transformer_blocks.23.proj_mlp",
    "transformer.single_transformer_blocks.23.proj_out",
    "transformer.single_transformer_blocks.23.attn.to_q",
    "transformer.single_transformer_blocks.23.attn.to_k",
    "transformer.single_transformer_blocks.23.attn.to_v",
    "transformer.single_transformer_blocks.24.norm.linear",
    "transformer.single_transformer_blocks.24.proj_mlp",
    "transformer.single_transformer_blocks.24.proj_out",
    "transformer.single_transformer_blocks.24.attn.to_q",
    "transformer.single_transformer_blocks.24.attn.to_k",
    "transformer.single_transformer_blocks.24.attn.to_v",
    "transformer.single_transformer_blocks.25.norm.linear",
    "transformer.single_transformer_blocks.25.proj_mlp",
    "transformer.single_transformer_blocks.25.proj_out",
    "transformer.single_transformer_blocks.25.attn.to_q",
    "transformer.single_transformer_blocks.25.attn.to_k",
    "transformer.single_transformer_blocks.25.attn.to_v",
    "transformer.single_transformer_blocks.26.norm.linear",
    "transformer.single_transformer_blocks.26.proj_mlp",
    "transformer.single_transformer_blocks.26.proj_out",
    "transformer.single_transformer_blocks.26.attn.to_q",
    "transformer.single_transformer_blocks.26.attn.to_k",
    "transformer.single_transformer_blocks.26.attn.to_v",
    "transformer.single_transformer_blocks.27.norm.linear",
    "transformer.single_transformer_blocks.27.proj_mlp",
    "transformer.single_transformer_blocks.27.proj_out",
    "transformer.single_transformer_blocks.27.attn.to_q",
    "transformer.single_transformer_blocks.27.attn.to_k",
    "transformer.single_transformer_blocks.27.attn.to_v",
    "transformer.single_transformer_blocks.28.norm.linear",
    "transformer.single_transformer_blocks.28.proj_mlp",
    "transformer.single_transformer_blocks.28.proj_out",
    "transformer.single_transformer_blocks.28.attn.to_q",
    "transformer.single_transformer_blocks.28.attn.to_k",
    "transformer.single_transformer_blocks.28.attn.to_v",
    "transformer.single_transformer_blocks.29.norm.linear",
    "transformer.single_transformer_blocks.29.proj_mlp",
    "transformer.single_transformer_blocks.29.proj_out",
    "transformer.single_transformer_blocks.29.attn.to_q",
    "transformer.single_transformer_blocks.29.attn.to_k",
    "transformer.single_transformer_blocks.29.attn.to_v",
    "transformer.single_transformer_blocks.30.norm.linear",
    "transformer.single_transformer_blocks.30.proj_mlp",
    "transformer.single_transformer_blocks.30.proj_out",
    "transformer.single_transformer_blocks.30.attn.to_q",
    "transformer.single_transformer_blocks.30.attn.to_k",
    "transformer.single_transformer_blocks.30.attn.to_v",
    "transformer.single_transformer_blocks.31.norm.linear",
    "transformer.single_transformer_blocks.31.proj_mlp",
    "transformer.single_transformer_blocks.31.proj_out",
    "transformer.single_transformer_blocks.31.attn.to_q",
    "transformer.single_transformer_blocks.31.attn.to_k",
    "transformer.single_transformer_blocks.31.attn.to_v",
    "transformer.single_transformer_blocks.32.norm.linear",
    "transformer.single_transformer_blocks.32.proj_mlp",
    "transformer.single_transformer_blocks.32.proj_out",
    "transformer.single_transformer_blocks.32.attn.to_q",
    "transformer.single_transformer_blocks.32.attn.to_k",
    "transformer.single_transformer_blocks.32.attn.to_v",
    "transformer.single_transformer_blocks.33.norm.linear",
    "transformer.single_transformer_blocks.33.proj_mlp",
    "transformer.single_transformer_blocks.33.proj_out",
    "transformer.single_transformer_blocks.33.attn.to_q",
    "transformer.single_transformer_blocks.33.attn.to_k",
    "transformer.single_transformer_blocks.33.attn.to_v",
    "transformer.single_transformer_blocks.34.norm.linear",
    "transformer.single_transformer_blocks.34.proj_mlp",
    "transformer.single_transformer_blocks.34.proj_out",
    "transformer.single_transformer_blocks.34.attn.to_q",
    "transformer.single_transformer_blocks.34.attn.to_k",
    "transformer.single_transformer_blocks.34.attn.to_v",
    "transformer.single_transformer_blocks.35.norm.linear",
    "transformer.single_transformer_blocks.35.proj_mlp",
    "transformer.single_transformer_blocks.35.proj_out",
    "transformer.single_transformer_blocks.35.attn.to_q",
    "transformer.single_transformer_blocks.35.attn.to_k",
    "transformer.single_transformer_blocks.35.attn.to_v",
    "transformer.single_transformer_blocks.36.norm.linear",
    "transformer.single_transformer_blocks.36.proj_mlp",
    "transformer.single_transformer_blocks.36.proj_out",
    "transformer.single_transformer_blocks.36.attn.to_q",
    "transformer.single_transformer_blocks.36.attn.to_k",
    "transformer.single_transformer_blocks.36.attn.to_v",
    "transformer.single_transformer_blocks.37.norm.linear",
    "transformer.single_transformer_blocks.37.proj_mlp",
    "transformer.single_transformer_blocks.37.proj_out",
    "transformer.single_transformer_blocks.37.attn.to_q",
    "transformer.single_transformer_blocks.37.attn.to_k",
    "transformer.single_transformer_blocks.37.attn.to_v",
]


# %%


prompt_categories = {
    'facial_features': [
        "smiling portrait", "frowning expression", "surprised look", "angry glare", "neutral expression",
        "bearded individual", "freckled complexion", "face", "laughing face", "crying expression",
        "confused look", "excited smile", "thoughtful gaze", "wrinkled face", "dimpled cheeks",
        "strong jawline", "high cheekbones", "bushy eyebrows",
    ],
    'nature': [
        "landscape", "seascape", "mountain range", "forest", "sunset", "sunrise", "beach", "desert",
        "waterfall", "canyon", "meadow", "tropical island", "arctic tundra", "savanna", "volcano",
        "coral reef", "rainforest", "autumn leaves", "spring blossoms", "misty morning",
    ],
    'animals': [
        "dog", "cat", "bird", "fish", "elephant", "lion", "giraffe", "penguin", "dolphin", "butterfly",
        "owl", "tiger", "panda", "koala", "kangaroo", "zebra", "rhinoceros", "cheetah", "polar bear",
    ],
    'food': [
        "apple", "banana", "pizza", "sushi", "pasta", "steak", "salad", "ice cream", "chocolate",
        "burger", "taco", "soup", "cake", "fruit platter", "sandwich", "roast chicken", "seafood platter",
    ],
    'music': [
        "guitar", "piano", "violin", "drums", "saxophone", "trumpet", "flute", "cello", "harp",
        "clarinet", "accordion", "electric guitar", "synthesizer", "DJ turntables", "orchestra",
        "rock band", "jazz ensemble",
    ],
    'architecture': [
        "skyscraper", "castle", "bridge", "ancient ruins", "modern house", "cathedral",
        "pagoda", "lighthouse", "pyramid", "log cabin", "treehouse", "futuristic city",
    ],
    'vehicles': [
        "car", "airplane", "bicycle", "motorcycle", "boat", "train", "helicopter",
        "rocket", "submarine", "hot air balloon", "sailboat", "vintage car",
    ],
    'abstract_concepts': [
        "love", "time", "freedom", "peace", "chaos", "infinity", "balance",
        "evolution", "imagination", "dreams", "consciousness", "harmony",
    ],
}

# %%
import torch
import numpy as np
import pandas as pd
from diffusers import FluxPipeline
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import gc
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
import plotly.express as px
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

# Set a fixed seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# %%
# Directory to save plots
plot_dir = 'plots'

# Create directory if it doesn't exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# %%

# Define the MultiPromptActivityTracker class to capture layer activations
class MultiPromptActivityTracker:
    def __init__(self, layers_to_track):
        self.layers_to_track = layers_to_track
        self.outputs = {}
        self.handles = []

    def hook_fn(self, name):
        def hook(module, input, output):
            # Move output to CPU and avoid retaining GPU tensors
            if isinstance(output, torch.Tensor):
                self.outputs[name] = output.detach().cpu()
            elif isinstance(output, tuple):
                tensors = [t.detach().cpu() for t in output if isinstance(t, torch.Tensor)]
                self.outputs[name] = torch.cat(tensors)
        return hook

    def register_hooks(self, pipe):
        self.handles = []
        for layer_name in self.layers_to_track:
            parts = layer_name.split('.')
            module = pipe
            for part in parts:
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    print(f"Module '{module}' has no attribute '{part}'")
                    break
            else:
                handle = module.register_forward_hook(self.hook_fn(layer_name))
                self.handles.append(handle)

    def unregister_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def run_prompt(self, pipe, prompt, height=512, width=512, generator=None):
        self.outputs.clear()
        self.register_hooks(pipe)
        with torch.no_grad():
            _ = pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=1,
                generator=generator
            )
        self.unregister_hooks()
        torch.cuda.empty_cache()
        return self.outputs.copy()


# %%

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")


# %%
# Initialize the tracker
tracker = MultiPromptActivityTracker(available_layers_to_optimize)

# Create the generator with the fixed seed
generator = torch.Generator("cuda").manual_seed(SEED)

# Create a flat list of all prompts
prompts = [prompt for prompts_list in prompt_categories.values() for prompt in prompts_list]

# Assign a category label to each prompt for visualization
prompt_labels = {prompt: category for category, prompts_list in prompt_categories.items() for prompt in prompts_list}

# Collect feature vectors for all prompts
feature_vectors = {}
with tqdm(total=len(prompts), desc="Processing prompts", unit="prompt") as pbar:
    for prompt in prompts:
        outputs = tracker.run_prompt(pipe, prompt, generator=generator, height=512, width=512)

        # Compute feature vector for the prompt
        features = []
        for layer_name in available_layers_to_optimize:
            output = outputs.get(layer_name)
            if output is not None:
                activity = output.mean().item()
                features.append(activity)
            else:
                features.append(0.0)
        feature_vectors[prompt] = features

        # Clear variables and GPU memory
        del outputs
        torch.cuda.empty_cache()
        gc.collect()

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({"Current prompt": prompt[:30] + "..." if len(prompt) > 30 else prompt})


# %%
# Convert feature vectors to a DataFrame
df_features = pd.DataFrame.from_dict(feature_vectors, orient='index', columns=available_layers_to_optimize)
# Save the DataFrame to disk, preserving dtypes
df_features.to_pickle('feature_vectors.pkl')

# Uncomment the following line to load the DataFrame from disk in future runs:
# df_features = pd.read_pickle('feature_vectors.pkl')

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# Prepare labels for binary classification
selected_category = 'facial_features'  # The category we're focusing on
y_binary = np.array([1 if prompt_labels[prompt] == selected_category else 0 for prompt in df_features.index])

# Add category labels to the features DataFrame
df_features['category'] = df_features.index.map(prompt_labels)

# %% [markdown]
# ### Additional Plots and Analysis

# %%
# 1. Heatmap of Correlation Between Layers

# Compute correlation matrix
corr_matrix = df_features.drop('category', axis=1).corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='viridis', square=True)
plt.title('Correlation Heatmap of Layer Activations')
plt.tight_layout()
plt.savefig(f'{plot_dir}/correlation_heatmap.png')
plt.show()

# %%
# 2. Distribution of Mean Activations per Layer

for layer in df_features.columns[:-1]:  # Exclude 'category' column
    plt.figure(figsize=(8, 4))
    sns.histplot(df_features[layer], kde=True)
    plt.title(f'Distribution of Mean Activations for {layer}')
    plt.xlabel('Mean Activation')
    plt.ylabel('Frequency')
    plt.tight_layout()
    layer_name_safe = layer.replace('.', '_')
    plt.savefig(f'{plot_dir}/activation_distribution_{layer_name_safe}.png')
    plt.show()

# %%
# 3. Box Plots of Layer Activations by Category

for layer in df_features.columns[:-1]:  # Exclude 'category' column
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='category', y=layer, data=df_features)
    plt.title(f'Layer Activation by Category for {layer}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    layer_name_safe = layer.replace('.', '_')
    plt.savefig(f'{plot_dir}/boxplot_{layer_name_safe}.png')
    plt.show()

# %%
# Standardize the features (without 'category' column)
X_scaled = scaler.fit_transform(df_features.drop('category', axis=1))

# %% [markdown]
# ### Dimensionality Reduction and Visualization

# %%
# Apply UMAP with 3 components
reducer_umap = umap.UMAP(
    n_components=3,
    n_neighbors=5,
    random_state=SEED,
    metric='cosine'
)
embeddings_umap = reducer_umap.fit_transform(X_scaled)

# Create DataFrame for UMAP embeddings
embedding_df_umap = pd.DataFrame(embeddings_umap, columns=['Dim1', 'Dim2', 'Dim3'])
embedding_df_umap['prompt'] = df_features.index
embedding_df_umap['category'] = embedding_df_umap['prompt'].map(prompt_labels)

# Visualize UMAP embeddings
fig_umap = px.scatter_3d(
    embedding_df_umap, x='Dim1', y='Dim2', z='Dim3',
    color='category', hover_data=['prompt'],
    title='3D UMAP Embedding of Prompt Mean Layer Activities'
)
fig_umap.show()
fig_umap.write_image(f'{plot_dir}/umap_embedding_3d.png')

# %%
# Apply PCA with 3 components
pca = PCA(n_components=3)
embeddings_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_

# Create DataFrame for PCA embeddings
embedding_df_pca = pd.DataFrame(embeddings_pca, columns=['PC1', 'PC2', 'PC3'])
embedding_df_pca['prompt'] = df_features.index
embedding_df_pca['category'] = embedding_df_pca['prompt'].map(prompt_labels)

# Visualize PCA embeddings
fig_pca = px.scatter_3d(
    embedding_df_pca, x='PC1', y='PC2', z='PC3',
    color='category', hover_data=['prompt'],
    title='3D PCA Embedding of Prompt Mean Layer Activities'
)
fig_pca.show()
fig_pca.write_image(f'{plot_dir}/pca_embedding_3d.png')

# %%
# PCA Scree Plot
plt.figure()
plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'o-', linewidth=2)
plt.title('PCA Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(1, len(explained_variance_ratio) + 1))
plt.tight_layout()
plt.savefig(f'{plot_dir}/pca_scree_plot.png')
plt.show()

# %% [markdown]
# ### Model Training and Interpretation

# %%
# Logistic Regression with L1 Regularization
clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=SEED)
clf.fit(X_scaled, y_binary)

# Predict probabilities for ROC Curve
y_scores = clf.decision_function(X_scaled)

# ROC Curve
fpr, tpr, _ = roc_curve(y_binary, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression Model')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f'{plot_dir}/roc_curve_logistic_regression.png')
plt.show()

# %%
# Extract model coefficients
coef = clf.coef_[0]
coef_abs = np.abs(coef)

# Identify top layers based on coefficients
coef_df = pd.DataFrame({
    'layer': df_features.columns[:-1],  # Exclude 'category' column
    'importance': coef_abs
})
coef_df.sort_values(by='importance', ascending=False, inplace=True)
top_layers_logreg = coef_df.head(10)

# Visualize top layers from Logistic Regression
fig_logreg = px.bar(
    top_layers_logreg,
    x='importance',
    y='layer',
    orientation='h',
    title='Top Layers Identified by Logistic Regression Coefficients (Mean Layer Activity)',
    labels={'importance': 'Absolute Coefficient', 'layer': 'Layer'}
)
fig_logreg.update_layout(yaxis={'categoryorder': 'total ascending'})
fig_logreg.show()
fig_logreg.write_image(f'{plot_dir}/top_layers_logistic_regression.png')

# %%
# SHAP Analysis
explainer = shap.LinearExplainer(clf, X_scaled, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_scaled)
mean_shap_values = np.mean(np.abs(shap_values), axis=0)

# Identify top layers based on SHAP values
shap_df = pd.DataFrame({
    'layer': df_features.columns[:-1],  # Exclude 'category' column
    'mean_abs_shap_value': mean_shap_values
})
shap_df.sort_values(by='mean_abs_shap_value', ascending=False, inplace=True)
top_layers_shap = shap_df.head(10)

# Visualize top layers from SHAP analysis
fig_shap = px.bar(
    top_layers_shap,
    x='mean_abs_shap_value',
    y='layer',
    orientation='h',
    title='Top Layers Identified by SHAP Values (Mean Layer Activity)',
    labels={'mean_abs_shap_value': 'Mean Absolute SHAP Value', 'layer': 'Layer'}
)
fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'})
fig_shap.show()
fig_shap.write_image(f'{plot_dir}/top_layers_shap.png')

# SHAP Summary Plot
shap.summary_plot(shap_values, features=X_scaled, feature_names=df_features.columns[:-1], show=False)
plt.gcf().set_size_inches(10, 6)
plt.tight_layout()
plt.savefig(f'{plot_dir}/shap_summary_plot.png', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Layer Importance Comparison

# %%
# Compute contributions of each layer to each UMAP dimension
# Compute the Pearson correlation between each feature and each UMAP dimension
layer_contributions = pd.DataFrame(index=df_features.columns[:-1])  # Exclude 'category' column

for i in range(embeddings_umap.shape[1]):
    dim = embeddings_umap[:, i]
    correlations = []
    for idx in range(X_scaled.shape[1]):
        feature = X_scaled[:, idx]
        correlation = np.corrcoef(feature, dim)[0, 1]
        correlations.append(abs(correlation))
    layer_contributions[f'UMAP_Dim{i+1}'] = correlations

# Sum the absolute correlations across all UMAP dimensions
layer_contributions['umap_total_correlation'] = layer_contributions.sum(axis=1)

# Identify top layers based on UMAP contributions
layer_contributions_sorted = layer_contributions.sort_values(by='umap_total_correlation', ascending=False)
top_layers_umap = layer_contributions_sorted.head(10)

# Visualize top layers from UMAP contributions
fig_umap_contrib = px.bar(
    top_layers_umap.reset_index(),
    x='umap_total_correlation',
    y='index',
    orientation='h',
    title='Layers with Highest Correlation to UMAP Dimensions (Mean Layer Activity)',
    labels={'umap_total_correlation': 'Total Absolute Correlation', 'index': 'Layer'}
)
fig_umap_contrib.update_layout(yaxis={'categoryorder': 'total ascending'})
fig_umap_contrib.show()
fig_umap_contrib.write_image(f'{plot_dir}/top_layers_umap_correlation.png')

# %%
# Layer Importance Comparison Plot

# Normalize importance scores
coef_norm = coef_abs / np.max(coef_abs) if np.max(coef_abs) != 0 else coef_abs
shap_norm = mean_shap_values / np.max(mean_shap_values) if np.max(mean_shap_values) != 0 else mean_shap_values
umap_norm = layer_contributions['umap_total_correlation'] / layer_contributions['umap_total_correlation'].max() if layer_contributions['umap_total_correlation'].max() != 0 else layer_contributions['umap_total_correlation']

# Create DataFrame for combined importances
importance_df = pd.DataFrame({
    'Layer': df_features.columns[:-1],  # Exclude 'category' column
    'Logistic Regression Coef': coef_norm,
    'Mean SHAP Value': shap_norm,
    'UMAP Correlation': umap_norm
})

# Melt the DataFrame for plotting
importance_melted = importance_df.melt(id_vars='Layer', var_name='Importance Method', value_name='Normalized Importance')

# Plot the importance scores
fig_importance = px.bar(
    importance_melted.sort_values(by='Normalized Importance', ascending=False),
    x='Normalized Importance',
    y='Layer',
    color='Importance Method',
    orientation='h',
    title='Comparison of Layer Importance Scores from Different Methods'
)
fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
fig_importance.show()
fig_importance.write_image(f'{plot_dir}/layer_importance_comparison.png')

# %%
# Combine results for final recommendation
combined_importance = coef_norm + shap_norm + umap_norm.values

combined_df = pd.DataFrame({
    'layer': df_features.columns[:-1],  # Exclude 'category' column
    'combined_importance': combined_importance
})
combined_df.sort_values(by='combined_importance', ascending=False, inplace=True)
top_layers_combined = combined_df.head(10)

# Visualize top layers from combined analysis
fig_combined = px.bar(
    top_layers_combined,
    x='combined_importance',
    y='layer',
    orientation='h',
    title='Final Recommended Layers Based on Combined Importance (Mean Layer Activity)',
    labels={'combined_importance': 'Combined Importance Score', 'layer': 'Layer'}
)
fig_combined.update_layout(yaxis={'categoryorder': 'total ascending'})
fig_combined.show()
fig_combined.write_image(f'{plot_dir}/top_layers_combined_importance.png')

# %%
# Final recommendation
print("Final recommendation for layers to fine-tune:")
for idx, row in top_layers_combined.iterrows():
    print(f"{row['layer']} (Combined Importance: {row['combined_importance']:.4f})")

# %%
# Validate top layers with UMAP
top_layers = top_layers_combined['layer'].tolist()
X_top = df_features[top_layers]

# Standardize the features of top layers
scaler_top = StandardScaler()
X_top_scaled = scaler_top.fit_transform(X_top)

# Apply UMAP on top layers
embeddings_umap_top = reducer_umap.fit_transform(X_top_scaled)

# Create DataFrame for UMAP embeddings of top layers
embedding_df_umap_top = pd.DataFrame(embeddings_umap_top, columns=['Dim1', 'Dim2', 'Dim3'])
embedding_df_umap_top['prompt'] = df_features.index
embedding_df_umap_top['category'] = embedding_df_umap_top['prompt'].map(prompt_labels)

# Visualize UMAP embeddings for top layers
fig_umap_top = px.scatter_3d(
    embedding_df_umap_top, x='Dim1', y='Dim2', z='Dim3',
    color='category', hover_data=['prompt'],
    title='3D UMAP Embedding Using Only Top Recommended Layers (Mean Layer Activity)'
)
fig_umap_top.show()
fig_umap_top.write_image(f'{plot_dir}/umap_embedding_top_layers.png')

# %% [markdown]
# ### Additional Visualization: t-SNE Embedding

# %%
# t-SNE Visualization

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=SEED, perplexity=5, learning_rate='auto', init='random')
embeddings_tsne = tsne.fit_transform(X_scaled)

# Create DataFrame for t-SNE embeddings
embedding_df_tsne = pd.DataFrame(embeddings_tsne, columns=['Dim1', 'Dim2'])
embedding_df_tsne['prompt'] = df_features.index
embedding_df_tsne['category'] = embedding_df_tsne['prompt'].map(prompt_labels)

# Visualize t-SNE embeddings
fig_tsne = px.scatter(
    embedding_df_tsne, x='Dim1', y='Dim2',
    color='category', hover_data=['prompt'],
    title='t-SNE Embedding of Prompt Mean Layer Activities'
)
fig_tsne.show()
fig_tsne.write_image(f'{plot_dir}/tsne_embedding.png')
# %%

# %%
