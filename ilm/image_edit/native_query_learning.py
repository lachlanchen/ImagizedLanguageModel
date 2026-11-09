"""Let native continuous queries learn through an otherwise frozen reader."""
import torch

from .native_flow_step import pixels


def input_features(pipe, image_path, query_master):
    if query_master.shape != pipe.connector_module.meta_queries.shape or query_master.dtype != torch.float32:
        raise ValueError('native-shaped FP32 continuous query masters required')
    with torch.no_grad():
        image = pixels(pipe,image_path)
        features, grid = pipe.get_semantic_features_dynamic([image])
        reference = pipe.pixels_to_latents(image[None])[0]
    queries = query_master.to(dtype=pipe.transformer.dtype)[None].expand(2,pipe.num_queries,-1)
    inputs = pipe._ilm_condition(queries,image_embeds=torch.cat(features*2),image_grid_thw=grid.repeat(2,1))
    output = pipe.llm(**inputs,return_dict=True,output_hidden_states=True,use_cache=False)
    count = len(output.hidden_states)-1
    selected = list(range(count-1,0,-6))
    merged = torch.cat([output.hidden_states[i] for i in selected],dim=-1)
    if not torch.isfinite(merged).all() or not torch.isfinite(reference).all():
        raise FloatingPointError('nonfinite native continuous-query feature')
    return merged, reference.detach()
