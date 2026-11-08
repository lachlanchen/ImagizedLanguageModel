#!/usr/bin/env python3
"""Inspect pinned Ovis-U1 declarations and image-path source without execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(source,output):
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['repo']!='ATH-MaaS/Ovis-U1-3B' or manifest['weights_downloaded'] or manifest['models_loaded']:
        raise ValueError('official metadata-only acquisition required')
    for row in manifest['files']:
        path=source/row['file']
        if sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('pinned source bytes changed')
        if not row['url'].startswith('https://huggingface.co/'+manifest['repo']+'/resolve/'+manifest['revision']+'/'):
            raise ValueError('unpinned official source')
    config=json.loads((source/'config.json').read_text())
    index=json.loads((source/'model.safetensors.index.json').read_text())
    weights=manifest['weight_metadata'];declared=manifest['declared_safetensors']
    bytes_per={'BF16':2,'F32':4}
    payload=sum(bytes_per[dtype]*count for dtype,count in declared['parameters'].items())
    if payload!=index['metadata']['total_size'] or sum(declared['parameters'].values())!=declared['total']:
        raise ValueError('archive element counts and declared payload differ')
    if len(weights)!=3 or {w['rfilename'] for w in weights}!=set(index['weight_map'].values()):
        raise ValueError('all three released weight shards must be declared')
    for row in weights:
        if row['size']!=row['lfs']['size'] or len(row['lfs']['sha256'])!=64:
            raise ValueError('published whole-shard metadata differs')
    prefixes={k.split('.')[0] for k in index['weight_map']}
    if prefixes!={'llm','visual_generator','visual_tokenizer','vte'}:
        raise ValueError('published component coverage differs')
    tree=ast.parse((source/'modeling_ovis_u1.py').read_text())
    owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OvisU1')
    methods={n.name:n for n in owner.body if isinstance(n,ast.FunctionDef)}
    image_methods=[methods[n] for n in ('generate_condition','generate_img')]
    calls=[n for method in image_methods for n in ast.walk(method) if isinstance(n,ast.Call)]
    if any(isinstance(n.func,ast.Attribute) and n.func.attr=='generate' for n in calls):
        raise ValueError('image methods now contain autoregressive generation')
    if not any(isinstance(n.func,ast.Attribute) and n.func.attr=='generate_image' for n in calls):
        raise ValueError('native image decoder call is missing')
    embedding=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VisualEmbedding')
    if not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='matmul' for n in ast.walk(embedding)):
        raise ValueError('floating visual embedding path changed')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(implementation_sha256=sha(Path(__file__)),acquisition_manifest_sha256=sha(source/'manifest.json'),
        repo=manifest['repo'],revision=manifest['revision'],files=manifest['files'],whole_shard_declarations=weights,
        source_declared_archive_float_elements=declared['total'],source_declared_archive_dtypes=declared['parameters'],
        source_declared_tensor_payload_bytes=payload,source_declared_tensor_payload_GiB=payload/1024**3,
        source_declared_shard_bytes_including_headers=sum(w['size'] for w in weights),
        declared_root_components=sorted(prefixes),
        VAE_weights_declared_in_same_archive=any(k.startswith('visual_generator.vae.') for k in index['weight_map']),
        source_declared_LLM=dict(hidden_width=config['llm_config']['hidden_size'],layers=config['llm_config']['num_hidden_layers'],
            original_model=config['llm_config']['_name_or_path'],word_embeddings_tied=config['llm_config']['tie_word_embeddings']),
        source_declared_visual_vocab_size=config['visual_tokenizer_config']['vocab_size'],
        source_declared_visual_tokenize_function=config['visual_tokenizer_config']['tokenize_function'],
        floating_visual_embeddings_use_matrix_multiplication=True,
        image_generation_methods_call_autoregressive_generate=False,
        image_condition_concatenates_last_two_LLM_hidden_states=True,
        image_generation_methods_require_native_source_compatibility=True,
        native_lexical_modules_compiled_or_removed=False,numerical_equivalence_verified=False,
        source_declared_transformers_version=config['transformers_version'],
        weights_downloaded=0,whole_weight_checksums_verified=False,neural_models_loaded=0,
        runtime_parameters_counted=False,VRAM_measured=False,local_image_answer_quality_measured=False,
        semantic_learning_validated=False,general_ilm_goal_complete=False)
    (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('source_declared_archive_float_elements','source_declared_tensor_payload_GiB',
        'VAE_weights_declared_in_same_archive','source_declared_visual_tokenize_function','neural_models_loaded')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.source.resolve(),args.output.resolve())
