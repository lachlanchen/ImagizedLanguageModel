"""Training-only cell transcription and ordered answer scoring; unqualified by default.

The reader receives one physical cell and a fixed prompt. Questions, references,
source IDs and adjacent cells are absent from its request. Nothing here belongs
in deployed ILM answer generation or establishes a qualified policy reward.
"""
import base64
import hashlib
from io import BytesIO
import json
import unicodedata
import urllib.request

import numpy as np
from PIL import Image, ImageOps

ORIGIN = 'http://127.0.0.1:11434'
MODEL = 'qwen3.5:9b-q8_0'
MODEL_DIGEST = 'a8ee95bc3592ebaa624014ca8b429d25e0ef5f45030dc35786da7ff7e9e1b60a'
PROMPT = (
    'Read only the single visible printed glyph in this image. There is no word '
    'or sentence context. Do not repair strokes, infer a missing mark, translate, '
    'or replace a glyph by a similar character. If the visible form is damaged, '
    'ambiguous, or not identifiable as one character, return text:null and '
    'uncertain:true. Otherwise return exactly one Unicode character and '
    'uncertain:false. Return only a JSON object with keys text and uncertain.'
)
SCHEMA = dict(type='object', properties=dict(
    text=dict(type=['string', 'null'], maxLength=1), uncertain=dict(type='boolean')),
    required=['text', 'uncertain'], additionalProperties=False)


def packed_cell(ink):
    ink = np.asarray(ink)
    if ink.shape != (32, 32) or ink.dtype != np.uint8 or not np.isin(ink, [0, 1]).all():
        raise ValueError('Require the original binary 32x32 physical cell')
    return np.packbits(ink.reshape(-1), bitorder='big').tobytes()


def cell_key(packed):
    if not isinstance(packed, bytes) or len(packed) != 128:
        raise ValueError('Require 1024 original physical bits')
    return hashlib.sha256(packed).hexdigest()


def cell_png(packed):
    cell_key(packed)
    ink = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder='big').reshape(32, 32)
    image = Image.fromarray(255 - 255 * ink).resize((256, 256), Image.Resampling.NEAREST)
    image = ImageOps.expand(image, border=16, fill=255).convert('RGB')
    output = BytesIO(); image.save(output, format='PNG')
    return output.getvalue()


def payload(png):
    return dict(model=MODEL, stream=False, think=False, keep_alive='60s', format=SCHEMA,
        options=dict(temperature=0, seed=20261116, num_ctx=2048, num_predict=96),
        messages=[dict(role='user', content=PROMPT, images=[base64.b64encode(png).decode('ascii')])])


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('The local teacher must not redirect')


def local_json(path, body=None, *, timeout=10):
    if path not in {'/api/tags', '/api/ps', '/api/version', '/api/show', '/api/chat'}:
        raise ValueError('Use the fixed local model API only')
    request = urllib.request.Request(ORIGIN + path,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(1048577)
    if len(raw) > 1048576:
        raise ValueError('Teacher response exceeds 1 MiB')
    return json.loads(raw)


def fingerprint():
    version = local_json('/api/version')['version']
    models = local_json('/api/tags')['models']
    row, = [m for m in models if m['name'] == MODEL]
    if row['digest'] != MODEL_DIGEST or version != '0.32.6':
        raise ValueError('Registered local teacher/runtime identity changed')
    show = local_json('/api/show', dict(model=MODEL))
    if 'vision' not in show.get('capabilities', []):
        raise ValueError('The selected teacher must support images')
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return dict(model=MODEL, manifest_digest=row['digest'], ollama_version=version,
        capabilities=show['capabilities'], template_sha256=digest(show.get('template')),
        parameters_sha256=digest(show.get('parameters')), model_info_sha256=digest(show.get('model_info')))


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate teacher JSON key')
        result[key] = value
    return result


def parse_response(response):
    rejected = dict(text=None, uncertain=True, accepted=False)
    if (not isinstance(response, dict) or response.get('model') != MODEL or response.get('done') is not True
            or response.get('done_reason') != 'stop'):
        return dict(rejected, reason='incomplete_or_foreign_response')
    try:
        value = json.loads(response['message']['content'], object_pairs_hook=_object)
        if set(value) != {'text', 'uncertain'} or type(value['uncertain']) is not bool:
            raise ValueError('Unexpected teacher fields')
        text = value['text']
        if value['uncertain'] or not isinstance(text, str) or len(text) != 1:
            return dict(rejected, reason='ambiguous_or_non_single_character')
        if not text.strip() or unicodedata.category(text).startswith('C'):
            return dict(rejected, reason='invisible_nonblank_transcription')
        return dict(text=text, uncertain=False, accepted=True, reason='teacher_single_cell_reading')
    except (KeyError, TypeError, ValueError):
        return dict(rejected, reason='invalid_structured_reading')


def normalized(text):
    # Exactly the existing complete-response scorer's whitespace convention.
    return ''.join(text.split())


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def score_sequence(keys, readings, reference_text, reference_blank, *, stopped):
    if type(stopped) is not bool:
        raise ValueError('Stopping must come from the saved trajectory')
    if len(reference_text) != len(reference_blank):
        raise ValueError('Keep the complete physical reference extent')
    if not stopped:
        return dict(reward=0.0, complete_content=False, reason='capped', transcript=None)
    values = [readings.get(key) for key in keys]
    if not values or any(v is None or not v['accepted'] for v in values):
        return dict(reward=0.0, complete_content=False, reason='abstained_cell', transcript=None)
    observed = [v['text'] for v in values]
    if not any(c.strip() for c in observed):
        return dict(reward=0.0, complete_content=False, reason='all_white', transcript=''.join(observed))
    target = [' ' if blank else char for char, blank in zip(reference_text, reference_blank)]
    transcript = ''.join(observed)
    exact = normalized(transcript) == normalized(reference_text)
    distance = edit_distance(observed, target)
    similarity = max(0.0, 1 - distance / max(len(observed), len(target), 1))
    return dict(reward=.75 * similarity + .25 * exact, complete_content=exact,
        reason='ordered_visible_content', transcript=transcript, edit_distance=distance)
