"""Bind the four actual final checkpoints to the deployment palette and count memory."""
import json
from scripts.prepare_book_prefix import MODEL,PUB
from scripts.prepare_dialogue_ink import dump,sha

ARMS = ('ink-hybrid','ink-neural','byte-hybrid','byte-neural')

def package():
    path = MODEL/'bundle.json'
    bundle = json.loads(path.read_text())
    checkpoints = {}
    schedules = set()
    for name in ARMS:
        receipt = json.loads((PUB/f'{name}.json').read_text())
        digest = sha(MODEL/f'{name}.safetensors')
        if digest != receipt['checkpoint']:
            raise ValueError(name)
        schedules.add(receipt['schedule_sha'])
        checkpoints[name] = digest
    if len(schedules) != 1:
        raise ValueError('Unmatched training windows')
    bundle['checkpoints'] = checkpoints
    dump(path,bundle)

if __name__ == '__main__':
    package()
