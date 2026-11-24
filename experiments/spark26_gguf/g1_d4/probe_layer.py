"""Run exactly the predeclared serial development requests."""
import json
from pathlib import Path
import probe

if __name__ == '__main__':
    for i in json.loads(Path('plan.json').read_text())['positions']:
        probe.emit('numeric', ok=True, **probe.numeric('layer0', i))

