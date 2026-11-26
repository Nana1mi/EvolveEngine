"""One paid DeepSeek Search and one Fetch, independent of a Level 3 launch."""
import argparse
from getpass import getpass
from hashlib import sha256
import json
import os
from pathlib import Path
from urllib.request import urlopen

from evolveengine.deepseek import Provider
from evolveengine.v3_level3_web_research import WebResearch

QUERY = 'llama.cpp ggml cuda MMVQ kernel quantized matrix vector'
RAW = ('https://raw.githubusercontent.com/ggml-org/llama.cpp/'
       '07ba74babd586c6b5f558d196e3cf7cc1eea26f4/ggml/src/ggml-cuda/mmvq.cu')
CAP = 1 << 20


def smoke(out, provider):
    with urlopen(RAW, timeout=15) as source:
        reference = source.read(CAP+1)
    if not 0 < len(reference) <= CAP or b'mul_mat_vec' not in reference:
        raise ValueError('could not verify the frozen public source fixture')
    out.mkdir(parents=True, exist_ok=False)
    api = WebResearch(provider, out/'calls')
    result = {'fixture_sha256': sha256(reference).hexdigest(), 'fixture_bytes': len(reference),
              'endpoint': provider.base_url, 'search_nonempty': False, 'fetch_exact_bytes': None}
    try:
        found = api.search(QUERY, count=3)
        result['search_nonempty'] = bool(found['results'])
        if not found['results']:
            raise ValueError('search returned no results; stopping without retries')
        api.fetch(RAW)
        doc = json.loads((out/'calls'/'000002-response.json').read_bytes())
        content = doc['content'].encode('utf-8')
        result.update(fetch_bytes=len(content), fetch_exact_bytes=content == reference,
                      fetch_contains_full_source=reference.rstrip(b'\n') in content,
                      fetch_sha256=sha256(content).hexdigest())
    finally:
        result['research_ledger'] = api.snapshot()
        with (out/'summary.json').open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-paid-once', action='store_true', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--prompt-key', action='store_true', help='enter the credential privately without saving it')
    a = p.parse_args()
    key = 'DEEPSEEK_API_KEY'
    temporary = not os.environ.get(key) and a.prompt_key
    if temporary:
        os.environ[key] = getpass('DeepSeek API key: ')
    try:
        if not os.environ.get(key):
            p.error(f'{key} is missing; no request was made')
        summary = smoke(a.output, Provider('https://api.deepseek.com', 'deepseek-flash', key, 'DeepSeek-V4.1-Flash-1017', 393216))
    finally:
        if temporary:
            os.environ.pop(key, None)
    print(json.dumps(summary))
    return 0 if summary['fetch_contains_full_source'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
