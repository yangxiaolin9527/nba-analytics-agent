"""Run inside the default replay container; no external provider calls."""
import importlib.util
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:8000'


def request(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


assert request('/readyz')['status'] == 'ok'
for path in ['/pto', '/pto/', '/demo/static/curry.png', '/demo/static/james.png']:
    try:
        request(path)
    except urllib.error.HTTPError as error:
        assert error.code == 404
    else:
        raise AssertionError(path)

counts = {}
for path, post in [('/demo/cases', '/demo/fixture'),
                   ('/demo/agent/cases', '/demo/agent/replay')]:
    cases = request(path)['cases']
    for case in cases:
        result = request(post, {'case_id': case['case_id']})
        assert result['response'] and result['case_id'] == case['case_id']
        if post == '/demo/agent/replay':
            assert result['response']['status'] == case['status']
            assert result['provenance']
        else:
            assert result['response']['tool'] == case['tool']
            assert result['response']['status'] == 'complete'
    counts[path] = len(cases)
assert counts == {'/demo/cases': 5, '/demo/agent/cases': 11}, counts
assert not Path('/app/.env.local').exists()
assert not Path('/app/docs/pto').exists()
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('transformers') is None
assert os.getuid() == 10001
print(json.dumps({'replays': counts, 'uid': os.getuid(),
                  'torch_installed': False, 'private_config_present': False,
                  'resources': request('/demo/status')}, ensure_ascii=False, indent=2))
