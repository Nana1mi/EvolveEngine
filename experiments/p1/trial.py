"""Isolated P1 trial prototype. No scoring, GPU, batch dispatcher, or production imports of this module."""
from __future__ import annotations

from hashlib import sha256
import math
import subprocess
import time

from evolveengine.deepseek import ProviderBudget
from evolveengine.schema import Cost, canonical, in_scope, safe_path

LIMIT = Cost(wall_ms=180000, input_tokens=13500, output_tokens=4000)
SHARED = ('Work on this single card only. Report JSON: status (implemented, infeasible, or failed), '
          'intent and reason. Do not claim measured speedups. Read small source ranges. '
          'The controller may run a public syntax/diff check; do not try to access other tasks.\n')


class TrialBudget(ProviderBudget):
    """Keep BOTH directional caps. An uncertain response is charged to its original reservation."""
    def turn_output_ceiling(self, input_bound):
        if input_bound > self.limit.input_tokens - self.spent.input_tokens:
            return 0
        return min(super().turn_output_ceiling(input_bound), self.limit.output_tokens - self.spent.output_tokens)


def public_check(answer, delta):
    """No candidate code is run. Feedback has only allowlisted codes, paths and line numbers."""
    if (not isinstance(answer, dict) or set(answer) != {'status', 'intent', 'reason'}
            or answer['status'] not in ('implemented', 'infeasible', 'failed')
            or not all(isinstance(answer[k], str) and len(answer[k]) <= 4096 for k in ('intent', 'reason'))):
        return {'code': 'invalid_receipt'}
    if answer['status'] != 'implemented':
        return {'code': 'unexpected_edit' if delta else 'declined'}
    if not delta:
        return {'code': 'empty_patch'}
    for name, text in sorted(delta.items()):
        if len(text.encode()) > 1048576:
            return {'code': 'oversize', 'path': name}
        if name.endswith('.py'):
            try:
                compile(text, name, 'exec', dont_inherit=True)
            except (SyntaxError, ValueError, OverflowError) as err:
                return {'code': 'syntax_error', 'path': name, 'line': getattr(err, 'lineno', None)}
    return {'code': 'ok'}  # Not a semantic, fidelity, or performance verdict.


def run_trial(pi, repo, parent, *, card_bytes, card_sha256, arm, scope):
    """Controller supplies ONE public card and allowlisted sources, NEVER a judge bundle.

    Single fresh checkout per trial; repair uses the same working files, the same budget
    and one extra guarded Pi call (not an extra independent observation). No retry on
    provider/guard/timeout/invalid terminal JSON. Protocol never reads a judge verdict.
    """
    if arm not in ('single', 'short') or not isinstance(card_bytes, bytes) or not 0 < len(card_bytes) <= 16384:
        raise ValueError('Invalid trial/card')
    if sha256(card_bytes).hexdigest() != card_sha256 or not scope or not all(safe_path(n) and in_scope(n, repo.editable) for n in scope):
        raise ValueError('Card or source scope not locked')
    card = card_bytes.decode('utf-8'); first_prompt = SHARED + card
    started = time.monotonic(); calls = []; edits = {}; final = None; state = 'failed'; error = None
    broker = TrialBudget(pi.provider, LIMIT)
    try:
        with broker, repo.checkout(parent) as tree:
            before = repo.editable_snapshot(tree, repo.editable)
            if any(not any(in_scope(name, [path]) for name in before) for path in scope):
                raise ValueError('Unknown source scope')
            index = sorted(n for n in before if in_scope(n, scope))
            first_prompt += "\nPublic source navigation (both arms): " + canonical(index)
            prompt = first_prompt
            for i in range(2 if arm == 'short' else 1):
                if time.monotonic() >= broker.deadline:
                    raise subprocess.TimeoutExpired('P1 trial', 180)
                n = len(broker.rows)
                final, trace = pi.session(tree, prompt, broker, editing=scope, read_scope=repo.editable)
                if len(broker.rows) == n:
                    raise ValueError('Unmetered Pi call')
                after = repo.editable_snapshot(tree, repo.editable)
                if before.keys() != after.keys():
                    raise ValueError('Source inventory changed')
                edits = {name:text for name,text in after.items() if before[name] != text}
                if any(not in_scope(name, scope) for name in edits):
                    raise ValueError('Source outside trial scope changed')
                feedback = public_check(final, edits)
                calls.append({'answer': final, 'public_feedback': feedback, 'trace': trace,
                              'provider_row_range': [n,len(broker.rows)]})
                if feedback['code'] in ('ok','declined') or i==1 or arm=='single':
                    state = 'submitted'; break
                # Carry only the same card, previous public claim and static checker receipt.
                prompt = first_prompt + '\nOne correction remains in this same checkout.\n' + canonical({
                    'previous_receipt': final if feedback['code']!='invalid_receipt' else None,
                    'public_feedback':feedback})
            if time.monotonic() > broker.deadline:
                raise subprocess.TimeoutExpired('P1 trial',180)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        # A guard/budget failure is not repaired. Do not submit partially produced edits.
        state, error, edits = 'failed', type(exc).__name__ + ': ' + str(exc)[:400], {}
    spent = broker.spent if broker.settled.is_set() else LIMIT
    return {'protocol':'p1-public-check-v1', 'arm':arm, 'card_sha256':card_sha256, 'state':state,
            'calls':calls, 'final_receipt':final, 'edits':edits, 'error':error, 'provider_rows':broker.rows,
            'cost':{'wall_ms':math.ceil((time.monotonic()-started)*1000),'input_tokens':spent.input_tokens,
                    'output_tokens':spent.output_tokens}, 'independent_verdict':None}
