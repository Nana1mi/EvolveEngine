"""Application-wire conformance on synthetic cases; not a timed E0/E0 workload."""
import json
from urllib.error import HTTPError
from chat_sse_observer import request

TOOL = {'type':'function','function':{'name':'record_color','description':'Return the requested color',
        'parameters':{'type':'object','properties':{'color':{'type':'string','enum':['blue']}},
                      'required':['color'],'additionalProperties':False}}}


def send(text, *, thinking=False, cap=32, **extra):
    return request(dict(model='spark26-q8',messages=[dict(role='user',content=text)],stream=True,
                        stream_options={'include_usage':True}, temperature=0, max_tokens=cap,
                        chat_template_kwargs={'enable_thinking':thinking}, **extra))


def smoke_cases():
    base = 'Reply with the single word READY.'
    yield 'text_stop', lambda: send(base), lambda r: (r.text.strip() == 'READY' and r.public['finish'] == 'stop'
                                                    and not r.reasoning and not r.tools)
    yield 'text_length', lambda: send(base,cap=1), lambda r: (r.public['completion_tokens'] == 1 and
                                                             r.public['finish'] == 'length')
    yield 'reasoning_and_answer', lambda: send('What is 3+4? Answer with the number only.',thinking=True,cap=128), lambda r: (
        r.text.strip() == '7' and bool(r.reasoning) and r.public['finish'] == 'stop' and
        r.public['first_payload_ms'] < r.public['first_answer_ms'])
    yield 'tool_only', lambda: send('Call record_color with color blue. Do not add prose.',cap=96,tools=[TOOL],
        tool_choice={'type':'function','function':{'name':'record_color'}}), lambda r: (
        r.tools == [dict(name='record_color',arguments={'color':'blue'})] and r.public['finish']=='tool_calls'
        and r.public['first_answer_ms'] is not None)


def main():
    total, passed = 0, 0
    for name, call, check in smoke_cases():
        total += 1
        try:
            seen = call()
        except HTTPError as e:
            print(json.dumps(dict(kind='wire_case',case=name,passed=False,http_status=e.code)),flush=True)
        except (ValueError,OSError) as e:
            # No request/response text or URLs go into the receipt.
            print(json.dumps(dict(kind='wire_case',case=name,passed=False,error_class=type(e).__name__)),flush=True)
        else:
            ok = bool(check(seen))
            print(json.dumps(dict(kind='wire_case',case=name,passed=ok,**seen.public)),flush=True)
            passed += int(ok)
    print(json.dumps(dict(kind='wire_summary',passed=passed,total=total)),flush=True)
    raise SystemExit(passed!=total)


if __name__ == '__main__':
    main()
