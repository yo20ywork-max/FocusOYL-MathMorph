"""Lexical diagnostics and fail-closed, evidence-scoped release policy."""
from __future__ import annotations
import math, re
from collections import Counter
from dataclasses import dataclass


def lexical(text):
    # Unicode words; CJK is tokenized by individual characters. Not model tokens.
    terms = re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[\u3400-\u9fff]", text.lower()[-6000:])
    windows = [tuple(terms[i:i+12]) for i in range(max(0, len(terms)-11))]
    duplicate = 1 - len(set(windows))/len(windows) if windows else 0.
    sentences = []
    for s in re.split(r'[.!?。！？\n]+', text.lower()[-6000:]):
        words = re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', s)
        if len(words) >= 6: sentences.append(' '.join(words))
    return {'tail_words': len(terms), 'duplicate12_rate': duplicate,
            'most_repeated_sentence': max(Counter(sentences).values(), default=0)}


@dataclass
class LoopMonitor:
    required_hits: int = 3
    stride_chars: int = 800
    threshold: float = .65
    min_words: int = 256
    hits: int = 0
    last_chars: int = 0
    def update(self, reason, final=''):
        if len(reason) - self.last_chars < self.stride_chars:
            return False
        self.last_chars = len(reason)
        d = lexical(reason)
        flag = not final.strip() and d['tail_words'] >= self.min_words and d['duplicate12_rate'] >= self.threshold
        self.hits = self.hits+1 if flag else 0
        return self.hits >= self.required_hits


def parse_final(text, kind):
    """Extract only final content; never receives gold, never scans hidden reasoning."""
    if not isinstance(text, str): return None
    token = r'([A-E])\b' if kind == 'mcq' else r'([+-]?\d+)(?![\w,/]|\.\d)'
    if kind not in ('mcq', 'integer'): raise ValueError('Unknown check type')
    hits = re.findall(r'(?im)\bFINAL\s*:\s*\**\s*'+token, text)
    if not hits:
        hits = re.findall(r'(?i)\b(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\**\(?'+token, text)
    if not hits:
        v = text.strip().strip('*').strip()
        pattern = r'[A-E]' if kind == 'mcq' else r'[+-]?\d+'
        if re.fullmatch(pattern, v, flags=re.I): hits = [v]
    if not hits: return None
    vals = [v.upper() if kind=='mcq' else str(int(v)) for v in hits]
    return vals[0] if len(set(vals)) == 1 else None


def summarize(rows):
    status = Counter(x['status'] for x in rows)
    measured = [x for x in rows if x.get('completion_tokens') is not None]
    lengths = sorted(x['completion_tokens'] for x in measured)
    n = len(rows)
    return {'n': n, 'correct': sum(x['correct'] for x in rows), 'status': dict(status),
            'reasoning_nonempty': sum(x['reasoning_chars'] > 0 for x in rows),
            'reported_completion_tokens': sum(lengths), 'usage_known_rows': len(measured),
            'missing_usage_rows': n-len(measured),
            'mean_reported_tokens': sum(lengths)/len(lengths) if lengths else None,
            'p95_reported_tokens': lengths[max(0, math.ceil(.95*len(lengths))-1)] if lengths else None,
            'max_tail_duplicate12_rate': max((x['lexical']['duplicate12_rate'] for x in rows), default=0),
            'categories': {c: {'n': sum(x['category']==c for x in rows),
                             'correct': sum(x['category']==c and x['correct'] for x in rows)}
                           for c in sorted({x['category'] for x in rows})}}


def assess(base, candidate, suite_purpose='regression_screen'):
    if not base or [x['id'] for x in base] != [x['id'] for x in candidate]:
        return {'screen': 'INVALID', 'release': 'RETAIN_ORIGINAL', 'reasons': ['Missing/unpaired rows']}
    reasons = []
    for label, rows in [('baseline', base), ('candidate', candidate)]:
        if any(x['status'] in ('USER_CANCELLED', 'TRANSPORT_ERROR') for x in rows):
            return {'screen': 'INVALID', 'release': 'RETAIN_ORIGINAL', 'reasons': [label+' has missing or interrupted measurements']}
    b, c = summarize(base), summarize(candidate)
    wins = sum(not x['correct'] and y['correct'] for x, y in zip(base, candidate))
    losses = sum(x['correct'] and not y['correct'] for x, y in zip(base, candidate))
    if losses: reasons.append('At least one baseline success regressed')
    for key in ('LOOP_GUARD_ABORT', 'TOKEN_LIMIT', 'WALL_TIME_CENSORED'):
        if c['status'].get(key, 0) > b['status'].get(key, 0): reasons.append(key+' increased')
    if any(c['categories'][k]['correct'] < v['correct'] for k, v in b['categories'].items()):
        reasons.append('Protected category regressed')
    token_ratio = None
    if b['missing_usage_rows']==0 and c['missing_usage_rows']==0:
        token_ratio = c['reported_completion_tokens']/max(1, b['reported_completion_tokens'])
        if token_ratio > 1.15: reasons.append('Generated-token cost grew more than 15%')
    else:
        reasons.append('Missing usage prevents a cost certificate')
    discord = wins+losses
    p = sum(math.comb(discord, i) for i in range(wins, discord+1))/2**discord if discord else 1.
    return {'screen': 'REJECT' if reasons else 'PASS_SCOPED_REGRESSION',
            'release': 'RETAIN_ORIGINAL', 'wins': wins, 'losses': losses,
            'one_sided_exact_p_descriptive': p, 'token_ratio': token_ratio,
            'reasons': reasons, 'baseline': b, 'candidate': c, 'purpose': suite_purpose,
            'general_gain_proven': False,
            'note': 'A regression screen or replay is never sufficient for automatic capability promotion.'}
