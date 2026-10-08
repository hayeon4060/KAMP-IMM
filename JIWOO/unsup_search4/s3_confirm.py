"""
3단계: 후보 확인 (PREREG)
  후보 = 제품별 W1 공식 AP 상위 10 ∪ 두 기간 lift 기하평균 상위 10 (K1~K8 전체에서, 중복 제거)
  각 후보: W1(GMM 시드 5) · w2_all · eq_all · W2 재학습 + 같은 열로 lmaha300 (W1·W2)
  앙상블(K7)은 구성 조합 50개를 각 창에서 다시 학습해 백분위 평균
  비교: unsup_search3 s4 점수의 R0(제출 방식) · R1(w2_all·dv26·GMM2) · S3(unsup_search3 1위 AE), 같은 창
출력: outputs/s3_confirm.csv, outputs/scores/s3_{p}.npz
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import lib4 as L

WINS = ['W1', 'w2_all', 'eq_all', 'W2']
SEEDS = range(5)
N_BOOT = 500
RNG = np.random.default_rng(42)
S3 = L.ROOT.parent / 'unsup_search3' / 'outputs' / 'scores' / 's4_scores.pkl'


def fit_member(p, win, idx, m):
    seeds = SEEDS if (m == 'gmm2' and win == 'W1') else [0]
    s_, pc_ = [], []
    for sd in seeds:
        res, _ = L.fit_score(p, win, idx, [m], seed=sd)
        s, pct, err = res[m]
        if s is None:
            return None, None
        s_.append(s)
        pc_.append(pct)
    return np.mean(s_, 0), (None if pc_[0] is None else np.mean(pc_, 0))


def job(key, p, win, members, m):
    """members: 열 번호 리스트들. 1개면 점수 그대로, 여러 개면 백분위 평균"""
    out = [fit_member(p, win, idx, m) for idx in members]
    if len(members) == 1:
        return key, out[0][0]
    P = [o[1] for o in out if o[1] is not None]
    return key, (np.mean(P, 0).astype(np.float32) if P else None)


def ap_vec(y, s):
    o = np.argsort(-s, kind='stable')
    ys = y[o]
    return float((ys * np.cumsum(ys) / np.arange(1, len(y) + 1)).sum() / y.sum())


def boot(ev, per, a, b):
    e = ev[ev['period'] == per]
    y = e['y'].values.astype(float)
    sa, sb = L.L3._tb(a[e['pos'].values]), L.L3._tb(b[e['pos'].values])
    pi, ni = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = []
    for _ in range(N_BOOT):
        i = np.concatenate([RNG.choice(pi, len(pi)), RNG.choice(ni, len(ni))])
        d.append(ap_vec(y[i], sa[i]) - ap_vec(y[i], sb[i]))
    return np.percentile(d, [2.5, 97.5])


def refs(p, LBL):
    """unsup_search3 점수를 lib4 라벨 쇼트 순서로 맞춤"""
    S = pd.read_pickle(S3)
    LL3 = pd.read_pickle(L.ROOT.parent / 'unsup_search3' / 'outputs' / 'cache' / 'lib_labeled.pkl')[p]
    pos = pd.Series(np.arange(len(LL3)), index=pd.DatetimeIndex(LL3['TimeStamp']))
    idx = pos.loc[pd.DatetimeIndex(LBL[p]['meta']['TimeStamp'])].values
    W = {'W1': 'W1', 'W2': 'W2', 'w2_all': 'main'}
    out = {}
    for tag, base in (('R0', (p, 'cur_0901', 'c17', 'final_c17')), ('R1', (p, 'w2_all', 'dv26', 'gmm2')),
                      ('S3', (p, 'w2_all', 'dv26+rz5+mx5+drift0', 'ae'))):
        for w4, w3 in W.items():
            k = base + (w3,)
            if k in S and S[k] is not None:
                out[(tag, w4)] = np.asarray(S[k], float)[idx]
    return out


def main(n_jobs=15):
    U, LBL, cols = L.load()
    B = L.blocks(cols)
    combos = json.load(open(L.OUT / 's2_combos.json', encoding='utf-8'))
    s2 = pd.read_csv(L.OUT / 's2_results.csv')
    s2['id'] = s2['id'].astype(str)
    path8 = pd.read_csv(L.OUT / 's2b_path.csv') if (L.OUT / 's2b_path.csv').exists() else None
    tasks, CAND, EV, REF = [], {}, {}, {}
    for p in L.C.PRODUCTS:
        L.prepare_memmaps(U, LBL, p, WINS)
        EV[p] = L.eval_table(p, LBL)
        REF[p] = refs(p, LBL)
        cm = {str(c['id']): c for c in combos[p]}
        idx_of = lambda c: c.get('cols') or sorted(j for b in c['blocks'] for j in B[b])   # noqa: E731
        a = s2[(s2['product'] == p) & s2['off_AP'].notna()].copy()
        prev = {k: EV[p].loc[EV[p]['period'] == v, 'y'].mean() for k, v in (('off', 'official'), ('ext', 'extra'))}
        a['gm'] = np.sqrt((a['off_AP'] / prev['off']).clip(lower=1e-6) * (a['ext_AP'] / prev['ext']).clip(lower=1e-6))
        pick = pd.concat([a.nlargest(10, 'off_AP').assign(why='W1 공식 AP 상위 10'),
                          a.nlargest(10, 'gm').assign(why='lift 기하평균 상위 10')])
        cands = {}
        for r in pick.itertuples():
            if r.kind == 'K7':
                members = [idx_of(cm[str(i)]) for i in str(r.blocks).split('|')]
                desc = f'K7 앙상블 {r.id}'
            else:
                members = [idx_of(cm[r.id])]
                desc = f'{r.kind} #{r.id}: {r.blocks}' if r.blocks == r.blocks else f'{r.kind} #{r.id}'
            key = (desc, r.model)
            cands[key] = (members, cands.get(key, (None, ''))[1] + ('; ' if key in cands else '') + r.why)
        if path8 is not None:
            q = path8[path8['product'] == p]
            for m, g in q.groupby('model'):
                last = g.loc[g['step'].idxmax()]
                bl = last['set'].split('+')
                cands[(f'K8 전진 선택: {last["set"]}', m)] = ([sorted(j for b in bl for j in B[b])],
                                                         'K8 전진 선택 최종(과적합 대조군)')
        CAND[p] = cands
        for (desc, m), (members, why) in cands.items():
            for w in WINS:
                tasks.append(((p, desc, m, w), p, w, members, m))
            if len(members) == 1:
                for w in ('W1', 'W2'):
                    tasks.append(((p, desc, 'lmaha300', w), p, w, members, 'lmaha300'))
    L.log(f'3단계 후보 {sum(map(len, CAND.values()))}개, 작업 {len(tasks)}개')
    tasks.sort(key=lambda t: -sum(len(x) for x in t[3]))
    SC = {}
    for i, (k, s) in enumerate(Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
            delayed(job)(*t) for t in tasks), 1):
        SC[k] = s
        if i % 50 == 0:
            L.log(f'{i}/{len(tasks)}')
    rows = []
    for (p, desc, m, w), s in SC.items():
        if s is None:
            continue
        r = {'product': p, 'config': desc, 'model': m, 'window': w, **L.evaluate(EV[p], s)}
        why = CAND[p].get((desc, m), (None, '같은 열 lmaha300'))[1]
        r['why'] = why
        if w == 'W1':
            for tag in ('R0', 'R1', 'S3'):
                if (tag, 'W1') in REF[p]:
                    for per, pk in (('official', 'off'), ('extra', 'ext')):
                        lo, hi = boot(EV[p], per, np.asarray(s, float), REF[p][(tag, 'W1')])
                        r[f'd{pk}_{tag}_lo'], r[f'd{pk}_{tag}_hi'] = lo, hi
        rows.append(r)
    for p in L.C.PRODUCTS:
        for (tag, w), s in REF[p].items():
            rows.append({'product': p, 'config': f'{tag} (unsup_search3 점수)', 'model': tag, 'window': w,
                         'why': '비교', **L.evaluate(EV[p], s)})
    pd.DataFrame(rows).to_csv(L.OUT / 's3_confirm.csv', index=False, encoding='utf-8-sig')
    for p in L.C.PRODUCTS:
        ks = [k for k in SC if k[0] == p and SC[k] is not None]
        np.savez(L.OUT / 'scores' / f's3_{p}.npz', keys=np.array(['|'.join(k[1:]) for k in ks]),
                 s=np.vstack([SC[k] for k in ks]))
    L.log('완료')


if __name__ == '__main__':
    main()
