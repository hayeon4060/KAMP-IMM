"""[탐색, 선택 실행] 시계열 파운데이션 모델(Chronos) 예측 오차를 이상 점수로 — 제출 모델·기준선과 같은 쇼트에서 비교

실행 (패키지 루트, run_all.py를 먼저 한 번 실행해 outputs/가 있어야 함)
    pip install chronos-forecasting torch
    python src/explore/exp_chronos.py                       # 기본 amazon/chronos-2 (첫 실행 때 가중치를 인터넷에서 내려받음)
    CHRONOS_MODEL=amazon/chronos-bolt-small python src/explore/exp_chronos.py   # 가벼운 단변량 모델
    DEVICE=cuda python src/explore/exp_chronos.py           # GPU가 있으면
    MOCK=1 python src/explore/exp_chronos.py                # 점검용: 모델 대신 '직전 값' 예측(가중치 불필요)

방법 (결과를 보기 전에 고정)
    - 같은 Run 안 직전 최대 64쇼트(최소 10쇼트)의 원값 21개를 맥락으로 다음 쇼트 1개를 예측(다변량; Bolt는 변수별 단변량).
      맥락은 맥락 중앙값을 빼서 넣는다(위치 큰 변수의 스케일 문제 방지).
    - 변수별 오차 e = (실제 − 예측 중앙값) / max(예측 90% − 10% 폭, 기록 눈금). 점수 = rms(e), 보조 = max|e|.
    - 경보 기준 = 정상 추정 비라벨 기준 쇼트(제출 모델과 같은 build_reference) 중 무작위 3,000쇼트 점수의 99.9백분위.
    - 비교(원장 쇼트, 정답 = 어느 쪽이든 불량): AP, 기간 쇼트 상위 5% 안 불량, 운용점 경보 수·잡은 불량,
      그리고 제출 모델이 운용점에서 놓쳤는데 새로 잡은 불량 / 놓친 기존 불량을 쇼트 단위로.
결과: outputs/experiments/explore/chronos_scores.csv, chronos_compare.csv
지위: 탐색적. 제출 모델·원장 불변. 외부 사전학습 모델 → 보고서에 모델 이름·버전 명기.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_model as FM  # noqa: E402
from paths import INTER, LEDGER, EXP  # noqa: E402

MODEL = os.environ.get('CHRONOS_MODEL', 'amazon/chronos-2')
DEVICE = os.environ.get('DEVICE', 'cpu')
MOCK = os.environ.get('MOCK') == '1'
CTX, MINCTX, NREF, BATCH, Q = 64, 10, 3000, 128, [0.1, 0.5, 0.9]
OUT = EXP / 'explore'
OUT.mkdir(parents=True, exist_ok=True)


def load_pipeline():
    if MOCK:
        return None, 'mock'
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map=DEVICE, torch_dtype=torch.float32)
    kind = 'chronos2' if type(pipe).__name__ == 'Chronos2Pipeline' else 'univariate'
    print(f'모델 {MODEL} ({type(pipe).__name__}, {kind})', flush=True)
    return pipe, kind


def forecast(pipe, kind, ctxs):
    """ctxs: list of (n_var, L) 배열(중앙값 뺀 맥락) → (q10, q50, q90) 각 (n, n_var)"""
    n, V = len(ctxs), ctxs[0].shape[0]
    if kind == 'mock':                                   # 직전 값 + 맥락 표준편차 폭
        last = np.stack([c[:, -1] for c in ctxs]); sd = np.stack([c.std(1) for c in ctxs])
        return last - 1.2816 * sd, last, last + 1.2816 * sd
    import torch
    out = np.zeros((3, n, V))
    for b in range(0, n, BATCH):
        part = ctxs[b:b + BATCH]
        if kind == 'chronos2':
            qs, _ = pipe.predict_quantiles([torch.tensor(c, dtype=torch.float32) for c in part],
                                           prediction_length=1, quantile_levels=Q)
            arr = np.stack([q.detach().cpu().numpy()[:, 0, :] for q in qs])            # (b, V, 3)
        else:                                                                           # 변수별 단변량
            flat = [torch.tensor(c[v], dtype=torch.float32) for c in part for v in range(V)]
            qs, _ = pipe.predict_quantiles(flat, prediction_length=1, quantile_levels=Q)
            arr = qs.detach().cpu().numpy()[:, 0, :].reshape(len(part), V, 3)
        out[:, b:b + len(part)] = np.moveaxis(arr, 2, 0)
        print(f'    {min(b + BATCH, n)}/{n}', end='\r', flush=True)
    return out[0], out[1], out[2]


def score_shots(pipe, kind, df, RAW, res, pick=None):
    """같은 Run 안 직전 맥락으로 각 쇼트를 예측 → rms, max 오차. pick: 점수 낼 쇼트 인덱스(없으면 전부)"""
    df = df.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    X = df[RAW].to_numpy(float); idx, ctxs, ys = [], [], []
    want = set(range(len(df))) if pick is None else set(pick)
    for _, ii in df.groupby('run_id', sort=False).indices.items():
        for j, i in enumerate(ii):
            if j < MINCTX or i not in want:
                continue
            c = X[ii[max(0, j - CTX):j]]
            if np.isnan(c).any() or np.isnan(X[i]).any():
                continue
            med = np.median(c, 0); idx.append(i); ctxs.append((c - med).T); ys.append(X[i] - med)
    q10, q50, q90 = forecast(pipe, kind, ctxs)
    e = (np.array(ys) - q50) / np.maximum(q90 - q10, res)
    e = np.clip(e, -50, 50)
    out = df.loc[idx, ['TimeStamp']].copy()
    out['rms'] = np.sqrt((e ** 2).mean(1)); out['max'] = np.abs(e).max(1)
    return out


def main():
    pipe, kind = load_pipeline()
    L = pd.read_csv(LEDGER / 'ledger_v1_plus.csv'); L['TimeStamp'] = pd.to_datetime(L.TimeStamp)
    S = L.groupby(['product', 'period', 'TimeStamp']).agg(
        y=('y', 'max'), pct=('pct', 'first'), score=('score', 'first'), z=('z', 'first'), rf=('rf', 'first'),
        reason=('Reason', lambda x: '·'.join(sorted(set(x.dropna()))))).reset_index()
    known = set(S.TimeStamp[(S.y == 1) & (S.pct >= 0.999)])
    scores, rows = [], []
    for p in ('cn7', 'rg3'):
        sh = FM.load_labeled_shots(p, str(INTER)); ref, _ = FM.build_reference(p, sh, str(INTER))
        RAW = [c for c in ref.columns if c.endswith('_raw')]
        steps, _, _ = FM.load_scales(p, ref)
        res = np.array([steps.get(c[:-4], FM.resolution(ref[c])) or FM.resolution(ref[c]) for c in RAW], float)
        res = np.where(res > 0, res, 1e-3)
        print(f'[{p}] 라벨 쇼트 점수', flush=True)
        lab = score_shots(pipe, kind, sh, RAW, res)
        refs = ref.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
        pick = np.random.RandomState(0).choice(len(refs), min(NREF, len(refs)), replace=False)
        print(f'\n[{p}] 기준 쇼트 {len(pick)}개 점수(경보 기준)', flush=True)
        rs = score_shots(pipe, kind, refs, RAW, res, pick)
        thr = {k: float(np.quantile(rs[k], 0.999)) for k in ('rms', 'max')}
        lab['TimeStamp'] = pd.to_datetime(lab.TimeStamp); lab['product'] = p.upper(); scores.append(lab)
        for per in ('official', 'extra'):
            E = S[(S['product'] == p.upper()) & (S.period == per)].merge(lab, on=['TimeStamp'], how='left')
            k5 = int(np.ceil(0.05 * len(E)))
            meth = {'제출 모델': (E.score, E.pct >= 0.999), '기준선(강건 z)': (E.z, None), '지도 RF': (E.rf, None),
                    f'Chronos rms ({MODEL if not MOCK else "MOCK"})': (E.rms, E.rms >= thr['rms']),
                    f'Chronos max ({MODEL if not MOCK else "MOCK"})': (E['max'], E['max'] >= thr['max'])}
            for name, (s, al) in meth.items():
                s = s.fillna(s.min() if s.notna().any() else 0)
                top = s.rank(ascending=False, method='first') <= k5
                r = dict(제품=p.upper(), 기간=per, 방법=name, 쇼트=len(E), 불량=int(E.y.sum()),
                         AP=round(float(average_precision_score(E.y, s)), 3), 상위5pct_불량=int((top & (E.y == 1)).sum()))
                if al is not None:
                    caught = set(E.TimeStamp[al & (E.y == 1)])
                    r.update(경보=int(al.sum()), 경보중불량=len(caught),
                             새로잡음=', '.join(f'{t:%m/%d %H:%M}' for t in sorted(caught - known)) or '-',
                             놓친기존=', '.join(f'{t:%m/%d %H:%M}' for t in sorted((known & set(E.TimeStamp)) - caught)) or '-')
                rows.append(r)
    pd.concat(scores).to_csv(OUT / 'chronos_scores.csv', index=False, encoding='utf-8-sig')
    R = pd.DataFrame(rows); R.to_csv(OUT / 'chronos_compare.csv', index=False, encoding='utf-8-sig')
    pd.set_option('display.width', 250); print('\n' + R.to_string(index=False))
    print(f'→ {OUT / "chronos_compare.csv"}')


if __name__ == '__main__':
    main()
