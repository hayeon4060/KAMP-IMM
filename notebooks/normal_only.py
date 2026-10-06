"""
normal_only.py — '정상만 학습'의 학습 데이터를 바꿔 최종 모델 방식(free 17개 + Mahalanobis)을 다시 평가

최종 모델은 이미 '정상만 학습'임 — 다만 정상 = unlabeled 생산 샷(라벨 없음, 대부분 정상이라고 가정)
여기서는 '라벨로 확인된 정상 샷(labeled PassOrFail=0)'으로 학습하면 어떤지 비교

방법 (같은 변수·같은 모델·같은 전처리, 학습 데이터만 다름)
  A  ref        : 현재 최종 모델 — unlabeled 정상 기준 (outputs/final_model 점수 그대로)
  B  lab_normal : labeled 정상 샷만 — Run 하나씩 빼기(LORO)
                  테스트 Run은 정상·불량 모두 학습에서 제외 → 평가 샷을 학습에 쓰지 않음
  C  lab+ref    : labeled 정상(다른 Run) + unlabeled 정상 기준을 합쳐 학습 — LORO

주의
  - labeled 정상은 '라벨을 써서 고른' 학습 데이터 → B·C는 완전 비지도가 아님 (불량 샷을 빼는 데 라벨 사용)
  - Run마다 학습 모델이 달라 점수 척도가 다름 → 각 fold의 '학습 정상 대비 백분위(pct)'로 통일해서 이어 붙임
  - 경보 임계값 q도 각 fold의 학습 정상 점수 분위수 → '정상 오경보율 q'의 의미 유지
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from final_model import (FinalModel, FEATURES, INITIAL, TYPE_GROUPS, load_labeled_shots, load_official_rows,
                         make_features, build_reference, recall_at_k_by_run, ap_ci)
from compare_final import _paired_delta

QS = (0.99, 0.999)


def _pct_score(model, df):
    s = model.score(df)
    z = (s - np.median(model.s_ref)) / max(np.subtract(*np.percentile(model.s_ref, [75, 25])), 1e-9)
    return model.pct(s) + 1e-6 * np.tanh(z / 10), model.pct(s)        # (순위용: 동점 깨기, 경보용)


def prepare(p, base_dir='.', model_dir='./outputs/final_model'):
    base = FinalModel.load(f'{model_dir}/{p}_final_model.pkl')
    shots = load_labeled_shots(p, base_dir)
    lab = make_features(shots, base.steps, base.fr_step)
    ref_shots, _ = build_reference(p, shots, base_dir)
    ref = make_features(ref_shots, base.steps, base.fr_step)
    sc = pd.read_csv(f'{model_dir}/{p}_scores.csv', parse_dates=['TimeStamp'], usecols=['TimeStamp', 'score', 'pct'])
    lab = lab.merge(sc, on='TimeStamp', how='left')
    lab['in_eval'] = (lab['split'] != '제외') & (lab['Reason'] != INITIAL)
    return lab, ref, base


def run_methods(p, lab, ref, base):
    """반환: lab에 각 방법의 순위 점수(R_*)·경보용 백분위(P_*) 열 추가"""
    out = lab.copy()
    # A: 현재 최종 모델
    z = (out['score'] - np.median(base.s_ref)) / max(np.subtract(*np.percentile(base.s_ref, [75, 25])), 1e-9)
    out['R_A_ref'] = out['pct'] + 1e-6 * np.tanh(z / 10)
    out['P_A_ref'] = out['pct']
    for m in ['B_lab_normal', 'C_lab+ref']:
        out[f'R_{m}'] = np.nan
        out[f'P_{m}'] = np.nan
    log = []
    for r in out['run_id'].unique():
        te = out['run_id'] == r
        tr_norm = out[(~te) & (out['PassOrFail'] == 0) & out['in_eval']]          # 다른 Run의 확인된 정상
        for m, train in [('B_lab_normal', tr_norm), ('C_lab+ref', pd.concat([tr_norm, ref], ignore_index=True))]:
            mdl = FinalModel().fit(train)
            rs, ps = _pct_score(mdl, out[te])
            out.loc[te, f'R_{m}'] = rs
            out.loc[te, f'P_{m}'] = ps
            log.append({'제품': p, '빼둔 Run': r, '방법': m, '학습 샷': len(train)})
    return out, pd.DataFrame(log)


METHODS = ['A_ref', 'B_lab_normal', 'C_lab+ref']


def evaluate(p, out):
    e = out[out['in_eval']].reset_index(drop=True)
    y = e['PassOrFail'].values.astype(int)
    big = e.groupby('run_id')['PassOrFail'].sum().idxmax()
    inr = (e['run_id'] == big).values
    rows = []
    for m in METHODS:
        s = e[f'R_{m}'].values
        g = pd.DataFrame({'run_id': e['run_id'], 'score': s, 'y': y})
        ap, lo, hi = ap_ci(y, s)
        r = {'제품': p, '방법': m, 'AP': ap, 'AP_lo': lo, 'AP_hi': hi, 'AP/불량률': ap / y.mean()}
        for name, rs in TYPE_GROUPS.items():
            pm = e['Reason'].isin(rs).values
            if pm.sum():
                ev = pm | (y == 0)
                r[f'AP_{name}'] = average_precision_score(pm[ev].astype(int), s[ev])
        r['Run별 상위10%'] = '%d/%d' % recall_at_k_by_run(g, 0.10, g['y'] == 1)
        r['전체 상위10%'] = '%d/%d' % recall_at_k_by_run(g.assign(run_id='ALL'), 0.10, g['y'] == 1)
        if y[inr].sum() and (y[inr] == 0).any():
            r[f'{big} 내부 AP'] = average_precision_score(y[inr], s[inr])
        for q in QS:
            al = e[f'P_{m}'].values >= q
            tp = int((al & (y == 1)).sum())
            r[f'q{q} 경보'] = int(al.sum())
            r[f'q{q} 적중'] = tp
            r[f'q{q} 정상 중 경보 비율'] = (al & (y == 0)).sum() / max(1, (y == 0).sum())
        if m != 'A_ref':
            o, l, h, v = _paired_delta(y, s, e['R_A_ref'].values)
            r.update({'ΔAP(vs 현재)': o, 'Δ CI': f'[{l:.3f}, {h:.3f}]', '판정': v.replace('기준', '현재')})
        rows.append(r)
    return pd.DataFrame(rows)


def eval_frame1(p, out):
    rows = load_official_rows(p)
    y = rows['PassOrFail'].values.astype(int)
    r = {'제품': p}
    for m in METHODS:
        s = rows['TimeStamp'].map(pd.Series(out[f'R_{m}'].values, index=out['TimeStamp'].values)).fillna(0).values
        r[f'① AP {m}'] = average_precision_score(y, s)
    return r
