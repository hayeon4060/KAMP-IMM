"""
unified_model.py — 52번: 제품별 '통합 불량 예측 모델' (현장은 불량 유형을 모름 → 제품마다 검사 목록 1개)

구조: 안에서는 유형별 탐지기, 밖으로는 하나의 검사 순위
  CN7 통합 = 최종 모델 경보(pct≥0.999, 급변형·미성형 담당) 를 맨 위 + 나머지를 std4 흔들림 순     (라벨 0개)
  RG3 통합 = 가스 RF 순위와 N1(미성형) 순위를 68 : 32 비율로 번갈아 합친 하나의 순위            (labeled)
             (검사 K샷이면 앞에서부터 약 0.68K는 가스 RF 상위, 0.32K는 N1 상위 — 43·51번 예산 나누기를 순위 하나로)
  ※ 각 검사 샷에 '어느 탐지기가 올렸는지'(= 추정 원인)를 같이 출력 → 작업자 점검 방향 안내

비교 (같은 평가 조건: 2차 / 전체 평가 틀, 검사 5·10·15%, 시드 3개, 지표는 '불량 전체'가 기준)
  B0 무작위 검사 (20번 평균)
  B1 최종 모델 점수 단독 (비지도 Mahalanobis, 라벨 0개)
  B2 단일 RF — '불량 전체'를 한 번에 학습 (Run 하나씩 빼고 LORO / 시간 순 학습)
  B3 단일 RF + unlabeled 정상 보강 — '불량 전체' 타깃 N1 방식 (LORO)
  P  통합 모델 (위 구조) — RG3는 가스 RF를 LORO / 시간 순 두 가지로
  (운영) 100샷 묶음마다 10% 검사 루프: 통합 순위 고정 vs 불량 발견 시 앞뒤 5샷 확장

출력
  curves   : 모델 × 검사 비율 — 불량(전체) a/b, Precision·Recall·F1, + 보조로 가스·미성형 a/b   (가스+미성형은 양쪽에 셈)
  rank     : R-Precision · ROC-AUC · AP (검사 비율 무관)
  rows     : 2차 틀의 공식 행(LH/RH) 단위
  source   : 검사 10% 안에서 '탐지기(추정 원인) × 실제 유형' 표
  operation: 운영 루프
  perm     : 통합 모델 불량 포착 수의 우연 대비 p (CN7은 라벨 미사용이라 즉시, RG3는 재학습 포함 — 선택)
  unlabeled: 통합 목록을 unlabeled 전체 기간에 적용 → 날짜별 경보율 vs ERR_FACT_QTY 불량률 (labeled 날짜 제외)
"""
import os
import numpy as np
import pandas as pd

import semi_final as SF
import novel_common as NC
import inspect_curve as IC
import active_learning as AL
import semi_normal as SN
import semi_sup as SS

OUT = './outputs/unified_model'
RATES = (0.05, 0.10, 0.15)
SHARE = 0.68
P_NAME = {'cn7': 'P 통합: 경보 먼저 + std4', 'rg3_loro': 'P 통합: 가스 RF(LORO) + N1', 'rg3_fwd': 'P 통합: 가스 RF(시간 순) + N1'}


# ====================================================================== 순위 합치기
def _fill(v):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    return np.where(ok, v, (v[ok].min() - 1) if ok.any() else 0.0)


def interleave(lab, gas, short, share=SHARE):
    """두 순위를 share : (1−share) 비율로 번갈아 합친 하나의 점수 + 출처('가스 RF'/'N1'). 평가 대상 밖은 −inf"""
    ev = np.flatnonzero(lab['eval'].values)
    jit = np.random.default_rng(0).random(len(ev)) * 1e-9
    g, s = _fill(np.asarray(gas)[ev]) + jit, _fill(np.asarray(short)[ev]) + jit
    rg, rs = np.empty(len(ev)), np.empty(len(ev))
    rg[np.argsort(-g)] = np.arange(len(ev))
    rs[np.argsort(-s)] = np.arange(len(ev))
    kg, ks = (rg + 1) / share, (rs + 1) / (1 - share)
    score = np.full(len(lab), -np.inf)
    score[ev] = -np.minimum(kg, ks)
    src = np.full(len(lab), '', dtype=object)
    src[ev] = np.where(kg <= ks, '가스 RF', 'N1')
    return score, src


def forward_rf(D, target, seed=0):
    """시간 순 학습: 각 Run은 먼저 시작한 Run들의 라벨로만 학습한 RF (target: '가스' / '불량전체')"""
    lab, Zl = D['lab'], D['Zl']
    y, use = SS.target_y(lab, target)
    runs = lab['run_id'].values
    order = lab.groupby('run_id')['TimeStamp'].min().sort_values().index
    s = np.full(len(lab), np.nan)
    for k, r in enumerate(order):
        tr = np.isin(runs, list(order[:k])) & use
        if y[tr].sum() >= 2:
            te = runs == r
            s[te] = SN.fold_predict('M0_supervised', Zl[tr], y[tr], Zl[te], D['Zr'], target, seed)
    return s


# ====================================================================== 제품별 점수
def scores_cn7(frame, seeds):
    D0 = SN.load_std('cn7')
    D = dict(D0, lab=SF.apply_frame(D0['lab'], 'cn7', frame))
    lab = D['lab']
    lab_c, sc = IC.cn7_alarm_plus_gas_score()
    uni = SF._to(lab, lab_c, sc)
    src = np.where(SF._to(lab, lab_c, lab_c['pct'].values) >= 0.999, '최종 모델 경보', 'std4 흔들림')
    S = {sd: {'B1 최종 모델 점수 (비지도)': lab['score'].values,
              'B2 단일 RF 불량전체 (LORO)': SN.loro(D, 'M0_supervised', '불량전체', sd),
              'B2 단일 RF 불량전체 (시간 순)': forward_rf(D, '불량전체', sd),
              'B3 단일 RF 불량전체 + 정상 보강 (LORO)': SN.loro(D, 'N1_normal_augment', '불량전체', sd),
              P_NAME['cn7']: uni} for sd in seeds}
    return D, S, {P_NAME['cn7']: src}


def scores_rg3(frame, seeds):
    D0 = SN.load_std('rg3')
    D = dict(D0, lab=SF.apply_frame(D0['lab'], 'rg3', frame))
    Dg = SF.gas_target(D)
    lab = D['lab']
    S, SRC = {}, {}
    for sd in seeds:
        gas_l = SN.loro(Dg, 'M0_supervised', '가스', sd)
        gas_f = forward_rf(Dg, '가스', sd)
        n1 = SN.loro(D, 'N1_normal_augment', '미성형', sd)
        u_l, s_l = interleave(lab, gas_l, n1)
        u_f, s_f = interleave(lab, gas_f, n1)
        S[sd] = {'B1 최종 모델 점수 (비지도)': lab['score'].values,
                 'B2 단일 RF 불량전체 (LORO)': SN.loro(D, 'M0_supervised', '불량전체', sd),
                 'B2 단일 RF 불량전체 (시간 순)': forward_rf(D, '불량전체', sd),
                 'B3 단일 RF 불량전체 + 정상 보강 (LORO)': SN.loro(D, 'N1_normal_augment', '불량전체', sd),
                 '(참고) 가스 RF 단독 (LORO)': gas_l,
                 P_NAME['rg3_loro']: u_l, P_NAME['rg3_fwd']: u_f}
        if sd == seeds[0]:
            SRC = {P_NAME['rg3_loro']: s_l, P_NAME['rg3_fwd']: s_f}
    return D, S, SRC


# ====================================================================== 평가
def random_curve(lab, rates, n=20):
    T = pd.concat([IC.curve(lab, np.random.default_rng(1000 + b).random(len(lab)), rates, name='B0 무작위 검사')
                   for b in range(n)], ignore_index=True)
    return T


def source_table(lab, score, src, rate=0.10):
    """검사 rate 안에서 탐지기(추정 원인) × 실제 유형"""
    ev = np.flatnonzero(lab['eval'].values)
    K = int(np.ceil(rate * len(ev)))
    sel = ev[np.argsort(-_fill(np.asarray(score)[ev]))[:K]]
    reason = lab['Reason'].values[sel]
    actual = np.where(reason == '가스+미성형', '가스+미성형',
                      np.where(lab['type'].values[sel] == '정상', '정상', lab['type'].values[sel]))
    t = pd.crosstab(pd.Series(np.asarray(src)[sel], name='탐지기(추정 원인)'), pd.Series(actual, name='실제'))
    t['검사 샷'] = t.sum(axis=1)
    return t.reset_index()


def operation(p, frame, uni_score, lab_ref):
    Dal = AL.prepare(p)
    Dal['lab'] = SF.apply_frame(Dal['lab'], p, frame)
    Dal['base']['unified'] = np.nan_to_num(SF._to(Dal['lab'], lab_ref, np.where(np.isfinite(uni_score), uni_score, -1e9)),
                                           nan=-1e9)
    rows = []
    for st in ('S1 고정 순위', 'S2 고정 + 주변 확장'):
        rows.append(AL.summarize_log(Dal, AL.simulate(Dal, st, 'unified', seed=0), st, '통합 모델', 0))
    return rows


def run(frames=('2차', '전체'), products=('cn7', 'rg3'), seeds=(0, 1, 2), rates=RATES, n_perm=2000, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    curves, ranks, rowc, srcs, ops, perms = [], [], [], [], [], []
    for fr in frames:
        for p in products:
            D, S, SRC = (scores_cn7 if p == 'cn7' else scores_rg3)(fr, seeds)
            lab = D['lab']
            curves.append(random_curve(lab, rates).assign(제품=p, 평가틀=fr, seed=0))
            rw = SF.row_frame(p, lab) if fr == '2차' else None
            if rw is not None:
                rowc.append(pd.concat([IC.curve(rw, np.random.default_rng(1000 + b).random(len(rw)), rates, name='B0 무작위 검사')
                                       for b in range(20)]).assign(제품=p, 평가틀=fr, seed=0))
            for sd, sc in S.items():
                for nm, s in sc.items():
                    if nm.startswith('B1') and sd != seeds[0]:
                        continue
                    curves.append(IC.curve(lab, s, rates, name=nm).assign(제품=p, 평가틀=fr, seed=sd))
                    ranks.append({'평가틀': fr, '제품': p, '모델': nm, 'seed': sd, **IC.rank_metrics(lab, _fill(s))})
                    if rw is not None:
                        rowc.append(IC.curve(rw, SF.row_score(rw, lab, s), rates, name=nm).assign(제품=p, 평가틀=fr, seed=sd))
            s0 = S[seeds[0]]
            for nm, src in SRC.items():
                srcs.append(source_table(lab, s0[nm], src).assign(제품=p, 평가틀=fr, 모델=nm))
            main = P_NAME['cn7'] if p == 'cn7' else P_NAME['rg3_fwd']
            ops += [dict(r, 평가틀=fr, 모델=main) for r in operation(p, fr, s0[main], lab)]
            if p == 'cn7':                                             # 라벨 미사용 → 선택이 고정, 라벨만 원형 이동
                ev = np.flatnonzero(lab['eval'].values)
                K = int(np.ceil(0.10 * len(ev)))
                flag = np.zeros(len(lab), bool)
                flag[ev[np.argsort(-_fill(s0[main][ev]))[:K]]] = True
                perms.append({'평가틀': fr, '제품': p, '모델': main, '검사 비율': '10%', **NC.evaluate(lab, flag),
                              **NC.circ_null(lab, flag, n_perm)})
            if verbose:
                print(f'[{fr}] {p} 완료', flush=True)
    C_ = pd.concat(curves, ignore_index=True)
    R = {'curves': summarize(C_), 'curves_by_seed': C_,
         'rank': pd.DataFrame(ranks).groupby(['평가틀', '제품', '모델'], sort=False).mean(numeric_only=True).reset_index(),
         'source': pd.concat(srcs, ignore_index=True), 'operation': pd.DataFrame(ops), 'perm_cn7': pd.DataFrame(perms)}
    if rowc:
        R['rows'] = summarize(pd.concat(rowc, ignore_index=True))
    R['headline'] = headline(R)
    for k in ('curves', 'rank', 'source', 'operation', 'perm_cn7', 'rows', 'headline'):
        if k in R:
            R[k].to_csv(f'{OUT}/{k}.csv', index=False, encoding='utf-8-sig')
    return R


def summarize(C_):
    T = SF._mean_counts(C_, ['평가틀', '제품', '모델', '검사 비율'])
    base = T[T['모델'] == 'B0 무작위 검사'][['평가틀', '제품', '검사 비율', '불량 수']].rename(columns={'불량 수': '무작위 불량'})
    T = T.merge(base, on=['평가틀', '제품', '검사 비율'], how='left')
    T['무작위 대비(배)'] = T['불량 수'] / T['무작위 불량']
    T['불량'] = T['불량 수'].round(1).astype(str) + '/' + T['불량 전체'].astype(int).astype(str)
    for ty in ('가스', '미성형'):
        if f'{ty} 수' in T.columns:
            T[ty] = T[f'{ty} 수'].round(1).astype(str) + '/' + T[f'{ty} 전체'].astype('Int64').astype(str)
    cols = ['평가틀', '제품', '모델', '검사 비율', '검사 샷', '불량', 'Precision', 'Recall', 'F1', '무작위 대비(배)', '가스', '미성형']
    return T[[c for c in cols if c in T.columns]]


def headline(R, rate='10%'):
    """2장 비교표: 제품 × 모델, 검사 10%의 불량(전체) 기준 — 2차 / 전체 나란히"""
    T = R['curves']
    t = T[T['검사 비율'] == rate]
    w = None
    for fr, g in t.groupby('평가틀', sort=False):
        g = g[['제품', '모델', '불량', 'Precision', 'Recall', 'F1', '무작위 대비(배)', '가스', '미성형']].rename(
            columns=lambda c: c if c in ('제품', '모델') else f'{c} [{fr}]')
        w = g if w is None else w.merge(g, on=['제품', '모델'], how='outer')
    rk = R['rank'][R['rank']['평가틀'] == R['rank']['평가틀'].iloc[-1]][['제품', '모델', 'ROC-AUC', 'R-Precision']]
    return w.merge(rk.rename(columns={'ROC-AUC': 'ROC-AUC [전체]', 'R-Precision': 'R-Precision [전체]'}),
                   on=['제품', '모델'], how='left')


# ====================================================================== RG3 통합 모델 순열 (재학습 포함, 선택)
def perm_rg3(frame='2차', n=20, seed=0, rate=0.10, forward=True):
    """라벨을 Run 안에서 원형 이동 → 가스 RF·N1 재학습 → 통합 순위 → 검사 10%의 불량 포착 수 분포"""
    D0 = SN.load_std('rg3')
    D = dict(D0, lab=SF.apply_frame(D0['lab'], 'rg3', frame))

    def stat(Dx):
        Dg = SF.gas_target(Dx)
        g = forward_rf(Dg, '가스', seed) if forward else SN.loro(Dg, 'M0_supervised', '가스', seed)
        u, _ = interleave(Dx['lab'], g, SN.loro(Dx, 'N1_normal_augment', '미성형', seed))
        T = IC.curve(Dx['lab'], u, (rate,))
        return SF._num(T['불량'].iloc[0])

    obs = stat(D)
    lab0 = D['lab']
    ev = np.flatnonzero(lab0['eval'].values)
    runs = lab0['run_id'].values
    rng = np.random.default_rng(seed + 77)
    null = []
    for b in range(n):
        L = lab0.copy()
        cols = ['type', 'Reason', 'PassOrFail'] + (['gas_any'] if 'gas_any' in L.columns else [])
        for r in pd.unique(runs[ev]):
            ii = ev[runs[ev] == r]
            sh = rng.integers(0, len(ii))
            for c in cols:
                L.iloc[ii, L.columns.get_loc(c)] = np.roll(lab0[c].values[ii], sh)
        null.append(stat(dict(D, lab=L)))
        if b % 5 == 0:
            print('  순열', b, flush=True)
    null = np.asarray(null)
    return {'관측 불량(검사 10%)': obs, '무작위 평균': float(null.mean()), '무작위 95%': float(np.quantile(null, 0.95)),
            'p': float((1 + (null >= obs).sum()) / (n + 1))}


# ====================================================================== unlabeled: 통합 목록의 일반화
def unlabeled_unified(pkgs, products=('cn7', 'rg3'), min_shots=100, n_perm=5000, top=0.10):
    """unlabeled 전체 기간에 통합 목록 적용 → 비labeled 날짜 점수 상위 top을 경보로 → 날짜별 경보율 vs 불량률"""
    from daily_check import _perm_p
    rows, days = [], {}
    for p in products:
        f, err = SF.production_frame(p, pkgs[p])
        f['eval'] = ~f['labeled_day']
        if p == 'cn7':
            u = np.where(f['final_pct'] >= 0.999, 2.0 + f['final_pct'], f['std4_pct'])
        else:
            u, _ = interleave(f, f['gas_rf'].values, f['n1'].values)
        thr = np.quantile(u[f['eval'].values & np.isfinite(u)], 1 - top)
        f['통합 경보'] = (u >= thr).astype(float)
        d = f.groupby('plan_date').agg(n=('TimeStamp', 'size'), labeled_day=('labeled_day', 'max'), 경보율=('통합 경보', 'mean'))
        d['err'] = err['불량_합계']
        d['불량률'] = d['err'] / (2 * d['n'])
        days[p] = d
        use = d[(d['n'] >= min_shots) & d['err'].notna()]
        for nm, dd in (('labeled 날짜 제외', use[~use['labeled_day']]), ('전체 날짜', use)):
            ok = dd[['경보율', '불량률']].dropna()
            r, pv = _perm_p(ok['경보율'].values, ok['불량률'].values, n=n_perm) if len(ok) >= 5 else (np.nan, np.nan)
            rc, pc = _perm_p((dd['경보율'] * dd['n']).values, dd['err'].values, n=n_perm) if len(ok) >= 5 else (np.nan, np.nan)
            rows.append({'제품': p, '날짜 범위': nm, '날짜 수': len(ok), 'ρ(경보율, 불량률)': r, 'p': pv,
                         'ρ(경보 수, 불량 수)': rc, 'p(수)': pc})
        print(f'[{p}] unlabeled 통합 목록 완료', flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/unlabeled_unified.csv', index=False, encoding='utf-8-sig')
    return {'corr': T, 'days': days}
