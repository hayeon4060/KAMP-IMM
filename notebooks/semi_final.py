"""
semi_final.py — 51번: 제품·불량별 '현재 최선' 방법 묶음 (semi-final) — 학습 · labeled 결과 · unlabeled 일반화 점검

[제품·불량별 최선 방법 — 지금까지 비교 결과 기준]
  CN7 미성형 : 최종 모델 경보 (M_maha_free, 정상 기준 pct ≥ 0.999)                     라벨 0개 학습
               + 물리 설명용 I1 재료부족 규칙 (48번, Holm p 0.048 — 같은 3샷을 잡음)
  CN7 가스   : 경보 먼저 + std4 흔들림 순위 (38번 규칙) + 불량 발견 시 앞뒤 5샷 확장(49번)  라벨 0개 학습
               ⚠ 가스 11개가 모두 R1 → 검증 불가, 사례 수준
  RG3 가스   : 가스 RF (직전 10샷 std 24개, 1000트리, 깊이 4)                            labeled 지도학습
               Run 하나씩 빼고(LORO) + 시간 순 학습(forward, 49b) 두 방식으로 평가
  RG3 미성형 : N1 = 미성형 RF + unlabeled 3,000샷을 정상으로 추가(가중치 0.3) (41번)
               + 가스 RF와 검사 예산 나누기 (가스 68% : 미성형 32%, 43번)
               ⚠ 50번(라벨 없는 EWMA·std4) 결과에 따라 교체 가능 → SHORT_RG3 로 선택

[A] labeled 결과 (run_labeled) — 평가 틀 2개를 나란히
      GAS_INCLUDES_MIXED=True: '가스+미성형'을 가스에도 셈 (CN7 가스 11 → 13, 미성형 4는 그대로, 불량 총수는 그대로)
      '2차' : 공식 2차 데이터 샷만 (RG3 지도 모델은 학습도 2차 샷만) + 공식 2차 '행'(LH/RH) 단위 표
      '전체': 1차 + 2차 labeled 전체 (초기허용·RG3 R6 제외는 공통)
    - 검사 비율 5·10·15%의 Precision·Recall·F1·불량/가스/미성형 몇 개 중 몇 개, R-Precision·ROC-AUC
    - 경보 규칙(CN7 미성형)은 경보 샷 기준 표
    - 운영 루프(100샷 묶음마다 10% 검사): CN7 고정 vs 주변 확장 / RG3 시간 순 가스 RF
[B] 배포용 학습 (fit_deploy) — labeled 전체로 최종 학습해 저장 (outputs/semi_final/deploy_{p}.pkl)
[C] unlabeled 일반화 점검 (unlabeled_check) — unlabeled에는 샷 라벨이 없으므로 할 수 있는 3가지
    1) 날짜별 약한 라벨: ERR_FACT_QTY(생산계획일 × 캐비티 불량 수)로 만든 '날짜별 불량률'과
       규칙의 '날짜별 경보율·평균 점수'의 Spearman 상관 + 순열 p  (labeled 날짜는 빼서 독립 검증)
    2) 경보율 안정성: 월별 경보율이 정상 기준에서 정한 값(예: 0.1%, 5%, 10%)과 비슷한가
       (정상 기준 기간 9~11월은 학습에 쓴 기간 → 그 밖의 달이 진짜 '처음 보는 데이터')
    3) 현장 부담: 날짜별 경보(검사) 샷 수
    ⚠ ERR_FACT_QTY는 모든 불량 유형 합계 · 기록된 샷 수가 실제 생산보다 적은 날이 많음 → '관련성' 수준 근거
"""
import os
import pickle
import numpy as np
import pandas as pd

import novel_common as NC
import cn7_alarm_plus_gas as C
import inspect_curve as IC
import active_learning as AL
import semi_normal as SN
import semi_sup as SS
from final_model import raw, FEATURES, FinalModel, make_features, load_labeled_shots, load_official_rows, REF_PERIOD

OUT = './outputs/semi_final'
MD = './outputs/final_model'
RATES = (0.05, 0.10, 0.15)
GAS_SHARE = 0.68
GAS_INCLUDES_MIXED = True        # True: '가스+미성형'을 가스에도 셈(가스·미성형 양쪽) + RG3 가스 RF 학습에서도 가스로 사용
SHORT_RG3 = 'N1'                 # 'N1'(기본) / 'BOCPD_std4'(46번, 라벨 없음) / 'S1_ewma_std4'(50번, 라벨 없음)


# ====================================================================== 공통
def _num(s, k=0):
    return int(str(s).split('/')[k])


def _mean_counts(T, by):
    """seed별 curve 표 → 평균 (불량·가스·미성형 개수 평균 + 지표 평균)"""
    T = T.copy()
    for c in ('불량', '가스', '미성형'):
        if c in T.columns:
            T[c + ' 수'] = T[c].map(_num)
            T[c + ' 전체'] = T[c].map(lambda s: _num(s, 1))
    agg = {c + ' 수': 'mean' for c in ('불량', '가스', '미성형') if c in T.columns}
    agg.update({c + ' 전체': 'first' for c in ('불량', '가스', '미성형') if c in T.columns})
    agg.update({k: 'mean' for k in ('Precision', 'Recall', 'F1', '정규화 Precision') if k in T.columns})
    agg['검사 샷'] = 'first'
    return T.groupby(by, sort=False).agg(agg).reset_index()


def i1_rule(ref, lab, model, rate=0.01):
    """48번 I1 재료부족 = (−충전 + 속도 − 사출압)/3, 정상 기준 경보율 1%"""
    f = lambda Z, c: Z[:, FEATURES.index(c)]
    ind = lambda Z: (-f(Z, 'dev_Filling_Time') + f(Z, 'dev_Max_Injection_Speed') - f(Z, 'dev_Max_Injection_Pressure')) / 3
    vr, vl = ind(model._prep(ref)), ind(model._prep(lab))
    return vl > NC.threshold(vr, rate), vr, vl


FRAMES = {'2차': '2차 데이터만 (공식 평가 대상 샷 — RG3 지도 모델은 학습도 2차 샷만)',
          '전체': '1차 + 2차 labeled 전체'}
_SEC = {}


def secondary_times(p, base_dir='.'):
    """공식 2차 행(in_secondary)의 샷 시각"""
    if p not in _SEC:
        _SEC[p] = pd.to_datetime(load_official_rows(p, base_dir)['TimeStamp']).unique()
    return _SEC[p]


def apply_frame(lab, p, frame):
    """frame='2차' → 평가(+지도학습) 대상을 2차 샷으로 제한. '전체' → 그대로 (초기허용·RG3 R6 제외는 공통)"""
    lab = lab.copy()
    if GAS_INCLUDES_MIXED and 'Reason' in lab.columns:
        lab['gas_any'] = lab['Reason'].isin(['가스', '가스+미성형']).values
    if frame == '2차':
        lab['eval'] = lab['eval'].values & pd.to_datetime(lab['TimeStamp']).isin(secondary_times(p)).values
    return lab


def gas_target(D):
    """가스 RF 학습용: GAS_INCLUDES_MIXED면 '가스+미성형' 샷의 type을 '가스'로 (평가 표의 유형 집계는 gas_any로 따로 함)"""
    if not GAS_INCLUDES_MIXED:
        return D
    lab = D['lab'].copy()
    lab.loc[lab['Reason'] == '가스+미성형', 'type'] = '가스'
    return dict(D, lab=lab)


def bocpd_std4(p='rg3'):
    """46번 BOCPD·std4 (라벨 미사용) — RG3 미성형 5/8(검사 21%)이 나왔던 신호. (lab, ref 통계량, lab 통계량)"""
    from change_point import bocpd
    ref, lab, _ = NC.load(p)
    zr, zl = NC.base_signals(ref, lab)['std4']
    return lab, bocpd(zr, NC.segments(ref)), bocpd(zl, NC.segments(lab))


def _to(labD, lab_src, s):
    m = pd.Series(np.asarray(s, float), index=pd.to_datetime(lab_src['TimeStamp']).values)
    return m[~m.index.duplicated()].reindex(pd.to_datetime(labD['TimeStamp']).values).values


def rg3_short_score(D, seed, cache):
    if SHORT_RG3 == 'N1':
        return SN.loro(D, 'N1_normal_augment', '미성형', seed)
    if SHORT_RG3 == 'BOCPD_std4':
        lab, _, sl = cache.setdefault('bocpd', bocpd_std4('rg3'))
        return _to(D['lab'], lab, sl)
    import flow_monitor as FM                                          # 'S1_ewma_std4' (50번)
    if 'ewma' not in cache:
        ref, lab, model = NC.load('rg3')
        cache['ewma'] = (lab, FM.FlowScorer().fit(ref, model).score(lab)['S1 EWMA·std4'])
    lab, s = cache['ewma']
    return _to(D['lab'], lab, s)


def short_name():
    return {'N1': 'N1', 'BOCPD_std4': 'BOCPD·std4', 'S1_ewma_std4': 'EWMA·std4'}[SHORT_RG3]


# ---------------------------------------------------------------- 공식 2차 '행'(LH/RH) 단위 평가
def row_frame(p, lab):
    """공식 2차 행 → IC.curve가 읽는 형태 (eval·PassOrFail·type), 초기허용·제외 Run 빼고. 반환 (행 표, 시각)"""
    from final_model import INITIAL, TYPE_GROUPS
    rows = load_official_rows(p)
    m = lab[['TimeStamp', 'run_id', 'split']].drop_duplicates('TimeStamp')
    rows = rows.merge(m, on='TimeStamp', how='left')
    rows = rows[(rows['split'] != '제외') & (rows['Reason'] != INITIAL)].reset_index(drop=True)
    rows['eval'] = True
    if GAS_INCLUDES_MIXED:
        rows['gas_any'] = rows['Reason'].isin(['가스', '가스+미성형']).values
    rows['type'] = np.select([rows['Reason'].isin(TYPE_GROUPS['가스']), rows['Reason'].isin(TYPE_GROUPS['미성형'])],
                             ['가스', '미성형'], np.where(rows['PassOrFail'] == 1, '기타', '정상'))
    return rows


def row_score(rows, lab_src, s):
    m = pd.Series(np.asarray(s, float), index=pd.to_datetime(lab_src['TimeStamp']).values)
    m = m[~m.index.duplicated()]
    return m.reindex(pd.to_datetime(rows['TimeStamp']).values).values


# ====================================================================== [A] labeled 결과
def _frame_run(frame, seeds, rates, verbose):
    import rg3_active as RA
    rows_rule, curves, ranks, ops, rowc = [], [], [], [], []
    add = lambda T, p, sd: curves.append(T.assign(제품=p, seed=sd))

    # ---------------- CN7 (라벨 미사용 → 평가 대상만 바뀜)
    ref, lab0, model = NC.load('cn7')
    lab = apply_frame(lab0, 'cn7', frame)
    alarm = lab['pct'].values >= 0.999
    i1, _, _ = i1_rule(ref, lab, model)
    rows_rule.append(NC.evaluate(lab, alarm, 'CN7 미성형 | 최종 모델 경보 pct≥0.999', 제품='cn7'))
    rows_rule.append(NC.evaluate(lab, i1, 'CN7 미성형 | I1 재료부족 규칙(경보율 1%, 물리 설명용)', 제품='cn7'))
    rows_rule.append(NC.evaluate(lab, alarm & i1, 'CN7 미성형 | 경보 AND I1 (둘 다)', 제품='cn7'))
    lab_c, sc = IC.cn7_alarm_plus_gas_score()
    lab_c = apply_frame(lab_c, 'cn7', frame)
    add(IC.curve(lab_c, sc, rates, name='CN7 | 경보 먼저 + std4 순위'), 'cn7', 0)
    ranks.append({'제품': 'cn7', '모델': 'CN7 | 경보 먼저 + std4 순위', 'seed': 0, **IC.rank_metrics(lab_c, sc)})
    if frame == '2차':
        rw = row_frame('cn7', lab_c)
        rowc.append(IC.curve(rw, row_score(rw, lab_c, sc), rates, name='CN7 | 경보 먼저 + std4 순위').assign(제품='cn7', seed=0))
        rowc.append(IC.curve(rw, row_score(rw, lab_c, lab_c['score'].values), rates, name='CN7 | 최종 모델 점수 (참고)').assign(제품='cn7', seed=0))
    Dal = AL.prepare('cn7')
    Dal['lab'] = apply_frame(Dal['lab'], 'cn7', frame)
    for st in ('S1 고정 순위', 'S2 고정 + 주변 확장'):
        ops.append(AL.summarize_log(Dal, AL.simulate(Dal, st, 'combo', seed=0), st, 'CN7 경보+std4', 0))
    if verbose:
        print(f'  [{frame}] cn7 완료', flush=True)

    # ---------------- RG3
    D0 = SN.load_std('rg3')
    D = dict(D0, lab=apply_frame(D0['lab'], 'rg3', frame))
    labr = D['lab']
    Dg = gas_target(D)                                                 # 가스 RF 학습용 (혼합을 가스 양성으로)
    cache = {}
    labb, br, bl = cache.setdefault('bocpd', bocpd_std4('rg3'))
    labb = apply_frame(labb, 'rg3', frame)
    rows_rule.append(NC.evaluate(labb, bl > NC.threshold(br, 0.10),
                                 'RG3 미성형 | BOCPD·std4 경보 (정상 기준 10%, 실제 경보율은 더 높음)', 제품='rg3'))
    sname = short_name()
    rw = row_frame('rg3', labr) if frame == '2차' else None
    for sd in seeds:
        gas = SN.loro(Dg, 'M0_supervised', '가스', sd)
        fwd, _ = RA.forward_gasrf(Dg, sd)
        short = rg3_short_score(D, sd, cache)
        for nm, s in (('RG3 가스 | 가스 RF (LORO)', gas), ('RG3 가스 | 가스 RF (시간 순 학습)', fwd),
                      (f'RG3 미성형 | {sname} 단독', short)):
            add(IC.curve(labr, s, rates, name=nm), 'rg3', sd)
            ranks.append({'제품': 'rg3', '모델': nm, 'seed': sd, **IC.rank_metrics(labr, s)})
            if rw is not None:
                rowc.append(IC.curve(rw, row_score(rw, labr, s), rates, name=nm).assign(제품='rg3', seed=sd))
        for nm, g in ((f'RG3 결합 | 가스 RF(LORO) + {sname} (g={GAS_SHARE})', gas),
                      (f'RG3 결합 | 가스 RF(시간 순) + {sname} (g={GAS_SHARE})', fwd)):
            add(IC.split_curve(labr, g, short, GAS_SHARE, rates, name=nm), 'rg3', sd)
            if rw is not None:
                rowc.append(IC.split_curve(rw, row_score(rw, labr, g), row_score(rw, labr, short), GAS_SHARE, rates,
                                           name=nm).assign(제품='rg3', seed=sd))
        if verbose:
            print(f'  [{frame}] rg3 seed {sd} 완료', flush=True)
    Dr = AL.prepare('rg3')
    Dr['lab'] = apply_frame(Dr['lab'], 'rg3', frame)
    fwd, _ = RA.forward_gasrf(Dg, seeds[0])
    fwd = RA._map(D['lab'], Dr['lab'], fwd)
    combo_rank = pd.Series(Dr['base']['combo']).rank(pct=True).values
    Dr['base']['gasrf_forward'] = np.where(np.isfinite(fwd), fwd, combo_rank)
    for st in ('S1 고정 순위', 'S2 고정 + 주변 확장'):
        ops.append(AL.summarize_log(Dr, AL.simulate(Dr, st, 'gasrf_forward', seed=0), st, 'RG3 가스 RF(시간 순)', 0))

    C_ = pd.concat(curves, ignore_index=True)
    out = {'rules': pd.DataFrame(rows_rule), 'curves': _mean_counts(C_, ['제품', '모델', '검사 비율']), 'curves_by_seed': C_,
           'rank': pd.DataFrame(ranks).groupby(['제품', '모델'], sort=False).mean(numeric_only=True).reset_index(),
           'operation': pd.DataFrame(ops)}
    if rowc:
        out['rows_curves'] = _mean_counts(pd.concat(rowc, ignore_index=True), ['제품', '모델', '검사 비율'])
    out['best'] = best_table(out)
    return out


def run_labeled(frames=('2차', '전체'), seeds=(0, 1, 2), rates=RATES, verbose=True):
    """frames: '2차'(2차 데이터만) · '전체'(1차+2차). 표마다 '평가 틀' 열로 구분"""
    os.makedirs(OUT, exist_ok=True)
    parts = {}
    for fr in frames:
        if verbose:
            print(f'[{fr}] {FRAMES[fr]}', flush=True)
        parts[fr] = _frame_run(fr, seeds, rates, verbose)
    R = {}
    for k in ('rules', 'curves', 'rank', 'operation', 'best', 'rows_curves', 'curves_by_seed'):
        tabs = [v[k].assign(**{'평가 틀': fr}) for fr, v in parts.items() if k in v]
        if tabs:
            T = pd.concat(tabs, ignore_index=True)
            R[k] = T[['평가 틀'] + [c for c in T.columns if c != '평가 틀']]
            if k != 'curves_by_seed':
                R[k].to_csv(f'{OUT}/labeled_{k}.csv', index=False, encoding='utf-8-sig')
    R['compare'] = compare_frames(R)
    R['compare'].to_csv(f'{OUT}/labeled_compare_frames.csv', index=False, encoding='utf-8-sig')
    return R


def compare_frames(R):
    """최선 방법 요약을 평가 틀(2차 / 전체) 나란히"""
    B = R['best']
    keep = ['제품', '불량', '방법', '기준', '잡은 수', 'Precision', 'Recall(해당 유형)']
    w = None
    for fr, g in B.groupby('평가 틀', sort=False):
        g = g[keep].rename(columns={c: f'{c} [{fr}]' for c in ('기준', '잡은 수', 'Precision', 'Recall(해당 유형)')})
        w = g if w is None else w.merge(g, on=['제품', '불량', '방법'], how='outer')
    return w


def _ratio(a, b):
    return a / b if b else np.nan


def best_table(R, rate='10%'):
    """제품·불량별 최선 방법 한 장 요약 (검사 10% 또는 경보 기준)"""
    rows = []
    c = R['curves']

    def pick(m):
        t = c[(c['모델'] == m) & (c['검사 비율'] == rate)]
        return t.iloc[0] if len(t) else None

    def cnt(g, ty, fmt='{:.1f}'):
        if g is None or f'{ty} 수' not in g.index or pd.isna(g.get(f'{ty} 전체')):
            return '-', np.nan
        return f'{fmt.format(g[f"{ty} 수"])}/{int(g[f"{ty} 전체"])}', _ratio(g[f'{ty} 수'], g[f'{ty} 전체'])

    r = R['rules'].iloc[0]
    sh = r.get('미성형', '0/0') if isinstance(r.get('미성형'), str) else '0/0'
    rows.append({'제품': 'CN7', '불량': '미성형', '방법': '최종 모델 경보 (pct≥0.999)', '기준': f'경보 {r["경보(검사) 샷"]}샷',
                 '잡은 수': sh, 'Precision': r['Precision'], 'Recall(해당 유형)': _ratio(_num(sh), _num(sh, 1)),
                 '학습 라벨': '0개', '신뢰성': 'R1 한 에피소드(3샷 연속) · 48번 I1 Holm p 0.048 · 47번 물리 일치'})
    g = pick('CN7 | 경보 먼저 + std4 순위')
    op = R['operation']
    o2 = op[(op['기본 점수'] == 'CN7 경보+std4') & (op['전략'] == 'S2 고정 + 주변 확장')].iloc[0]
    s_, rc = cnt(g, '가스', '{:.0f}')
    rows.append({'제품': 'CN7', '불량': '가스', '방법': '경보 + std4 순위 / 운영: + 주변 5샷 확장',
                 '기준': f'검사 {rate} ({int(g["검사 샷"])}샷)' if g is not None else '-',
                 '잡은 수': f'{s_} → 확장 시 {o2["가스"]}/{o2["가스 전체"]}', 'Precision': g['Precision'] if g is not None else np.nan,
                 'Recall(해당 유형)': rc, '학습 라벨': '0개', '신뢰성': '가스가 모두 R1 → 검증 불가(사례)'})
    for nm, lbl, rel in (('RG3 가스 | 가스 RF (LORO)', 'labeled (다른 Run)', '순열 p 0.007 · 시드 안정 · 날짜 상관 ρ 0.50'),
                         ('RG3 가스 | 가스 RF (시간 순 학습)', '이전 Run만', '현장 조건 (미래 라벨 없음, 49b)')):
        g = pick(nm)
        s_, rc = cnt(g, '가스')
        rows.append({'제품': 'RG3', '불량': '가스', '방법': nm.split('| ')[1],
                     '기준': f'검사 {rate} ({int(g["검사 샷"])}샷)' if g is not None else '-', '잡은 수': s_,
                     'Precision': g['Precision'] if g is not None else np.nan, 'Recall(해당 유형)': rc, '학습 라벨': lbl, '신뢰성': rel})
    sm = [m for m in c['모델'].unique() if m.startswith('RG3 미성형')][0]
    g = pick(sm)
    s_, rc = cnt(g, '미성형')
    rows.append({'제품': 'RG3', '불량': '미성형', '방법': sm.split('| ')[1],
                 '기준': f'검사 {rate} ({int(g["검사 샷"])}샷)' if g is not None else '-', '잡은 수': s_,
                 'Precision': g['Precision'] if g is not None else np.nan, 'Recall(해당 유형)': rc,
                 '학습 라벨': 'labeled + unlabeled(정상)' if SHORT_RG3 == 'N1' else '0개',
                 '신뢰성': '50번 보정p 0.072~0.32 → 우연 수준 못 벗어남'})
    rb = R['rules'][R['rules']['방법'].str.startswith('RG3 미성형 | BOCPD')]
    if len(rb):
        rb = rb.iloc[0]
        s2 = rb.get('미성형') if isinstance(rb.get('미성형'), str) else '0/0'
        rows.append({'제품': 'RG3', '불량': '미성형', '방법': 'BOCPD·std4 경보 (라벨 없음, 참고)', '기준': f'경보 {rb["경보(검사) 샷"]}샷',
                     '잡은 수': s2, 'Precision': rb['Precision'], 'Recall(해당 유형)': _ratio(_num(s2), _num(s2, 1)),
                     '학습 라벨': '0개', '신뢰성': '46번 전체 탐색 보정 p 0.013 · 단, 경보율이 목표의 2~3배(검사 ~21%)'})
    cm = [m for m in c['모델'].unique() if m.startswith('RG3 결합 | 가스 RF(LORO)')][0]
    g = pick(cm)
    sg, _ = cnt(g, '가스')
    ss, _ = cnt(g, '미성형')
    rows.append({'제품': 'RG3', '불량': '가스+미성형', '방법': cm.split('| ')[1],
                 '기준': f'검사 {rate} ({int(g["검사 샷"])}샷)' if g is not None else '-', '잡은 수': f'가스 {sg} · 미성형 {ss}',
                 'Precision': g['Precision'] if g is not None else np.nan, 'Recall(해당 유형)': g['Recall'] if g is not None else np.nan,
                 '학습 라벨': 'labeled', '신뢰성': '43번 순열 결과 참고'})
    return pd.DataFrame(rows)


# ====================================================================== [B] 배포용 학습
def rg3_feature_fn(D):
    """semi_sup.load와 같은 방식(std10 + robust z, 정상 기준 통계)으로 임의 샷 묶음의 RG3 입력 24개 계산"""
    ref = D['ref']
    cols = [n.replace('|std10', '') for n in D['names']]
    rawc = [c + '_raw' if (c + '_raw') in ref.columns else c for c in cols]
    F = lambda d: pd.DataFrame({n: SS._std10(d, c) for n, c in zip(D['names'], rawc)}).replace([np.inf, -np.inf], np.nan)
    Fr = F(ref)
    med = Fr.median()
    q75, q25 = Fr.quantile(0.75), Fr.quantile(0.25)
    iqr = (q75 - q25).where((q75 - q25) > 1e-9, Fr.std().where(Fr.std() > 1e-9, 1.0))
    return lambda d: np.clip(((F(d).fillna(med) - med) / iqr).values, -20, 20)


def fit_deploy(seed=0, n_unl=3000):
    """labeled 전체로 최종 학습 → 저장. CN7은 라벨을 안 쓰는 규칙이라 정상 기준 통계만 저장"""
    import synthetic_injection as SI
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for p in ('cn7', 'rg3'):
        ref, lab, model = NC.load(p)
        pkg = {'product': p, 'final_model_path': f'{MD}/{p}_final_model.pkl', 'std4': SI.Std4Scorer().fit(ref),
               'alarm_q': 0.999}
        if p == 'rg3':
            D = SN.load_std('rg3')
            yg, ug = SS.target_y(gas_target(dict(D, lab=apply_frame(D['lab'], 'rg3', '전체')))['lab'], '가스')
            pkg['gas_rf'] = SN._rf('가스', seed).fit(D['Zl'][ug], yg[ug])
            ys, us = SS.target_y(D['lab'], '미성형')
            ui = np.random.default_rng(seed).choice(len(D['Zr']), min(n_unl, len(D['Zr'])), replace=False)
            X = np.vstack([D['Zl'][us], D['Zr'][ui]])
            y = np.r_[ys[us], np.zeros(len(ui), int)]
            w = np.r_[np.ones(us.sum()), 0.3 * np.ones(len(ui))]
            pkg['short_n1'] = SN._rf('미성형', seed).fit(X, y, sample_weight=w)
            pkg['n1_train_times'] = list(pd.to_datetime(D['ref']['TimeStamp'].values[ui]))
            pkg['feature_names'] = D['names']
            pkg['labeled_gas_scores'] = pkg['gas_rf'].predict_proba(D['Zl'])[:, 1]
        with open(f'{OUT}/deploy_{p}.pkl', 'wb') as f:
            pickle.dump({k: v for k, v in pkg.items() if k != 'std4'}, f)     # std4 계산기는 정상 기준에서 바로 다시 만듦
        out[p] = pkg
        print(f'[{p}] 배포용 학습 저장 → {OUT}/deploy_{p}.pkl', flush=True)
    return out


# ====================================================================== [C] unlabeled 일반화 점검
def production_frame(p, pkg, base_dir='.'):
    """unlabeled 전체 기간 생산 샷 → 최종 모델·std4 (+ RG3 가스 RF·N1) 점수"""
    from daily_check import load_daily_errors, production_shots
    model = FinalModel.load(pkg['final_model_path'])
    u, err = load_daily_errors(p, base_dir)
    s, _ = production_shots(u)
    new = [c for c in SS.NEW if c in u.columns]
    if new:
        s = s.merge(u.drop_duplicates('TimeStamp')[['TimeStamp'] + new], on='TimeStamp', how='left')
    f = make_features(s, model.steps, model.fr_step)
    f['final_pct'] = model.pct(model.score(f))
    f['std4_pct'] = pkg['std4'].pct(f)
    if p == 'rg3':
        D = SN.load_std('rg3', base_dir)
        Z = rg3_feature_fn(D)(f)
        f['gas_rf'] = pkg['gas_rf'].predict_proba(Z)[:, 1]
        f['n1'] = pkg['short_n1'].predict_proba(Z)[:, 1]
        f.loc[f['TimeStamp'].isin(pkg['n1_train_times']), 'n1'] = np.nan       # N1 학습에 쓴 샷은 제외
    lts = pd.to_datetime(load_labeled_shots(p, base_dir)['TimeStamp'])
    lab_days = list(lts.dt.normalize().unique())
    f['labeled_shot'] = f['TimeStamp'].isin(list(lts))
    f['labeled_day'] = pd.to_datetime(f['plan_date']).isin(lab_days) | f.groupby('plan_date')['labeled_shot'].transform('max')
    f['ref_period'] = (f['TimeStamp'] >= REF_PERIOD[0]) & (f['TimeStamp'] <= pd.Timestamp(REF_PERIOD[1]) + pd.Timedelta(days=1))
    return f, err


def rules_on(f, p):
    """shot 단위 경보(규칙) — 임계값은 모두 라벨 없이 정한 값 또는 '비labeled 날짜 점수의 상위 10%'"""
    R = {'최종 모델 경보 (pct≥0.999)': f['final_pct'] >= 0.999,
         'std4 흔들림 (pct≥0.95)': f['std4_pct'] >= 0.95,
         '경보 or std4 (CN7 가스 규칙)': (f['final_pct'] >= 0.999) | (f['std4_pct'] >= 0.95)}
    if p == 'rg3':
        base = ~f['labeled_day']
        for c, nm in (('gas_rf', '가스 RF 상위 10%'), ('n1', 'N1 상위 10%')):
            thr = f.loc[base, c].quantile(0.90)
            R[nm] = f[c] >= thr
    return R


def unlabeled_check(pkgs=None, products=('cn7', 'rg3'), min_shots=100, n_perm=5000, base_dir='.'):
    from daily_check import _perm_p
    os.makedirs(OUT, exist_ok=True)
    if pkgs is None:
        pkgs = {p: pickle.load(open(f'{OUT}/deploy_{p}.pkl', 'rb')) for p in products}
    import synthetic_injection as SI
    for p in products:
        if 'std4' not in pkgs[p]:
            pkgs[p]['std4'] = SI.Std4Scorer().fit(NC.load(p)[0])
    CORR, MON, DAYS = [], [], {}
    for p in products:
        f, err = production_frame(p, pkgs[p], base_dir)
        R = rules_on(f, p)
        for nm, fl in R.items():
            f['R|' + nm] = fl.astype(float)
        score_cols = {'최종 모델 pct 평균': 'final_pct', 'std4 pct 평균': 'std4_pct'}
        if p == 'rg3':
            score_cols.update({'가스 RF 점수 평균': 'gas_rf', 'N1 점수 평균': 'n1'})
        agg = {'n': ('TimeStamp', 'size'), 'labeled_day': ('labeled_day', 'max'), 'ref_period': ('ref_period', 'mean')}
        agg.update({f'경보율|{nm}': ('R|' + nm, 'mean') for nm in R})
        agg.update({nm: (c, 'mean') for nm, c in score_cols.items()})
        day = f.groupby('plan_date').agg(**agg)
        day['err'] = err['불량_합계']
        day['불량률'] = day['err'] / (2 * day['n'])
        DAYS[p] = day
        use = day[(day['n'] >= min_shots) & day['err'].notna()]
        for subset, dd in (('labeled 날짜 제외', use[~use['labeled_day']]), ('전체 날짜', use)):
            for x in [c for c in day.columns if c.startswith('경보율|')] + list(score_cols):
                ok = dd[[x, '불량률']].dropna()
                if len(ok) < 5:
                    CORR.append({'제품': p, '날짜 범위': subset, 'x (날짜별)': x, '날짜 수': len(ok), '비고': '날짜 5개 미만'})
                    continue
                r, pv = _perm_p(ok[x].values, ok['불량률'].values, n=n_perm)
                CORR.append({'제품': p, '날짜 범위': subset, 'x (날짜별)': x, '날짜 수': len(ok), 'Spearman ρ': r, '순열 p(양측)': pv})
        f['month'] = pd.to_datetime(f['TimeStamp']).dt.to_period('M').astype(str)
        mon = f.groupby('month').agg(샷=('TimeStamp', 'size'), 정상기준기간_비율=('ref_period', 'mean'),
                                     **{nm: ('R|' + nm, 'mean') for nm in R}).reset_index()
        mon.insert(0, '제품', p)
        MON.append(mon)
        print(f'[{p}] unlabeled {len(f):,}샷 · 비교 날짜 {len(use)}일 (labeled 날짜 {int(use["labeled_day"].sum())}일)', flush=True)
    Cr, Mo = pd.DataFrame(CORR), pd.concat(MON, ignore_index=True)
    Cr.to_csv(f'{OUT}/unlabeled_daily_corr.csv', index=False, encoding='utf-8-sig')
    Mo.to_csv(f'{OUT}/unlabeled_monthly_rate.csv', index=False, encoding='utf-8-sig')
    for p, d in DAYS.items():
        d.to_csv(f'{OUT}/unlabeled_daily_{p}.csv', encoding='utf-8-sig')
    return {'corr': Cr, 'monthly': Mo, 'days': DAYS}


def plot_daily(U, p, xs=None, save=None):
    import matplotlib.pyplot as plt
    NC.style()
    d = U['days'][p]
    d = d[d['err'].notna() & (d['n'] >= 100)]
    xs = xs or [c for c in d.columns if c.startswith('경보율|')]
    fig, axes = plt.subplots(1, len(xs), figsize=(4.2 * len(xs), 3.6))
    for ax, x in zip(np.atleast_1d(axes), xs):
        for lb, c, mk in ((False, '#1f4e79', 'o'), (True, '#e67e22', 's')):
            m = d['labeled_day'] == lb
            ax.scatter(d.loc[m, x], d.loc[m, '불량률'], c=c, marker=mk, s=22, label='labeled 날짜' if lb else '그 외 날짜')
        ax.set_xlabel(x.replace('경보율|', '경보율 · '), fontsize=8); ax.set_ylabel('날짜별 불량률')
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(f'{p.upper()} unlabeled 날짜별 — 규칙 경보율 vs ERR_FACT_QTY 불량률', x=0.01, ha='left')
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
