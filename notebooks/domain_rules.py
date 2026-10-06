"""
domain_rules.py — 48번: 도메인 지식(사출 물리 · KAMP 가이드북 · SPC) 규칙 — 학습 없이 미리 정한 규칙만

왜: 불량 에피소드가 1~2개뿐이면 데이터에서 규칙을 '배우면' 과적합 → 규칙을 지식으로 먼저 정하고 데이터는 채점에만 사용.
    12·13번(물리 검증)은 Run 안 z(미래 샷까지 사용)로 분석용이었음 → 여기서는 전부 '인과적'(과거 샷만) 지표 + 라벨 없는 경보 기준.

지표 (최종 모델 입력 = 같은 Run 과거 20샷 기준 편차, 정상 기준으로 표준화. + 방향 = 해당 메커니즘)
  I1 재료부족      (미성형) = (−충전시간 + 사출속도 − 사출압) / 3         가이드북: 수지량 부족·계량 부족
  I2 쿠션부족      (미성형) = −쿠션                                      가이드북: 보압 여유 없음
  I3 금형 냉각     (미성형) = −금형온도3 편차                            가이드북: 금형온도 낮음 → 조기 고화
  I4 점도↑         (미성형) = 흐름저항(사출압/속도) 편차                  가이드북: 수지온도 낮음·유동성 부족
  I5 계량지연      (가스)   = 계량시간 편차                              가이드북: 재료 공급·수분·계량 불안정
  I6 체류시간↑     (가스)   = 사이클 편차                                가이드북: 체류 길어짐 → 열분해 가스
  I7 공정 흔들림   (가스)   = std4 흔들림 점수(probit)                    ⚠ RG3 가스에서 나온 가설 → RG3에는 사후, CN7에는 독립 검증
  I8 호퍼 과열 추세(가스)   = 호퍼 기울기(20샷)                           ⚠ CN7 R1에서 본 신호 → CN7에는 사후
규칙
  - 단일 지표: 정상 기준 데이터에서 경보율 1·3·5%가 되는 임계값 초과
  - 유형 묶음(사전 등록): 미성형 규칙 = I1~I4 중 하나 / 가스 규칙 = I5·I6·I7 중 하나
      묶음은 '지표별 정상 기준 백분위의 최대값'에 같은 경보율 기준 → 묶어도 전체 경보율은 그대로
  - 지속 규칙(디바운스): 위 규칙이 최근 3샷 중 2샷 이상 → 순간 튀는 값 제거
  - SPC Western Electric(관리도 규칙, 위쪽만) — final · std4 신호 (probit, 정상 ≈ N(0,1)):
      WE1: 1샷 > 3σ / WE2: 3샷 중 2샷 > 2σ / WE3: 5샷 중 4샷 > 1σ / WE4: 8샷 연속 > 0
      (경보율을 맞추지 않은 고정 규칙 → 정상 기준 경보율을 같이 보고)
검증
  - 원형 이동 순열 p + 사전 등록 규칙끼리 Holm 보정 (사후 규칙은 보정 대상에서 빼고 '사후'로 표시)
  - 방향 일치: 불량이 있는 Run마다 불량 샷의 지표 중앙값이 + 쪽인가 (Run 몇 개 중 몇 개)
"""
import os
import numpy as np
import pandas as pd

import novel_common as NC
from final_model import FEATURES

OUT = './outputs/domain_rules'
RATES = (0.01, 0.03, 0.05)

IND = {   # 이름: (유형, 출처, 계산식)
    'I1 재료부족': ('미성형', '가이드북', lambda f, s: (-f('dev_Filling_Time') + f('dev_Max_Injection_Speed') - f('dev_Max_Injection_Pressure')) / 3),
    'I2 쿠션부족': ('미성형', '가이드북', lambda f, s: -f('dev_Cushion_Position')),
    'I3 금형 냉각': ('미성형', '가이드북', lambda f, s: -f('dev_Mold_Temperature_3')),
    'I4 점도↑(흐름저항)': ('미성형', '가이드북', lambda f, s: f('dev_flow_res')),
    'I5 계량지연': ('가스', '가이드북', lambda f, s: f('dev_Plasticizing_Time')),
    'I6 체류시간↑': ('가스', '가이드북', lambda f, s: f('dev_Cycle_Time')),
    'I7 공정 흔들림(std4)': ('가스', 'RG3 가스에서 온 가설', lambda f, s: s),
    'I8 호퍼 과열 추세': ('가스', 'CN7 R1에서 본 신호', lambda f, s: f('hopper_slope20')),
}
GROUPS = {'미성형 규칙(I1~I4)': ['I1 재료부족', 'I2 쿠션부족', 'I3 금형 냉각', 'I4 점도↑(흐름저항)'],
          '가스 규칙(I5·I6·I7)': ['I5 계량지연', 'I6 체류시간↑', 'I7 공정 흔들림(std4)'],
          '가스 규칙 + 호퍼(I5~I8, 사후 포함)': ['I5 계량지연', 'I6 체류시간↑', 'I7 공정 흔들림(std4)', 'I8 호퍼 과열 추세'],
          '전체(I1~I7)': ['I1 재료부족', 'I2 쿠션부족', 'I3 금형 냉각', 'I4 점도↑(흐름저항)', 'I5 계량지연', 'I6 체류시간↑',
                         'I7 공정 흔들림(std4)']}


def posthoc(name, p):
    """CN7: 호퍼(I8)는 CN7 R1을 보고 나온 신호 / RG3: std4(I7)는 RG3 가스를 보고 나온 신호"""
    cn = ('I8' in name) or ('사후' in name)
    rg = ('I7' in name) or ('std4' in name) or ('전체' in name) or ('가스 규칙' in name)
    return (p == 'cn7' and cn) or (p == 'rg3' and rg)


def indicators(ref, lab, model):
    """반환: {지표: (ref 값, lab 값)}"""
    Zr, Zl = model._prep(ref), model._prep(lab)
    s4l, s4r = NC.std4_pct(ref, lab)
    out = {}
    for name, (_, _, fn) in IND.items():
        fr = lambda c: Zr[:, FEATURES.index(c)]
        fl = lambda c: Zl[:, FEATURES.index(c)]
        out[name] = (fn(fr, NC.probit(s4r)), fn(fl, NC.probit(s4l)))
    return out


def persist(flag, segs, need=2, win=3):
    """최근 win샷 중 need샷 이상 경보 (재가동 구간 안에서만)"""
    f = np.asarray(flag, int)
    out = np.zeros(len(f), bool)
    for ix in segs:
        c = pd.Series(f[ix]).rolling(win, min_periods=1).sum().values
        out[ix] = c >= need
    return out


def western_electric(z, segs):
    """위쪽 Western Electric 규칙 4개 (z: 정상 ≈ N(0,1))"""
    z = np.where(np.isfinite(z), z, 0.0)
    R = {k: np.zeros(len(z), bool) for k in ('WE1', 'WE2', 'WE3', 'WE4')}
    for ix in segs:
        s = pd.Series(z[ix])
        R['WE1'][ix] = (s > 3).values
        R['WE2'][ix] = ((s > 2).astype(int).rolling(3, min_periods=3).sum() >= 2).values
        R['WE3'][ix] = ((s > 1).astype(int).rolling(5, min_periods=5).sum() >= 4).values
        R['WE4'][ix] = ((s > 0).astype(int).rolling(8, min_periods=8).sum() >= 8).values
    R['WE 아무거나'] = R['WE1'] | R['WE2'] | R['WE3'] | R['WE4']
    return R


def rules(ref, lab, model, rates=RATES):
    """반환: list of (규칙 이름, 유형, 출처, 목표경보율, ref 경보, lab 경보)"""
    I = indicators(ref, lab, model)
    P = {k: (NC.to_pct(vr, vr), NC.to_pct(vr, vl)) for k, (vr, vl) in I.items()}   # 지표별 정상 기준 백분위
    sr, sl = NC.segments(ref), NC.segments(lab)
    out = []
    for rate in rates:
        for name, (ty, src, _) in IND.items():
            vr, vl = I[name]
            thr = NC.threshold(vr, rate)
            out.append((name, ty, src, rate, vr > thr, vl > thr))
        for g, mem in GROUPS.items():
            mr = np.max([P[k][0] for k in mem], axis=0)
            ml = np.max([P[k][1] for k in mem], axis=0)
            thr = NC.threshold(mr, rate)
            ty = '미성형' if g.startswith('미성형') else '가스' if g.startswith('가스') else '전체'
            src = '가이드북 묶음' if '사후' not in g else '사후 포함 묶음'
            out.append((g, ty, src, rate, mr > thr, ml > thr))
            out.append((g + ' · 3샷 중 2샷', ty, src, rate, persist(mr > thr, sr), persist(ml > thr, sl)))
    sig = NC.base_signals(ref, lab)
    for sname in ('final', 'std4'):
        Wr, Wl = western_electric(sig[sname][0], sr), western_electric(sig[sname][1], sl)
        for k in Wr:
            out.append((f'SPC {k} · {sname}', '전체', 'SPC 관리도 규칙', np.nan, Wr[k], Wl[k]))
    return out


def run(products=('cn7', 'rg3'), rates=RATES, n_perm=1000, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    rows, DIR = [], []
    for p in products:
        ref, lab, model = NC.load(p)
        for name, ty, src, rate, fr, fl in rules(ref, lab, model, rates):
            r = NC.evaluate(lab, fl, name, 제품=p, 대상유형=ty, 출처=src, 목표경보율=rate,
                            **{'정상 기준 실제 경보율': float(np.mean(fr))})
            r['사후 여부'] = '사후' if posthoc(name, p) else '사전'
            r.update(NC.circ_null(lab, fl, n_perm))
            rows.append(r)
        DIR.append(direction_table(p, lab, indicators(ref, lab, model)))
        if verbose:
            print(f'[{p}] 완료', flush=True)
    T = pd.DataFrame(rows)
    # Holm: 제품·경보율별로 '사전' 단일 지표 + 사전 묶음의 대상 유형 p
    T['대상 유형 p'] = [r[f'p_{r["대상유형"]}'] if r['대상유형'] in ('가스', '미성형') else r['p_불량'] for _, r in T.iterrows()]
    T['Holm 보정 p'] = np.nan
    for (p, rate), g in T[(T['사후 여부'] == '사전') & T['목표경보율'].notna()].groupby(['제품', '목표경보율']):
        T.loc[g.index, 'Holm 보정 p'] = NC.holm(g['대상 유형 p'].values)
    D = pd.concat(DIR, ignore_index=True)
    T.to_csv(f'{OUT}/rules.csv', index=False, encoding='utf-8-sig')
    D.to_csv(f'{OUT}/direction.csv', index=False, encoding='utf-8-sig')
    return {'table': T, 'direction': D}


def direction_table(p, lab, I):
    """지표별·유형별: 불량 샷 지표 중앙값, 정상 샷 중앙값, 불량 Run 중 불량 샷 중앙값이 + 인 Run 수"""
    ev, t = lab['eval'].values, lab['type'].values
    rows = []
    for name, (_, vl) in I.items():
        ty0 = IND[name][0]
        for ty in ('가스', '미성형'):
            m = ev & (t == ty)
            if not m.any():
                continue
            runs = pd.unique(lab.loc[m, 'run_id'])
            pos = sum(np.nanmedian(vl[m & (lab['run_id'] == r).values]) > 0 for r in runs)
            rows.append({'제품': p, '지표': name, '지표가 겨냥한 유형': ty0, '불량 유형': ty,
                         '불량 샷 중앙값': float(np.nanmedian(vl[m])), '정상 샷 중앙값': float(np.nanmedian(vl[ev & (t == '정상')])),
                         '+ 방향 Run': f'{pos}/{len(runs)}', '불량 샷 중 + 비율': float(np.mean(vl[m] > 0))})
    return pd.DataFrame(rows)


def view(R, p, rate=0.03, target=None):
    T = R['table']
    t = T[(T['제품'] == p) & ((T['목표경보율'] == rate) | T['목표경보율'].isna())]
    if target:
        t = t[t['대상유형'].isin([target, '전체'])]
    cols = ['방법', '대상유형', '출처', '사후 여부', '정상 기준 실제 경보율', '불량 없는 Run 경보율', '경보(검사) 샷', '불량', '가스', '미성형',
            'Precision', 'Recall', 'F1', '대상 유형 p', 'Holm 보정 p']
    return t[[c for c in cols if c in t.columns]].reset_index(drop=True)
