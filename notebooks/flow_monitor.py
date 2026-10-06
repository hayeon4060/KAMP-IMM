"""
flow_monitor.py — 50번: 46·47·48번에서 나온 단서로 만든 '공정 흐름' 감시 신호 + 검증

배경 (46·47·48번 결과)
  - 46: 첫 경보가 Run 시작 3번째 샷에 몰림 → 재가동 직후는 원래 불안정 → 경보에서 빼야 함
        누적형(CUSUM)은 정상 기준(긴 구간)과 labeled(짧은 Run)의 길이 차이로 기준이 안 맞음 → 창(window) 방식으로
        RG3 미성형: EWMA·std4(라벨 미사용)가 43샷으로 3/8 (사후 선택)
  - 47: 최종 모델은 계단형 이상도 5샷 뒤에는 기준선(직전 20샷)이 따라가서 못 봄 → '고정 기준선'이 필요
        실제 가스 = 배럴·호퍼 +0.2~0.5배(약한 과열, 단발로는 탐지 한계 밖) → 여러 샷 평균으로 누적해야 보임
        실제 RG3 미성형 = 계량시간 +0.75배·배럴5 +0.9배 (재료부족형 아님, 계량지연형)
  - 48: RG3 미성형에서 계량지연(I5)·흔들림(I7)·호퍼 추세(I8)가 Run 4개 중 3개에서 + 방향 (단, 우연 수준)

신호 (모두 라벨 미사용, 재가동 후 WARM=10샷은 경보 제외)
  S1 EWMA·std4            : 흔들림 점수(probit)의 EWMA λ=0.2                         ← 46번 RG3 미성형 후보 (사후)
  S2 과열 드리프트         : 배럴2~5·호퍼의 최근 30샷 평균 − 재가동 구간 11~30샷 중앙값 (정상 변동 배수, 5개 평균, + 방향)
                             '고정 기준선' → 서서히 오르는 이상도 기준선이 따라가지 않음      ← 47번 가스 방향
  S3 계량 수준 이동        : 계량시간·사이클의 최근 10샷 평균 − 같은 고정 기준선 (+ 방향)   ← 47·48번 RG3 미성형 방향
  S4 변수 관계 변화        : 배럴2~5·호퍼·계량시간 15쌍의 20샷 이동 상관 → 정상 기준 Mahalanobis
                             (히터 제어·재료 공급이 불안정해지면 변수끼리 같이 움직이던 관계가 깨짐)
  S5 미성형 후보 결합      : max(S1, S3)의 정상 기준 백분위
  S6 가스 후보 결합        : max(S2, S4)의 정상 기준 백분위
  ※ S2·S3은 재가동 30샷 이후부터 점수 (기준선 계산에 필요)
경보 기준: 정상 기준 데이터에서 경보율 3·5·10%

검증
  1) 원형 이동 순열 p + **탐색 보정 p**: 순열마다 이 모듈 신호 6개 × 경보율 3개 중 '가장 잘 나온 것'의 분포와 비교
     (여러 개 중 고른 효과를 빼는 정직한 p)
  2) adjusted_46: RG3 미성형 EWMA·std4를 46번의 방법 17개 × 경보율 4개 전체와 같은 방식으로 탐색 보정
  3) 같은 검사 샷 수(K) 비교: 단발 std4 상위 K, 최종 모델 점수 상위 K
  4) synthetic_check: 47번과 같은 주입 실험에서 새 신호가 서서히 변하는 이상(과열·냉각)과 계량지연을 잡는지
  5) run_phase: 재가동 후 몇 샷째에 불량이 몰렸나 (R1 가스가 예열 단계인지)
  6) rg3_combo: RG3 가스 RF(기존) + 미성형 후보(S1·S3·S5, 라벨 미사용) 결합 vs N1 — 검사 비율별 + 순열(가스 RF만 재학습)
"""
import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import novel_common as NC
import cn7_alarm_plus_gas as C
import synthetic_injection as SI
from change_point import ewma
from final_model import raw

OUT = './outputs/flow_monitor'
WARM = 10
BASE = (11, 30)
HEAT = ['Barrel_Temperature_2', 'Barrel_Temperature_3', 'Barrel_Temperature_4', 'Barrel_Temperature_5', 'Hopper_Temperature']
METER = ['Plasticizing_Time', 'Cycle_Time']
COUP = HEAT + ['Plasticizing_Time']
RATES = (0.03, 0.05, 0.10)
SIGNALS = ('S1 EWMA·std4', 'S2 과열 드리프트', 'S3 계량 수준 이동', 'S4 변수 관계 변화', 'S5 미성형 후보(S1|S3)',
           'S6 가스 후보(S2|S4)')
ORIGIN = {'S1 EWMA·std4': '46번 RG3 미성형에서 고름(사후)', 'S2 과열 드리프트': '47번 실제 가스 방향',
          'S3 계량 수준 이동': '47·48번 RG3 미성형 방향', 'S4 변수 관계 변화': '새 가설(사전)',
          'S5 미성형 후보(S1|S3)': '결합', 'S6 가스 후보(S2|S4)': '결합'}


# ====================================================================== 신호 계산
def frozen_level(d, vars_, scale, win):
    """재가동 구간마다: 최근 win샷 평균 − 그 구간 11~30샷 중앙값 (정상 변동 배수), 변수 평균. 30샷 이전은 NaN"""
    out = np.full(len(d), np.nan)
    sr_all = d['since_restart'].values
    for ix in NC.segments(d):
        sr = sr_all[ix]
        bm = (sr >= BASE[0]) & (sr <= BASE[1])
        if bm.sum() < 10:
            continue
        zs = []
        for v in vars_:
            c = raw(v)
            if c not in d.columns:
                continue
            x = d[c].values[ix].astype(float)
            b = np.nanmedian(x[bm])
            m = pd.Series(x).rolling(win, min_periods=max(3, win // 2)).mean().values
            zs.append((m - b) / scale[v])
        if not zs:
            continue
        with np.errstate(all='ignore'), __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore')
            z = np.nanmean(np.vstack(zs), axis=0)
        z[sr <= BASE[1]] = np.nan
        out[ix] = z
    return out


def corr_features(d, win=20):
    """COUP 변수 15쌍의 20샷 이동 상관 (재가동 구간 안), 값이 일정해 상관이 없으면 0"""
    cols = [raw(v) for v in COUP if raw(v) in d.columns]
    pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    F = np.zeros((len(d), len(pairs)))
    for ix in NC.segments(d):
        sub = d.iloc[ix]
        for j, (a, b) in enumerate(pairs):
            F[ix, j] = sub[a].rolling(win, min_periods=win // 2).corr(sub[b]).values
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


class FlowScorer:
    """정상 기준(ref)으로 맞춘 S1~S6 계산기 — labeled·주입 구간 어디에나 같은 기준으로 적용"""

    def fit(self, ref, model):
        self.scale = SI.within_run_scale(ref, model)
        self.s4 = SI.Std4Scorer().fit(ref)
        Fr = corr_features(ref)
        self.cz = C._robust(Fr)
        self.clw = LedoitWolf().fit(self.cz(Fr))
        self.ref_raw = self._raw(ref, Fr)
        self.ref_pct = {k: NC.to_pct(v, v) for k, v in self.ref_raw.items()}
        return self

    def _raw(self, d, F=None):
        segs = NC.segments(d)
        F = corr_features(d) if F is None else F
        S = {'S1 EWMA·std4': ewma(NC.probit(self.s4.pct(d)), segs, 0.2),
             'S2 과열 드리프트': frozen_level(d, HEAT, self.scale, 30),
             'S3 계량 수준 이동': frozen_level(d, METER, self.scale, 10),
             'S4 변수 관계 변화': np.sqrt(self.clw.mahalanobis(self.cz(F)))}
        warm = d['since_restart'].values <= WARM
        for k in S:
            S[k] = np.where(warm, np.nan, S[k])
        return S

    def score(self, d):
        S = self._raw(d)
        P = {k: NC.to_pct(self.ref_raw[k], v) for k, v in S.items()}
        for k in S:
            P[k] = np.where(np.isfinite(S[k]), P[k], np.nan)
        S['S5 미성형 후보(S1|S3)'] = np.fmax(P['S1 EWMA·std4'], P['S3 계량 수준 이동'])
        S['S6 가스 후보(S2|S4)'] = np.fmax(P['S2 과열 드리프트'], P['S4 변수 관계 변화'])
        return S

    def ref_scores(self):
        R = dict(self.ref_raw)
        P = self.ref_pct
        R['S5 미성형 후보(S1|S3)'] = np.fmax(np.where(np.isfinite(R['S1 EWMA·std4']), P['S1 EWMA·std4'], np.nan),
                                         np.where(np.isfinite(R['S3 계량 수준 이동']), P['S3 계량 수준 이동'], np.nan))
        R['S6 가스 후보(S2|S4)'] = np.fmax(np.where(np.isfinite(R['S2 과열 드리프트']), P['S2 과열 드리프트'], np.nan),
                                       np.where(np.isfinite(R['S4 변수 관계 변화']), P['S4 변수 관계 변화'], np.nan))
        return R


def flag_of(ref_s, lab_s, rate):
    thr = NC.threshold(ref_s, rate)
    return np.nan_to_num(lab_s, nan=-np.inf) > thr


# ====================================================================== 탐색 보정 순열
def search_adjusted(lab, flags, n=2000, seed=11):
    """flags: {이름: bool 배열}. 순열마다 모든 flag의 (불량·가스·미성형) 포착 수 최대값 → 각 flag의 보정 p"""
    ev = np.flatnonzero(lab['eval'].values)
    t = lab['type'].values[ev]
    names = list(flags)
    F = np.column_stack([np.asarray(flags[k], bool)[ev] for k in names]).astype(int)
    runs = lab['run_id'].values[ev]
    groups = [np.flatnonzero(runs == r) for r in pd.unique(runs)]
    tys = {'불량': lambda tt: tt != '정상', '가스': lambda tt: tt == '가스', '미성형': lambda tt: tt == '미성형'}
    obs = {k: F.T @ f(t).astype(int) for k, f in tys.items()}
    mx = {k: np.empty(n) for k in tys}
    rng = np.random.default_rng(seed)
    for b in range(n):
        tt = t.copy()
        for g in groups:
            tt[g] = np.roll(t[g], rng.integers(0, len(g)))
        for k, f in tys.items():
            mx[k][b] = (F.T @ f(tt).astype(int)).max()
    out = {}
    for i, nm in enumerate(names):
        out[nm] = {f'보정p_{k}': (float((1 + (mx[k] >= obs[k][i]).sum()) / (n + 1)) if obs[k][i] > 0 else np.nan)
                   for k in tys}
    return out


# ====================================================================== 실행
def run(products=('cn7', 'rg3'), rates=RATES, n_perm=1000, n_adj=2000, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    rows, EP, PH, CTX = [], [], [], {}
    for p in products:
        ref, lab, model = NC.load(p)
        fs = FlowScorer().fit(ref, model)
        Sr, Sl = fs.ref_scores(), fs.score(lab)
        s4l, _ = NC.std4_pct(ref, lab)
        comp = {'단발 std4 상위 K': s4l, '최종 모델 점수 상위 K': lab['score'].values}
        flags = {}
        for name in SIGNALS:
            for rate in rates:
                f = flag_of(Sr[name], Sl[name], rate)
                flags[(name, rate)] = f
                r = NC.evaluate(lab, f, name, 제품=p, 출처=ORIGIN[name], 목표경보율=rate)
                r.update(NC.circ_null(lab, f, n_perm))
                for cn, cs in comp.items():
                    ck = NC.evaluate(lab, NC.topk_flag(lab, cs, max(r['경보(검사) 샷'], 1)))
                    for ty in ('불량',) + NC.TYPES:
                        if ty in ck:
                            r[f'[같은 K] {cn} {ty}'] = ck[ty]
                rows.append(r)
        adj = search_adjusted(lab, {f'{k[0]}|{k[1]}': v for k, v in flags.items()}, n_adj)
        for r in rows:
            if r['제품'] == p:
                r.update(adj[f'{r["방법"]}|{r["목표경보율"]}'])
        EP.append(episode_table(p, lab, Sr, Sl))
        PH.append(run_phase(p, lab))
        CTX[p] = {'ref': ref, 'lab': lab, 'model': model, 'scorer': fs, 'Sr': Sr, 'Sl': Sl}
        if verbose:
            print(f'[{p}] 완료', flush=True)
    T = pd.DataFrame(rows)
    E, Ph = pd.concat(EP, ignore_index=True), pd.concat(PH, ignore_index=True)
    T.to_csv(f'{OUT}/summary.csv', index=False, encoding='utf-8-sig')
    E.to_csv(f'{OUT}/episodes.csv', index=False, encoding='utf-8-sig')
    Ph.to_csv(f'{OUT}/run_phase.csv', index=False, encoding='utf-8-sig')
    return {'table': T, 'episodes': E, 'phase': Ph, 'ctx': CTX}


def view(R, p, rate=0.05):
    T = R['table']
    t = T[(T['제품'] == p) & (T['목표경보율'] == rate)]
    cols = ['방법', '출처', '경보(검사) 샷', '불량 없는 Run 경보율', '불량', '가스', '미성형', 'Precision', 'F1',
            'p_가스', '보정p_가스', 'p_미성형', '보정p_미성형'] + [c for c in t.columns if c.startswith('[같은 K]')]
    return t[[c for c in cols if c in t.columns]].reset_index(drop=True)


def episode_table(p, lab, Sr, Sl, rate=0.05, before=50):
    """재가동 후 10샷 제외한 경보로: 불량 Run마다 첫 경보 seq, 미성형 직전 before샷 경보 수, 그 Run 가스 포착"""
    ev, t, seq = lab['eval'].values, lab['type'].values, lab['seq'].values
    y = (lab['PassOrFail'].values == 1) & ev
    rows = []
    for r in pd.unique(lab.loc[y, 'run_id']):
        m = (lab['run_id'] == r).values
        sh = seq[m & ev & (t == '미성형')]
        fs = int(sh.min()) if len(sh) else None
        for name in SIGNALS:
            a = m & ev & flag_of(Sr[name], Sl[name], rate)
            row = {'제품': p, 'Run': r, '방법': name, '경보율': rate, 'Run 불량': int((m & y).sum()),
                   'Run 가스': int((m & ev & (t == '가스')).sum()), '첫 불량 seq': int(seq[m & y].min()),
                   '첫 미성형 seq': fs, '첫 경보 seq': int(seq[a].min()) if a.any() else None,
                   'Run 안 경보 수': int(a.sum()), '잡은 가스': int((a & (t == '가스')).sum()),
                   '잡은 미성형': int((a & (t == '미성형')).sum())}
            if fs is not None:
                pre = a & (seq < fs) & (seq >= fs - before)
                row[f'미성형 직전 {before}샷 경보'] = int(pre.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def run_phase(p, lab, bins=(0, 10, 30, 60, 100, 300, 10 ** 9)):
    """재가동 후 몇 샷째인지 구간별: 평가 샷 수, 가스·미성형 수와 비율"""
    ev = lab['eval'].values
    t = lab['type'].values
    cut = pd.cut(lab['since_restart'], bins, labels=[f'{a + 1}~{b}' if b < 10 ** 9 else f'{a + 1}~' for a, b in zip(bins[:-1], bins[1:])])
    d = pd.DataFrame({'구간': cut[ev], 'type': t[ev], 'run': lab['run_id'].values[ev]})
    g = d.groupby('구간', observed=False)
    out = pd.DataFrame({'평가 샷': g.size(), '가스': g['type'].apply(lambda s: (s == '가스').sum()),
                        '미성형': g['type'].apply(lambda s: (s == '미성형').sum())}).reset_index()
    out['가스 비율'] = out['가스'] / out['평가 샷'].replace(0, np.nan)
    out['미성형 비율'] = out['미성형'] / out['평가 샷'].replace(0, np.nan)
    out.insert(0, '제품', p)
    return out


# ====================================================================== 46번 전체 탐색 기준 보정 (RG3 미성형 EWMA·std4)
def adjusted_46(p='rg3', rates=(0.01, 0.03, 0.05, 0.10), n=2000, target='미성형'):
    """46번 방법 17개 × 경보율 4개 + 이번 S1을 한 묶음으로 탐색 보정 → '46번에서 골라낸 효과'를 뺀 p"""
    import change_point as CP
    ref, lab, S = CP.statistics(p)
    flags = {f'{nm}|{r}': S[nm][1] > NC.threshold(S[nm][0], r) for nm in S for r in rates}
    adj = search_adjusted(lab, flags, n)
    rows = []
    for k, f in flags.items():
        e = NC.evaluate(lab, f, k)
        rows.append({'방법|경보율': k, '경보 샷': e['경보(검사) 샷'], target: e.get(target), **adj[k]})
    out = pd.DataFrame(rows).sort_values(f'보정p_{target}').reset_index(drop=True)
    out.to_csv(f'{OUT}/adjusted_46_{p}.csv', index=False, encoding='utf-8-sig')
    return out


# ====================================================================== 47번 주입 실험으로 새 신호 확인
def synthetic_check(products=('cn7', 'rg3'), n_ep=150, mags=(0.5, 1, 2, 3, 5, 8), rate=0.01, R=None, seed=0):
    """정상 구간에 47번 패턴 주입 → 새 신호(정상 기준 경보율 1%) vs 최종 모델 q0.999 탐지율"""
    from change_point import cusum
    rows = []
    for p in products:
        if R is not None and p in R['ctx']:
            ref, model, fs = R['ctx'][p]['ref'], R['ctx'][p]['model'], R['ctx'][p]['scorer']
        else:
            ref, _, model = NC.load(p)
            fs = FlowScorer().fit(ref, model)
        Sr = fs.ref_scores()
        thr = {k: NC.threshold(Sr[k], rate) for k in SIGNALS}
        cthr = NC.threshold(cusum(NC.probit(ref['pct'].values), NC.segments(ref), 0.5), 0.01)
        wins = SI.pick_windows(ref, n_ep, seed)
        jobs = [(None, 0)] + [(tpl, a) for tpl in SI.TEMPLATES for a in mags]
        for tpl, a in jobs:
            f = SI.score_frame(SI.build(ref, wins, tpl, a, fs.scale, seed), model, fs.s4, cthr)
            f['since_restart'] = f.groupby('run_id', sort=False).cumcount() + 1      # 창 시작 = 재가동으로 봄
            S = fs.score(f)
            inj = f['pos'].values >= 0
            dets = {'최종 모델 q0.999': f['final_pct'].values >= 0.999, 'std4 q0.999': f['std4_pct'].values >= 0.999}
            dets.update({k: np.nan_to_num(S[k], nan=-np.inf) > thr[k] for k in SIGNALS})
            for dn, hit in dets.items():
                g = pd.DataFrame({'ep': f['ep'].values[inj], 'hit': hit[inj], 'pos': f['pos'].values[inj]})
                rows.append({'제품': p, '패턴': tpl or '주입 없음', '크기': a, '탐지기': dn,
                             '에피소드 탐지율': float(g.groupby('ep')['hit'].any().mean()),
                             '샷 탐지율': float(g['hit'].mean()),
                             '뒤 5샷 탐지율': float(g.loc[g['pos'] >= SI.L - 5, 'hit'].mean())})
        print(f'  [{p}] 주입 확인 완료', flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(f'{OUT}/synthetic_check.csv', index=False, encoding='utf-8-sig')
    return T


def synthetic_pivot(T, p, col='에피소드 탐지율', patterns=None):
    t = T[T['제품'] == p]
    if patterns:
        t = t[t['패턴'].isin(patterns)]
    return t.pivot_table(index=['패턴', '크기'], columns='탐지기', values=col, sort=False).round(2)


# ====================================================================== RG3: 가스 RF + 라벨 없는 미성형 후보
def rg3_combo(R, seeds=(0, 1, 2), shares=(1.0, 0.68, 0.5), rates=(0.05, 0.10, 0.15), base_dir='.'):
    import semi_normal as SN
    import inspect_curve as IC
    D = SN.load_std('rg3', base_dir)
    labD = D['lab']
    lab = R['ctx']['rg3']['lab']
    Sl = R['ctx']['rg3']['Sl']
    pos = pd.Series(np.arange(len(lab)), index=lab['TimeStamp'].values)
    ii = pos.reindex(labD['TimeStamp'].values).values
    shorts = {nm: np.where(np.isfinite(ii), np.asarray(Sl[nm])[np.nan_to_num(ii, nan=0).astype(int)], np.nan)
              for nm in ('S1 EWMA·std4', 'S3 계량 수준 이동', 'S5 미성형 후보(S1|S3)')}
    rows = []
    for sd in seeds:
        gas = SN.loro(D, 'M0_supervised', '가스', sd)
        n1 = SN.loro(D, 'N1_normal_augment', '미성형', sd)
        for nm, sh in {**shorts, 'N1 (41·43번, 비교)': n1}.items():
            for g in shares:
                if g == 1.0 and nm != 'N1 (41·43번, 비교)':
                    continue
                T = IC.split_curve(labD, gas, sh, g, rates=rates, name=('가스 RF만' if g == 1.0 else f'가스 RF + {nm} (g={g})'))
                T['seed'] = sd
                rows.append(T)
    C_ = pd.concat(rows, ignore_index=True)
    for col in ('불량', '가스', '미성형'):
        C_[f'{col} 수'] = C_[col].str.split('/').str[0].astype(int)
    summ = C_.groupby(['모델', '검사 비율'], sort=False).agg(
        검사_샷=('검사 샷', 'first'), 불량=('불량 수', 'mean'), 가스=('가스 수', 'mean'), 미성형=('미성형 수', 'mean'),
        미성형_범위=('미성형 수', lambda x: f'{x.min()}~{x.max()}'), Precision=('Precision', 'mean'),
        Recall=('Recall', 'mean'), F1=('F1', 'mean')).reset_index()
    summ.to_csv(f'{OUT}/rg3_combo.csv', index=False, encoding='utf-8-sig')
    return {'summary': summ, 'curves': C_, 'D': D, 'shorts': shorts}


def rg3_combo_permutation(RC, short='S1 EWMA·std4', share=0.68, rate=0.10, n=30, seed=0):
    """라벨 원형 이동 → 가스 RF만 재학습(미성형 후보는 라벨을 안 쓰므로 그대로) → 결합 포착 수의 우연 분포"""
    import semi_normal as SN
    import inspect_curve as IC
    D = RC['D']
    lab0 = D['lab']
    sh = RC['shorts'][short]

    def stat(Dx):
        g = SN.loro(Dx, 'M0_supervised', '가스', seed)
        T = IC.split_curve(Dx['lab'], g, sh, share, rates=(rate,))
        a, b = int(T['가스'].iloc[0].split('/')[0]), int(T['미성형'].iloc[0].split('/')[0])
        return a, b, a + b

    obs = stat(D)
    ev = np.flatnonzero(lab0['eval'].values)
    runs = lab0['run_id'].values
    rng = np.random.default_rng(seed + 900)
    null = []
    for b in range(n):
        L = lab0.copy()
        tcol = L['type'].values.copy()
        for r in pd.unique(runs[ev]):
            ii = ev[runs[ev] == r]
            tcol[ii] = np.roll(lab0['type'].values[ii], rng.integers(0, len(ii)))
        L['type'] = tcol
        L['PassOrFail'] = np.where(np.isin(tcol, ['가스', '미성형', '기타']), 1, 0)
        null.append(stat(dict(D, lab=L)))
        if b % 10 == 0:
            print('  순열', b, flush=True)
    null = np.asarray(null)
    return pd.DataFrame([{'지표': nm, '관측': obs[i], '무작위 평균': round(float(null[:, i].mean()), 2),
                          '무작위 95%': float(np.quantile(null[:, i], 0.95)),
                          'p': float((1 + (null[:, i] >= obs[i]).sum()) / (n + 1))}
                         for i, nm in enumerate(('가스', '미성형', '합계'))])
