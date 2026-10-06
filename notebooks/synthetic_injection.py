"""
synthetic_injection.py — 47번: 물리 기반 합성 불량 주입 → 모델 '탐지 한계' 측정

왜: 실제 불량은 CN7 R1 한 에피소드뿐 → 실제 라벨로는 '어떤 종류·크기의 이상을 잡을 수 있나'를 말할 수 없음.
    정상 샷(unlabeled 기준 데이터)에 사출 물리로 정한 불량 패턴을 크기별로 넣고, 모델이 몇 %를 잡는지 보면
    라벨 없이도 탐지 한계(최소 탐지 크기)를 정량화할 수 있음. (22번의 '일반 이상치 주입'과 달리 메커니즘별 패턴)

불량 패턴 (사출 물리 · KAMP 가이드북의 불량 원인으로 결과 보기 전에 고정)
  미성형_재료부족 : 충전시간↓ · 사출속도↑ · 사출압↓ · 쿠션↓              (수지 부족 → 흐름저항↓)   계단형
  미성형_냉각점도 : 금형온도3↓ · 사출압↑ · 충전시간↑ · 사출속도↓          (조기 고화·점도↑)          서서히(램프)
  가스_과열       : 배럴2~5↑ · 호퍼↑ · 사이클↑                            (과열·체류 → 열분해)        서서히(램프)
  가스_공급불안정 : 배럴2·호퍼·쿠션·배럴5·계량시간에 샷마다 흔들림 추가     (std4 가설)                 잡음형
  가스_계량지연   : 계량시간↑ · 사이클↑                                    (재료 공급·계량 지연)        계단형
크기: 변수별 '같은 Run 안 정상 변동'(Run 안 IQR/1.349의 Run 중앙값, 최소 = 기록 단위)의 a배, a = 0.5·1·2·3·5·8
절차
  1) 정상 기준 데이터에서 재가동 구간 안의 위치를 n_ep개 무작위로 고름 (앞 60샷 이상 있는 곳)
  2) 구간 [시작−60, 시작+L) 을 잘라 L=20샷에 패턴 주입 (같은 위치를 모든 패턴·크기에 사용 = 짝 비교)
  3) 최종 모델과 같은 파생변수 계산(학습 때 기준값 그대로) → 점수 → pct
  4) 탐지: 주입 샷 중 pct ≥ q 비율, 에피소드 탐지율(20샷 중 하나라도), 첫 탐지까지 샷 수
     비교 기준 = 주입 안 한 같은 구간(a=0)의 경보율 (오경보)
탐지기: 최종 모델(q 0.99·0.999) / std4 흔들림(q 0.99·0.999) / CUSUM(최종, k=0.5, 정상 기준 경보율 1%) / 둘 중 하나(최종 or std4, q0.999)
현실성 점검: 실제 labeled 불량 샷이 같은 단위(정상 변동의 몇 배)로 얼마나 벗어났는지 → 곡선 위에 실제 불량 위치를 놓아 볼 수 있음
한계: 파생변수 중 'Run 첫 5샷 대비 상승'은 잘라낸 구간 시작 기준으로 계산됨 / 실제 불량이 이 패턴대로 생긴다는 보장은 없음
"""
import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import novel_common as NC
import cn7_alarm_plus_gas as C
from final_model import raw, make_features, ALL_RAW

OUT = './outputs/synthetic_injection'
MAGS = (0.5, 1, 2, 3, 5, 8)
L, PRE = 20, 60
STD4 = C.STD4

TEMPLATES = {
    '미성형_재료부족': dict(type='미성형', shape='step',
                      vars={'Filling_Time': -1, 'Max_Injection_Speed': +1, 'Max_Injection_Pressure': -1, 'Cushion_Position': -1}),
    '미성형_냉각점도': dict(type='미성형', shape='ramp',
                      vars={'Mold_Temperature_3': -1, 'Max_Injection_Pressure': +1, 'Filling_Time': +1, 'Max_Injection_Speed': -1}),
    '가스_과열': dict(type='가스', shape='ramp',
                  vars={'Barrel_Temperature_2': +1, 'Barrel_Temperature_3': +1, 'Barrel_Temperature_4': +1,
                        'Barrel_Temperature_5': +1, 'Hopper_Temperature': +1, 'Cycle_Time': +1}),
    '가스_공급불안정': dict(type='가스', shape='noise',
                     vars={'Barrel_Temperature_2': 1, 'Hopper_Temperature': 1, 'Cushion_Position': 1,
                           'Barrel_Temperature_5': 1, 'Plasticizing_Time': 1}),
    '가스_계량지연': dict(type='가스', shape='step', vars={'Plasticizing_Time': +1, 'Cycle_Time': +1}),
}


# ====================================================================== 준비
def within_run_scale(ref, model):
    """변수별 '같은 Run 안 정상 변동' = Run별 IQR/1.349의 중앙값 (최소 = 기록 단위)"""
    out = {}
    for v in ALL_RAW:
        c = raw(v)
        if c not in ref.columns:
            continue
        g = ref.groupby('run_id')[c]
        s = ((g.quantile(0.75) - g.quantile(0.25)) / 1.349).replace(0, np.nan).median()
        step = (getattr(model, 'steps', {}) or {}).get(v, None)
        if step is None or not np.isfinite(step):
            from final_model import resolution
            step = resolution(ref[c])
        out[v] = float(max(s if np.isfinite(s) else 0.0, step, 1e-6))
    return out


class Std4Scorer:
    """정상 기준으로 학습한 흔들림 점수 (직전 k샷 std 4개 → robust z → LedoitWolf Mahalanobis) — 임의 샷 묶음에 적용"""

    def __init__(self, k=10):
        self.k = k

    def _X(self, d):
        return np.column_stack([d[raw(v)].groupby(d['run_id'], sort=False)
                               .transform(lambda s: s.rolling(self.k, min_periods=3).std()).values for v in STD4])

    def fit(self, ref):
        Xr = self._X(ref)
        self.z = C._robust(Xr)
        self.lw = LedoitWolf().fit(self.z(Xr))
        self.s_ref = np.sort(np.sqrt(self.lw.mahalanobis(self.z(Xr))))
        return self

    def pct(self, d):
        s = np.sqrt(self.lw.mahalanobis(self.z(self._X(d))))
        return np.searchsorted(self.s_ref, s, side='right') / (len(self.s_ref) + 1)


def pick_windows(ref, n_ep, seed=0):
    """재가동 구간 안에서 앞에 PRE샷, 뒤에 L샷이 있는 시작 위치를 n_ep개"""
    rng = np.random.default_rng(seed)
    cand = []
    for ix in NC.segments(ref):
        if len(ix) >= PRE + L:
            cand += [(ix, j) for j in range(PRE, len(ix) - L + 1, L)]
    pick = rng.choice(len(cand), min(n_ep, len(cand)), replace=False)
    return [cand[i][0][cand[i][1] - PRE: cand[i][1] + L] for i in pick]


def build(ref, windows, tpl, a, scale, seed=0):
    """창마다 앞 PRE샷은 그대로, 뒤 L샷에 패턴 주입 → 하나의 DataFrame (run_id = 창 번호)"""
    rng = np.random.default_rng(seed)
    cols = ['run_id', 'TimeStamp'] + [raw(v) for v in ALL_RAW if raw(v) in ref.columns]
    parts = []
    ramp = np.linspace(1 / L, 1, L)
    for e, ix in enumerate(windows):
        w = ref.iloc[ix][cols].copy().reset_index(drop=True)
        w['run_id'] = f'ep{e:04d}'
        w['ep'], w['pos'] = e, np.arange(len(w)) - PRE            # pos ≥ 0 = 주입 샷
        if tpl is not None and a > 0:
            T = TEMPLATES[tpl]
            for v, sgn in T['vars'].items():
                c = raw(v)
                if c not in w.columns:
                    continue
                d = a * scale[v]
                if T['shape'] == 'step':
                    add = sgn * d * np.ones(L)
                elif T['shape'] == 'ramp':
                    add = sgn * d * ramp
                else:
                    add = rng.normal(0, d, L)
                w.loc[PRE:, c] = w.loc[PRE:, c].values + add
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def score_frame(W, model, s4, cusum_thr):
    from change_point import cusum
    f = make_features(W, model.steps, model.fr_step)              # run_id(=창)·시간 순 정렬
    f['final_pct'] = model.pct(model.score(f))
    f['std4_pct'] = s4.pct(f)
    segs = [np.asarray(ix) for ix in f.groupby('run_id', sort=False).indices.values()]
    f['cusum'] = cusum(NC.probit(f['final_pct'].values), segs, 0.5)
    f['cusum_flag'] = f['cusum'] > cusum_thr
    return f


DETECTORS = {
    '최종 모델 q0.99': lambda f: f['final_pct'] >= 0.99,
    '최종 모델 q0.999': lambda f: f['final_pct'] >= 0.999,
    'std4 q0.99': lambda f: f['std4_pct'] >= 0.99,
    'std4 q0.999': lambda f: f['std4_pct'] >= 0.999,
    'CUSUM(최종) 경보율1%': lambda f: f['cusum_flag'],
    '최종 or std4 q0.999': lambda f: (f['final_pct'] >= 0.999) | (f['std4_pct'] >= 0.999),
}


def detect_table(f, tpl, a):
    inj = f['pos'] >= 0
    rows = []
    for dn, fn in DETECTORS.items():
        hit = fn(f).values & inj.values
        g = pd.DataFrame({'ep': f['ep'].values[inj.values], 'hit': hit[inj.values], 'pos': f['pos'].values[inj.values]})
        ep_hit = g.groupby('ep')['hit'].any()
        first = g[g['hit']].groupby('ep')['pos'].min()
        rows.append({'패턴': tpl or '주입 없음', '크기(a×정상변동)': a, '탐지기': dn,
                     '샷 탐지율': float(g['hit'].mean()), '에피소드 탐지율': float(ep_hit.mean()),
                     '첫 탐지까지 샷(중앙값)': float(first.median()) if len(first) else np.nan,
                     '앞 5샷 탐지율': float(g.loc[g['pos'] < 5, 'hit'].mean()),
                     '뒤 5샷 탐지율': float(g.loc[g['pos'] >= L - 5, 'hit'].mean())})
    return rows


def run(products=('cn7', 'rg3'), n_ep=150, mags=MAGS, seed=0, verbose=True):
    from change_point import cusum
    os.makedirs(OUT, exist_ok=True)
    rows, REAL = [], []
    for p in products:
        ref, lab, model = NC.load(p)
        scale = within_run_scale(ref, model)
        s4 = Std4Scorer().fit(ref)
        cthr = NC.threshold(cusum(NC.probit(ref['pct'].values), NC.segments(ref), 0.5), 0.01)
        wins = pick_windows(ref, n_ep, seed)
        f0 = score_frame(build(ref, wins, None, 0, scale), model, s4, cthr)
        for r in detect_table(f0, None, 0):
            rows.append({'제품': p, **r})
        for tpl in TEMPLATES:
            for a in mags:
                f = score_frame(build(ref, wins, tpl, a, scale, seed), model, s4, cthr)
                for r in detect_table(f, tpl, a):
                    rows.append({'제품': p, '유형': TEMPLATES[tpl]['type'], **r})
            if verbose:
                print(f'  [{p}] {tpl} 완료', flush=True)
        REAL.append(real_defect_magnitude(p, ref, lab, scale))
        pd.DataFrame({'변수': list(scale), '정상 변동(1단위)': list(scale.values())}).assign(제품=p).to_csv(
            f'{OUT}/scale_{p}.csv', index=False, encoding='utf-8-sig')
    T = pd.DataFrame(rows)
    RD = pd.concat(REAL, ignore_index=True)
    T.to_csv(f'{OUT}/detection.csv', index=False, encoding='utf-8-sig')
    RD.to_csv(f'{OUT}/real_defect_magnitude.csv', index=False, encoding='utf-8-sig')
    return {'table': T, 'real': RD, 'mde': mde_table(T)}


def real_defect_magnitude(p, ref, lab, scale, w=20):
    """실제 불량 샷: (값 − 같은 Run 직전 w샷 중앙값) / 정상 변동 → 유형별 중앙값(부호 포함)·|값| 중앙값"""
    ev, t = lab['eval'].values, lab['type'].values
    vars_ = sorted({v for T in TEMPLATES.values() for v in T['vars']})
    rows = []
    for v in vars_:
        c = raw(v)
        if c not in lab.columns:
            continue
        base = lab[c].groupby(lab['run_id'], sort=False).transform(lambda s: s.shift(1).rolling(w, min_periods=5).median())
        dz = ((lab[c] - base) / scale[v]).values
        for ty in ('가스', '미성형', '정상'):
            m = ev & (t == ty) & np.isfinite(dz)
            if m.any():
                rows.append({'제품': p, '변수': v, '유형': ty, '샷 수': int(m.sum()),
                             '편차 중앙값(정상변동 배수)': float(np.median(dz[m])),
                             '|편차| 중앙값': float(np.median(np.abs(dz[m]))),
                             '|편차| 90%': float(np.quantile(np.abs(dz[m]), 0.9))})
    return pd.DataFrame(rows)


def mde_table(T, target=0.5, col='에피소드 탐지율'):
    """패턴·탐지기별 '에피소드 절반 이상 탐지'가 처음 되는 크기 (최소 탐지 크기)"""
    t = T[T['패턴'] != '주입 없음']
    out = []
    for (p, tpl, dn), g in t.groupby(['제품', '패턴', '탐지기'], sort=False):
        g = g.sort_values('크기(a×정상변동)')
        ok = g[g[col] >= target]
        out.append({'제품': p, '패턴': tpl, '탐지기': dn,
                    f'최소 탐지 크기({col}≥{target:.0%})': ok['크기(a×정상변동)'].iloc[0] if len(ok) else '>max'})
    M = pd.DataFrame(out)
    return M.pivot_table(index=['제품', '패턴'], columns='탐지기', values=M.columns[-1], aggfunc='first')


def plot(R, p='cn7', col='에피소드 탐지율', save=None):
    import matplotlib.pyplot as plt
    NC.style()
    T = R['table'][R['table']['제품'] == p]
    base = T[T['패턴'] == '주입 없음'].set_index('탐지기')[col]
    tpls = list(TEMPLATES)
    fig, axes = plt.subplots(1, len(tpls), figsize=(4 * len(tpls), 3.6), sharey=True)
    for ax, tpl in zip(axes, tpls):
        g = T[T['패턴'] == tpl]
        for dn, gg in g.groupby('탐지기', sort=False):
            ln, = ax.plot(gg['크기(a×정상변동)'], gg[col], marker='o', label=dn)
            ax.axhline(base.get(dn, np.nan), color=ln.get_color(), ls=':', lw=0.8)
        ax.set_title(tpl, fontsize=10)
        ax.set_xlabel('주입 크기 (정상 변동의 a배)')
    axes[0].set_ylabel(col + ' (점선 = 주입 없음)')
    axes[-1].legend(fontsize=7)
    fig.suptitle(f'{p.upper()} 합성 불량 주입 — 패턴·크기별 탐지율', x=0.01, ha='left')
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
