"""
search_final.py — 최적 변수 조합 × 비지도 모델 탐색 (선택 과적합을 막는 두 가지 방법)

탐색 범위: compare_final.COMBOS(12) × MODEL_PARAMS(10) = 120개 설정

방법 ① 중첩 LORO (라벨로 고르되 정직하게 평가)
   - 모든 설정은 reference(unlabeled)로만 학습 → labeled 점수는 한 번 계산해 둠
   - 바깥 루프: labeled Run 하나를 빼둠
     안쪽: 나머지 Run의 라벨(② 전체·샷·초기허용 제외)로 AP가 가장 높은 설정 선택
     → 빼둔 Run에 그 설정의 점수를 붙임
   - 모든 Run의 바깥 예측을 모아 AP 계산 = '고르는 과정까지 포함한' 성능
   - 설정마다 점수 척도가 달라서 각 설정의 '정상 기준 백분위(pct)'로 통일
     (pct = reference 정상 샷 중 이보다 낮은 비율 → 모든 설정에서 '정상 오경보율'이라는 같은 의미)

방법 ② 가상 불량 주입 (라벨 없이 고르기)
   - reference 각 Run을 시간 순으로 앞 70% = 학습용 / 뒤 30% = 주입·평가용으로 나눔
     (Run 단위로 나누면 CN7처럼 아주 긴 Run 하나가 통째로 한쪽에 가서 비율이 깨짐)
     파생변수는 Run 전체로 계산 → 뒤쪽 샷도 앞쪽 샷을 기준선으로 쓰고, 학습용 앞쪽 샷은 주입 영향 없음(인과적)
   - 주입용 Run의 '원값'에 물리적으로 그럴듯한 이상을 넣고 → 파생변수를 다시 계산 (인과적 계산 그대로)
   - 학습용으로 학습한 각 설정이 주입한 샷을 얼마나 위로 올리는지(AP) → 유형 평균 AP 1등 선택
   - 선택된 설정을 전체 reference로 다시 학습 → labeled로 '한 번만' 평가
   ⚠ 주입 유형은 공정 원리 + 12번 물리 검증(라벨 참고)에서 나온 가설 → 완전히 라벨 무관하지는 않음

참고: 120개 전체의 labeled 성능 표(사후 비교, 선택 과적합 있음)
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from final_model import (raw, EVENT_VARS, SEED, REF_MAX, INITIAL, TYPE_GROUPS, resolution,
                         load_labeled_shots, load_official_rows, build_reference, load_scales, make_features,
                         eval_frame1, eval_frame2, recall_at_k_by_run)
try:
    from plot_style import set_style
except ImportError:                                              # plot_style.py 없을 때
    def set_style():
        pass
from compare_final import Detector, COMBOS, MODEL_PARAMS, add_delta, _paired_delta, BASE_COMBO, BASE_MODEL

BASE_CFG = (BASE_COMBO, BASE_MODEL)
HOLD_FRAC = 0.30          # 각 Run의 뒤쪽 30% 샷 = 주입·평가용
MIN_RUN_HOLD = 60         # 60샷 이상 Run만 나눔 (짧은 Run은 전부 학습용)
INJ_RATE = 0.03           # 주입·평가 구간 샷 대비 주입 비율
INJ_RATE_RAMP = 0.01      # 열드리프트는 10샷 연속이라 시작점 비율을 낮춤 (양성 비율 ≈ 10%)
INJ_MIN_SEQ = 25          # Run 시작 후 25샷 이후에만 주입 (기준선 확보)
INJ_GAP = 30              # 주입 샷 간 최소 간격
INJ_K = (2.0, 6.0)        # 크기: Run 내 정상 퍼짐(robust σ)의 2~6배
RAMP_LEN = 10             # 열 드리프트 길이 (샷)
AFTER_EXCL = 5            # 주입 직후 5샷은 음성에서 제외 (기준선 영향)

# 주입 유형: {원변수: 방향(부호 × 상대 크기)}
INJ_TYPES = {
    '미성형형(충전↓·속도↑·쿠션↓)': {'Filling_Time': -1, 'Injection_Time': -1, 'Max_Injection_Speed': +1,
                                  'Cushion_Position': -1},
    '쿠션부족(재료 부족)':         {'Cushion_Position': -1, 'Max_Injection_Pressure': +0.5},
    '계량지연':                    {'Plasticizing_Time': +1, 'Average_Back_Pressure': +0.5},
    '점도상승(압력↑·속도↓)':       {'Max_Injection_Pressure': +1, 'Max_Switch_Over_Pressure': +1,
                                  'Max_Injection_Speed': -0.5},
    '열드리프트(10샷 상승)':        {'Mold_Temperature_3': +1, 'Mold_Temperature_4': +1, 'Hopper_Temperature': +0.5},
    '단일 변수 튐(무작위)':         None,
}
RAMP_TYPES = {'열드리프트(10샷 상승)'}


def all_configs(combos=COMBOS, models=MODEL_PARAMS):
    cfgs = [BASE_CFG] + [(c, m) for c in combos for m in models if (c, m) != BASE_CFG]
    return cfgs


def cfg_name(cfg):
    return f'{cfg[0]} | {cfg[1]}'


# ====================================================================== 데이터
def _split_time(ref_shots, frac=HOLD_FRAC):
    """반환: (Run·시간 순 정렬된 reference, 주입·평가용 마스크)"""
    d = ref_shots.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    seq = d.groupby('run_id').cumcount()
    n = d.groupby('run_id')['run_id'].transform('size')
    hold = (n >= MIN_RUN_HOLD) & (seq >= np.ceil((1 - frac) * n))
    return d, hold.values


def _feat(shots, steps, fr_step):
    return add_delta(make_features(shots, steps, fr_step), steps)


def _sample(f):
    return f.sample(REF_MAX, random_state=SEED).reset_index(drop=True) if len(f) > REF_MAX else f


def _robust_sigma(shots, steps):
    """변수별 Run 내 정상 퍼짐 = Run별 1.4826·MAD의 중앙값 (하한 = 기록 단위)"""
    out = {}
    for v in set(sum([list(t) for t in INJ_TYPES.values() if t], [])) | set(EVENT_VARS):
        c = raw(v)
        if c not in shots.columns:
            continue
        mad = shots.groupby('run_id')[c].apply(lambda s: 1.4826 * np.median(np.abs(s - np.median(s))))
        st = steps.get(v, resolution(shots[c]))
        out[v] = (max(float(np.nanmedian(mad)), st), st)
    return out


def inject(d, hold, tname, sig, steps, fr_step, seed=SEED):
    """
    d: Run·시간 순 reference 원값, hold: 주입·평가용 마스크
    뒤쪽(hold) 구간에만 한 유형 주입 → Run 전체로 파생변수 재계산
    반환: hold 구간 features + inj(양성), eval(평가 대상)
    """
    rng = np.random.default_rng(seed)
    d = d.copy()
    d['inj'] = 0
    d['excl'] = 0
    spec = INJ_TYPES[tname]
    ramp_type = tname in RAMP_TYPES
    rate = INJ_RATE_RAMP if ramp_type else INJ_RATE
    for r, idx in d.groupby('run_id').indices.items():
        idx = np.sort(idx)
        n = len(idx)
        hpos = [i for i in range(n) if hold[idx[i]] and i <= n - 1 - (RAMP_LEN if ramp_type else 0)]
        if not hpos:
            continue
        rng.shuffle(hpos)
        chosen = []
        for i in hpos:
            if len(chosen) >= max(1, int(rate * int(hold[idx].sum()))):
                break
            if all(abs(i - j) >= INJ_GAP for j in chosen):
                chosen.append(i)
        for i in chosen:
            k = rng.uniform(*INJ_K)
            if spec is None:                                  # 단일 변수 튐
                v = rng.choice(EVENT_VARS)
                sp = {v: rng.choice([-1, 1])}
            else:
                sp = spec
            if ramp_type:
                rows = idx[i:i + RAMP_LEN]
                ramp = np.arange(1, RAMP_LEN + 1) / RAMP_LEN
            else:
                rows = idx[[i]]
                ramp = np.array([1.0])
            for v, w in sp.items():
                if v not in sig:
                    continue
                s_, st = sig[v]
                c = raw(v)
                newv = d.loc[rows, c].values + w * k * s_ * ramp
                d.loc[rows, c] = np.round(newv / st) * st if st > 0 else newv
            d.loc[rows, 'inj'] = 1
            after = idx[i + len(rows): i + len(rows) + AFTER_EXCL]
            d.loc[after, 'excl'] = 1
    hold_ts = set(d.loc[hold, 'TimeStamp'])
    f = _feat(d.drop(columns=['inj', 'excl']), steps, fr_step)
    f = f[f['TimeStamp'].isin(hold_ts)].reset_index(drop=True)
    key = d.set_index('TimeStamp')
    f['inj'] = f['TimeStamp'].map(key['inj']).values
    f['eval'] = (f['inj'] == 1) | (f['TimeStamp'].map(key['excl']).values == 0)
    return f


def prepare_search(p, base_dir='.', seed=SEED):
    shots = load_labeled_shots(p, base_dir)
    ref_shots, _ = build_reference(p, shots, base_dir)
    steps, fr_step, _ = load_scales(p, ref_shots, base_dir)
    d, hold = _split_time(ref_shots)
    sig = _robust_sigma(d[~hold], steps)
    f_full = _feat(d, steps, fr_step)                       # 주입 없는 Run 전체 계산
    hold_ts = set(d.loc[hold, 'TimeStamp'])
    D = {'ref_full': _sample(f_full),
         'ref_tr': _sample(f_full[~f_full['TimeStamp'].isin(hold_ts)].reset_index(drop=True)),
         'lab': _feat(shots, steps, fr_step),
         'rows': load_official_rows(p, base_dir),
         'inj': {t: inject(d, hold, t, sig, steps, fr_step, seed) for t in INJ_TYPES},
         'sigma': sig}
    D['m2'] = ((D['lab']['split'] != '제외') & (D['lab']['Reason'] != INITIAL)).values
    n_hold_runs = d.loc[hold, 'run_id'].nunique()
    print(f'[{p}] reference 전체 {len(d):,}샷 → 학습용 {len(D["ref_tr"]):,} ({(~hold).mean():.0%}) / '
          f'주입·평가용 {int(hold.sum()):,} ({hold.mean():.0%}, Run {n_hold_runs}개의 뒤쪽 30%)')
    print('     유형별 주입 샷 (양성 비율): ' + ', '.join(
        f'{t.split("(")[0]} {int(v["inj"].sum())} ({v.loc[v["eval"], "inj"].mean():.0%})' for t, v in D['inj'].items()))
    return D


# ====================================================================== 점수 계산 (설정 120개)
def _zref(s, s_ref):
    med = np.median(s_ref)
    iqr = np.subtract(*np.percentile(s_ref, [75, 25]))
    return (s - med) / (iqr if iqr > 1e-12 else np.std(s_ref) + 1e-12)


def score_all(D, cfgs, verbose=True):
    """
    반환: {'lab': {cfg: z_ref 점수}, 'pct': {cfg: 정상 기준 백분위(+동점 깨기)}, 'syn': {cfg: {유형: AP}}}
    pct는 reference 최댓값을 넘는 샷끼리 동점이 되므로 z로 아주 작게 순서를 매김 (1e-6 < 백분위 1칸)
    """
    LAB, PCT, SYN = {}, {}, {}
    t0 = time.time()
    for i, cfg in enumerate(cfgs):
        cols = COMBOS[cfg[0]]
        # (a) 전체 reference 학습 → labeled 점수
        det = Detector(cfg[1], cols).fit(D['ref_full'])
        s_lab = det.score(D['lab'])
        LAB[cfg] = _zref(s_lab, det.s_ref)
        PCT[cfg] = det.pct(s_lab) + 1e-6 * np.tanh(LAB[cfg] / 10)
        # (b) 학습용 reference 학습 → 주입 데이터 탐지 AP
        det2 = Detector(cfg[1], cols).fit(D['ref_tr'])
        SYN[cfg] = {}
        for t, f in D['inj'].items():
            m = f['eval'].values
            SYN[cfg][t] = average_precision_score(f['inj'].values[m], det2.score(f)[m])
        if verbose and (i + 1) % 10 == 0:
            print(f'    {i + 1}/{len(cfgs)} 설정 완료 ({time.time() - t0:.0f}초)')
    return {'lab': LAB, 'pct': PCT, 'syn': SYN, 'version': 2}


def run_scores(DATA, cfgs=None, cache='./outputs/search_final/scores_v2.pkl', verbose=True):
    """제품별 점수 계산 (캐시 있으면 불러옴)"""
    cfgs = cfgs or all_configs()
    if cache and os.path.exists(cache):
        S = pickle.load(open(cache, 'rb'))
        if all(S.get(p, {}).get('version') == 2 and set(cfgs) <= set(S[p]['lab']) for p in DATA):
            print('캐시 사용:', cache)
            return S
    S = {}
    for p, D in DATA.items():
        print(f'[{p}] 설정 {len(cfgs)}개 점수 계산')
        S[p] = score_all(D, cfgs, verbose)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump(S, open(cache, 'wb'))
    return S


# ====================================================================== 평가 공통
def _ap_quick(y, s):
    return average_precision_score(y, s) if y.sum() else np.nan


def grid_table(DATA, S):
    """120개 전체: labeled ①·② AP + 가상 불량 평균 AP (참고: 사후 비교)"""
    rows = []
    for p, D in DATA.items():
        lab, m2, off = D['lab'], D['m2'], D['rows']
        y2 = lab['PassOrFail'].values[m2].astype(int)
        y1 = off['PassOrFail'].values.astype(int)
        for cfg, z in S[p]['lab'].items():
            shot = pd.Series(z, index=lab['TimeStamp'].values)
            syn = S[p]['syn'][cfg]
            r = {'제품': p, '변수 조합': cfg[0], '모델': cfg[1], '가상불량 평균AP': np.mean(list(syn.values())),
                 '①AP': _ap_quick(y1, off['TimeStamp'].map(shot).fillna(z.min()).values),
                 '②AP': _ap_quick(y2, z[m2])}
            for name, rs in TYPE_GROUPS.items():
                pm = lab['Reason'].isin(rs).values[m2]
                ev = pm | (y2 == 0)
                r[f'②AP_{name}'] = _ap_quick(pm[ev].astype(int), z[m2][ev])
            r.update({f'가상_{t.split("(")[0]}': v for t, v in syn.items()})
            rows.append(r)
    g = pd.DataFrame(rows)
    g['순위(가상)'] = g.groupby('제품')['가상불량 평균AP'].rank(ascending=False, method='min').astype(int)
    g['순위(②AP)'] = g.groupby('제품')['②AP'].rank(ascending=False, method='min').astype(int)
    return g


def full_eval(p, D, z, name):
    """선택된 설정 하나를 두 틀로 정식 평가"""
    lab, m2 = D['lab'], D['m2']
    shot = pd.Series(z, index=lab['TimeStamp'].values)
    r1 = eval_frame1(p, D['rows'], shot)
    g = pd.DataFrame({'run_id': lab['run_id'].values[m2], 'y': lab['PassOrFail'].values[m2].astype(int),
                      'Reason': lab['Reason'].values[m2], 'score': z[m2]})
    r2 = eval_frame2(p, g)
    return {'제품': p, '설정': name, '①AP': r1['AP'], '①유의': r1['유의'], '①상위10%': r1['상위10%'],
            '②AP': r2['AP'], '②AP_lo': r2['AP_lo'], '②AP_hi': r2['AP_hi'], '②Run별 상위10%': r2['Run별 상위10%'],
            '②AP_가스': r2.get('AP_가스', np.nan), '②AP_미성형': r2.get('AP_미성형', np.nan)}


def _delta_vs_base(p, D, z, zb):
    lab, m2, off = D['lab'], D['m2'], D['rows']
    a1 = off['TimeStamp'].map(pd.Series(z, index=lab['TimeStamp'].values)).fillna(z.min()).values
    b1 = off['TimeStamp'].map(pd.Series(zb, index=lab['TimeStamp'].values)).fillna(zb.min()).values
    o1, l1, h1, v1 = _paired_delta(off['PassOrFail'].values.astype(int), a1, b1, cluster=off['TimeStamp'].values)
    o2, l2, h2, v2 = _paired_delta(lab['PassOrFail'].values[m2].astype(int), z[m2], zb[m2])
    return {'①ΔAP(vs 최종)': o1, '①Δ CI': f'[{l1:.3f}, {h1:.3f}]', '①판정': v1,
            '②ΔAP(vs 최종)': o2, '②Δ CI': f'[{l2:.3f}, {h2:.3f}]', '②판정': v2}


# ====================================================================== 방법 ② 가상 불량 주입으로 선택
def select_by_synthetic(DATA, S, top=5):
    """제품별 가상불량 평균 AP 상위 설정 → labeled 한 번 평가 + 최종 모델 대비 차이"""
    rows = []
    for p, D in DATA.items():
        syn = pd.Series({cfg: np.mean(list(v.values())) for cfg, v in S[p]['syn'].items()}).sort_values(ascending=False)
        zb = S[p]['lab'][BASE_CFG]
        cand = list(syn.index[:top])
        if BASE_CFG not in cand:
            cand.append(BASE_CFG)
        for rank, cfg in enumerate(cand, 1):
            r = full_eval(p, D, S[p]['lab'][cfg], cfg_name(cfg))
            r.update({'가상 순위': int((syn > syn[cfg]).sum()) + 1, '가상불량 평균AP': syn[cfg]})
            if cfg != BASE_CFG:
                r.update(_delta_vs_base(p, D, S[p]['lab'][cfg], zb))
            else:
                r.update({'①판정': '(최종 모델)', '②판정': '(최종 모델)'})
            rows.append(r)
    return pd.DataFrame(rows)


def synthetic_validity(G):
    """가상 불량 성능과 실제 labeled 성능이 같이 움직이나 (설정 120개 Spearman)"""
    rows = []
    for p, g in G.groupby('제품'):
        for c in ['①AP', '②AP', '②AP_미성형', '②AP_가스']:
            ok = g[['가상불량 평균AP', c]].dropna()
            rho, pv = spearmanr(ok['가상불량 평균AP'], ok[c])
            rows.append({'제품': p, '실제 지표': c, 'Spearman ρ': rho, 'p': pv})
    return pd.DataFrame(rows)


# ====================================================================== 방법 ① 중첩 LORO
def nested_loro(DATA, S):
    """
    바깥: labeled Run 하나 빼기 / 안쪽: 나머지 Run ② AP 최고 설정 선택 (동점이면 최종 모델 우선)
    반환: (요약표, Run별 선택표)
    """
    summ, picks = [], []
    for p, D in DATA.items():
        lab, m2 = D['lab'], D['m2']
        run = lab['run_id'].values[m2]
        y = lab['PassOrFail'].values[m2].astype(int)
        reason = lab['Reason'].values[m2]
        Z = {cfg: z[m2] for cfg, z in S[p]['pct'].items()}       # 설정 간 공통 척도 = 정상 기준 백분위
        cfgs = list(Z)                                   # 첫 번째 = 최종 모델
        outer = np.zeros(len(y))
        for r in pd.unique(run):
            te = run == r
            tr = ~te
            if y[tr].sum() == 0:
                best = BASE_CFG
                aps = {BASE_CFG: np.nan}
            else:
                aps = {cfg: average_precision_score(y[tr], Z[cfg][tr]) for cfg in cfgs}
                best = max(cfgs, key=lambda c: (round(aps[c], 10), c == BASE_CFG))
            outer[te] = Z[best][te]
            picks.append({'제품': p, '빼둔 Run': r, '빼둔 Run 불량': int(y[te].sum()), '안쪽 불량': int(y[tr].sum()),
                          '선택 설정': cfg_name(best), '안쪽 AP(선택)': aps[best],
                          '안쪽 AP(최종 모델)': aps.get(BASE_CFG, np.nan)})
        zb = Z[BASE_CFG]
        o, lo, hi, v = _paired_delta(y, outer, zb)
        g_o = pd.DataFrame({'run_id': run, 'y': y, 'Reason': reason, 'score': outer})
        g_b = g_o.assign(score=zb)
        row = {'제품': p, '바깥 AP(탐색 포함)': average_precision_score(y, outer),
               '최종 모델 AP(고정)': average_precision_score(y, zb), 'ΔAP': o, 'Δ CI': f'[{lo:.3f}, {hi:.3f}]',
               '판정': v.replace('기준', '최종 모델'),
               'Run별 상위10%(탐색)': '%d/%d' % recall_at_k_by_run(g_o, 0.10, g_o['y'] == 1),
               'Run별 상위10%(최종)': '%d/%d' % recall_at_k_by_run(g_b, 0.10, g_b['y'] == 1),
               '선택된 서로 다른 설정 수': len({q['선택 설정'] for q in picks if q['제품'] == p})}
        summ.append(row)
    return pd.DataFrame(summ), pd.DataFrame(picks)


# ====================================================================== 그림
def plot_grid(G, product, value='②AP', save=None):
    """변수 조합 × 모델 히트맵"""
    set_style()
    g = G[G['제품'] == product].pivot(index='변수 조합', columns='모델', values=value)
    g = g.loc[list(COMBOS), list(MODEL_PARAMS)]
    fig, ax = plt.subplots(figsize=(1.0 * g.shape[1] + 3, 0.42 * g.shape[0] + 1.5))
    im = ax.imshow(g.values, cmap='Blues', aspect='auto')
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if v > np.nanmax(g.values) * 0.6 else 'black')
    bi, bj = list(COMBOS).index(BASE_COMBO), list(MODEL_PARAMS).index(BASE_MODEL)
    ax.add_patch(plt.Rectangle((bj - .5, bi - .5), 1, 1, fill=False, ec='red', lw=2))
    ax.set_xticks(range(g.shape[1])); ax.set_xticklabels(g.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(g.shape[0])); ax.set_yticklabels(g.index, fontsize=8)
    ax.set_title(f'{product.upper()} — {value} (빨간 칸 = 최종 모델)', loc='left', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


def plot_syn_vs_real(G, product, save=None):
    set_style()
    g = G[G['제품'] == product]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, c in zip(ax, ['①AP', '②AP']):
        a.scatter(g['가상불량 평균AP'], g[c], s=14, alpha=.6)
        b = g[(g['변수 조합'] == BASE_COMBO) & (g['모델'] == BASE_MODEL)]
        a.scatter(b['가상불량 평균AP'], b[c], s=60, c='red', label='최종 모델')
        a.set_xlabel('가상 불량 평균 AP (라벨 없이 계산)'); a.set_ylabel(f'실제 labeled {c}')
        a.legend(fontsize=8)
    fig.suptitle(f'{product.upper()} — 가상 불량 성능 vs 실제 성능 (점 = 설정 120개)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
