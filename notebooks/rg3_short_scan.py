"""
rg3_short_scan.py — 34번: RG3 미성형은 공정 데이터에 '신호 자체가 있는가' (모델을 바꾸기 전 상한 점검)

지금까지
  - 지도(하연님 프레임, SHORT6 변수): 미성형 AP 0.015 ≈ 기준선 0.014, 상위 10%에 0/8
  - 비지도(최종 모델): 무작위 수준
  → 모델·변수를 더 바꿔 보기 전에, 어떤 변수로든 미성형이 정상과 갈리는지부터 확인

RG3 미성형계열 10샷 (11번 결과): RH만 8 (R2 1, R3 4, R4 1, R5 2) / LH만 2 (R6)
  LH·RH는 한 샷에서 같은 공정값을 공유 → 한쪽 캐비티만의 원인이면 공정값으로는 원리상 안 보일 수 있음

하는 일
  A. 미성형 샷 프로필: Run, Run 안 순서, 재가동 후 몇 번째 샷, 직전 정지 시간, 불량 캐비티, 앞뒤 ±3샷의 판정
  B. 변수 은행 ~140개 (모두 라벨 미사용)
     원값 21개 × {수준, Run 중앙값 대비, 직전 20샷 robust 편차, 직전 대비 변화, 직전 10샷 표준편차, 다음 샷 대비 변화}
     + 조합: 금형온도4−3(좌우 금형 차이 후보), 사출시간−충전시간, 흐름저항(사출압/속도)
     + 맥락: Run 안 순서, 재가동 후 샷 수, 직전 정지 시간
     ⚠ 'Run 중앙값 대비'·'다음 샷 대비'는 미래 정보 포함 → 실시간 모델엔 못 씀. '신호가 있나'를 보는 상한 점검용
  C. 변수 하나씩 스캔: AUC, Cliff's δ(=2·AUC−1), 미성형 vs 정상
     - 비교군 ① 같은 평가 대상 정상 전체 / ② 시간 맞춘 정상(같은 Run, 미성형 앞뒤 ±10샷)
     - 순열 검정: 같은 Run 안에서만 라벨을 섞어 2,000번 → 변수별 p + '변수 ~140개 중 최대값' 기준 보정 p(FWER)
       → 보정 p < 0.05인 변수가 하나도 없으면 "이 데이터로는 RG3 미성형 신호를 찾을 수 없다"는 근거
  D. 시차 스캔: 라벨 시각과 공정 기록이 몇 샷 어긋났을 가능성 → 변수를 −5~+5샷 밀어서 같은 스캔
     특정 시차에서만 강하게 나오면 1차·2차 연결(시각 매칭) 문제를 의심
  E. 양성 대조: 같은 방법으로 RG3 가스도 스캔 (가스는 하연님 std4로 어느 정도 잡혔으므로 신호가 나와야 정상)
"""
import numpy as np
import pandas as pd

from final_model import (load_labeled_shots, raw, ALL_RAW, INITIAL, TYPE_GROUPS, SEED, causal_robust_dev,
                         resolution)

STOP_GAP_SEC = 5 * 60
MATCH_K = 10
N_PERM = 2000
LAGS = range(-5, 6)


# ====================================================================== 데이터
def load(p='rg3', base_dir='.'):
    s = load_labeled_shots(p, base_dir).sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    s['eval'] = ((s['split'] != '제외') if 'split' in s.columns else True) & (s['Reason'] != INITIAL)
    s['type'] = np.select([s['Reason'].isin(TYPE_GROUPS['가스']), s['Reason'].isin(TYPE_GROUPS['미성형']),
                           s['Reason'] == INITIAL, s['PassOrFail'].fillna(0) == 0],
                          ['가스', '미성형', '초기허용', '정상'], '기타')
    # 불량 캐비티 (공식 연결표의 행 단위 라벨)
    try:
        l = pd.read_csv(f'{base_dir}/linked/linked_labeled.csv', parse_dates=['TimeStamp'],
                        usecols=['product', 'TimeStamp', 'cavity', 'PassOrFail'])
        l = l[(l['product'] == p) & (l['PassOrFail'] == 1)]
        cav = l.groupby('TimeStamp')['cavity'].agg(lambda c: '+'.join(sorted(set(map(str, c)))))
        s['불량 캐비티'] = s['TimeStamp'].map(cav)
    except Exception as e:                                     # 연결표 형식이 다르면 생략
        print('캐비티 정보 생략:', e)
        s['불량 캐비티'] = np.nan
    g = s.groupby('run_id', sort=False)
    s['seq'] = g.cumcount() + 1
    gap = g['TimeStamp'].diff().dt.total_seconds()
    s['prev_gap_sec'] = gap
    restart = gap.isna() | (gap > STOP_GAP_SEC)
    seg = restart.groupby(s['run_id']).cumsum()
    s['since_restart'] = s.groupby([s['run_id'], seg]).cumcount() + 1
    return s


_MARK = {'정상': '·', '가스': 'G', '미성형': 'S', '초기허용': 'i'}


def _marks(nb, seq0):
    """앞뒤 샷 판정 문자열 (· 정상, G 가스, S 미성형, i 초기허용, [ ] = 해당 샷)"""
    out = []
    for q, t in zip(nb['seq'], nb['type']):
        c = _MARK.get(t, '?')
        out.append(f'[{c}]' if q == seq0 else c)
    return ' '.join(out)


def profile(s, target='미성형', k=3):
    """미성형 샷 한 줄씩: 위치·맥락·앞뒤 판정"""
    rows = []
    for i in np.flatnonzero((s['type'] == target).values):
        r = s.iloc[i]
        same = s['run_id'] == r['run_id']
        nb = s[same & s['seq'].between(r['seq'] - k, r['seq'] + k)]
        rows.append({'TimeStamp': r['TimeStamp'], 'run_id': r['run_id'], 'Reason': r['Reason'],
                     '불량 캐비티': r['불량 캐비티'], 'Run 안 순서': int(r['seq']),
                     'Run 샷 수': int(same.sum()), '재가동 후 샷': int(r['since_restart']),
                     '직전 간격(초)': r['prev_gap_sec'],
                     f'앞뒤 ±{k}샷 판정': _marks(nb, r['seq']),
                     '평가 대상': bool(r['eval'])})
    return pd.DataFrame(rows)


# ====================================================================== 변수 은행
def _by_run(s, x, f):
    out = pd.Series(np.nan, index=s.index)
    for _, idx in s.groupby('run_id', sort=False).groups.items():
        out.loc[idx] = f(np.asarray(x.loc[idx], float))
    return out


def feature_bank(s):
    """{이름: 값} (모두 라벨 미사용). 이름 = 변수|변환"""
    F = {}
    g = s.groupby('run_id', sort=False)

    def add(name, x):
        x = pd.Series(np.asarray(x, float), index=s.index)
        st = resolution(x) if x.notna().sum() > 1 else 1.0
        xg = x.groupby(s['run_id'])
        F[f'{name}|수준'] = x
        F[f'{name}|Run중앙값대비(미래포함)'] = x - xg.transform('median')
        F[f'{name}|편차20'] = _by_run(s, x, lambda a: causal_robust_dev(a, st))
        F[f'{name}|직전변화'] = xg.diff()
        F[f'{name}|직전10샷std'] = xg.transform(lambda v: v.rolling(10, min_periods=3).std())
        F[f'{name}|다음샷변화(미래)'] = -xg.diff(-1)

    for v in ALL_RAW:
        if raw(v) in s.columns:
            add(v, s[raw(v)])
    if raw('Mold_Temperature_4') in s.columns and raw('Mold_Temperature_3') in s.columns:
        add('금형온도4−3', s[raw('Mold_Temperature_4')] - s[raw('Mold_Temperature_3')])
    add('사출시간−충전시간', s[raw('Injection_Time')] - s[raw('Filling_Time')])
    add('흐름저항(사출압/속도)', (s[raw('Max_Injection_Pressure')] / s[raw('Max_Injection_Speed')])
        .replace([np.inf, -np.inf], np.nan))
    F['맥락|Run안순서'] = s['seq'].astype(float)
    F['맥락|재가동후샷'] = s['since_restart'].astype(float)
    F['맥락|직전간격(초)'] = s['prev_gap_sec']
    X = pd.DataFrame(F)
    X = X.loc[:, X.notna().sum() >= 10]
    X = X.loc[:, X.nunique() > 1]
    return X


def shift_within_run(s, X, k):
    """샷 t에 샷 t+k의 값을 붙임 (k>0 = 뒤 샷, k<0 = 앞 샷)"""
    return X.groupby(s['run_id'].values).shift(-k)


# ====================================================================== 스캔
def _ranks(X):
    """결측은 변수 중앙값으로 채운 뒤 평균 순위 (스캔 전용 단순화)"""
    Xf = X.fillna(X.median())
    return Xf.rank(method='average').values


def _auc_from_ranks(R, pos):
    n1 = pos.sum()
    n0 = len(pos) - n1
    return (R[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def _perm_within(groups, y, rng):
    yp = y.copy()
    for idx in groups:
        yp[idx] = rng.permutation(y[idx])
    return yp


def scan(s, X, target='미성형', control='matched', n_perm=N_PERM, seed=SEED, k=MATCH_K):
    """
    control = 'all' (평가 대상 정상 전체) / 'matched' (같은 Run, 대상 샷 ±k샷 안의 정상)
    반환: 변수별 AUC·δ·방향·순열 p·보정 p(FWER) 표 (|δ| 큰 순)
    """
    tgt = (s['type'] == target).values & s['eval'].values
    nrm = (s['type'] == '정상').values & s['eval'].values
    if control == 'matched':
        near = np.zeros(len(s), bool)
        for i in np.flatnonzero(tgt):
            near |= ((s['run_id'] == s['run_id'].iloc[i]) & s['seq'].between(s['seq'].iloc[i] - k,
                                                                           s['seq'].iloc[i] + k)).values
        nrm &= near
    m = tgt | nrm
    Xm, y = X[m].reset_index(drop=True), tgt[m]
    R = _ranks(Xm)
    auc = _auc_from_ranks(R, y)
    eff = np.abs(auc - 0.5)
    runs = s['run_id'].values[m]
    groups = [np.flatnonzero(runs == r) for r in pd.unique(runs)]
    rng = np.random.default_rng(seed)
    perm_eff = np.empty((n_perm, X.shape[1]))
    for b in range(n_perm):
        perm_eff[b] = np.abs(_auc_from_ranks(R, _perm_within(groups, y, rng)) - 0.5)
    p = (1 + (perm_eff >= eff).sum(0)) / (n_perm + 1)
    mx = perm_eff.max(1)
    p_fwer = (1 + (mx[:, None] >= eff[None, :]).sum(0)) / (n_perm + 1)
    out = pd.DataFrame({'변수': [c.split('|')[0] for c in X.columns],
                        '변환': [c.split('|')[1] if '|' in c else '' for c in X.columns],
                        'AUC': auc, "Cliff's δ": 2 * auc - 1,
                        '방향': np.where(auc > 0.5, f'{target}에서 ↑', f'{target}에서 ↓'),
                        '순열 p': p, '보정 p(FWER)': p_fwer,
                        '결측 비율': Xm.isna().mean().values, 'feature': X.columns})
    out = out.sort_values("Cliff's δ", key=np.abs, ascending=False).reset_index(drop=True)
    meta = {'대상': target, '비교군': control, f'{target} 샷': int(y.sum()), '정상 샷': int((~y).sum()),
            '변수 수': X.shape[1], '무작위 최대 |δ| 95% 지점': float(2 * np.quantile(mx, 0.95)),
            '관측 최대 |δ|': float(2 * eff.max()), '보정 p<0.05 변수 수': int((p_fwer < 0.05).sum())}
    return out, meta


def lag_scan(s, X, target='미성형', control='matched', n_perm=500, seed=SEED, lags=LAGS):
    """변수를 −5~+5샷 밀어서 스캔 → 시차별 최대 |δ|와 (시차×변수 전체) 보정 p"""
    cols, mats = [], []
    for k in lags:
        Xk = shift_within_run(s, X, k)
        Xk.columns = [f'{c}@{k:+d}' for c in X.columns]
        mats.append(Xk)
    XL = pd.concat(mats, axis=1)
    tbl, meta = scan(s, XL, target, control, n_perm, seed)
    tbl['시차(샷)'] = tbl['feature'].str.extract(r'@([+-]\d+)$')[0].astype(int)
    tbl['변환'] = tbl['변환'].str.replace(r'@[+-]\d+$', '', regex=True)
    rows = []
    for k, g in tbl.groupby('시차(샷)'):                         # tbl은 |δ| 큰 순 → 그룹 첫 행 = 최대
        rows.append({'시차(샷)': k, '최대 |δ|': abs(g["Cliff's δ"].iloc[0]),
                     '그 변수': g['변수'].iloc[0] + '|' + g['변환'].iloc[0],
                     '최소 보정 p': g['보정 p(FWER)'].min()})
    by_lag = pd.DataFrame(rows)
    return by_lag, tbl, meta


# ====================================================================== 실행
def run(p='rg3', base_dir='.', n_perm=N_PERM, out_dir='./outputs/rg3_short_scan', verbose=True):
    import os
    os.makedirs(out_dir, exist_ok=True)
    s = load(p, base_dir)
    X = feature_bank(s)
    if verbose:
        print(f'[{p}] 샷 {len(s):,} | 평가 대상 유형:', s.loc[s['eval'], 'type'].value_counts().to_dict(),
              f'| 변수 {X.shape[1]}개')
    R = {'shots': s, 'X': X, 'profile': profile(s, '미성형'), 'profile_gas': profile(s, '가스')}
    META = []
    for target in ('미성형', '가스'):
        for control in ('matched', 'all'):
            t, m = scan(s, X, target, control, n_perm)
            R[f'scan_{target}_{control}'] = t
            META.append(m)
            t.to_csv(f'{out_dir}/scan_{target}_{control}.csv', index=False, encoding='utf-8-sig')
            if verbose:
                print(f'  스캔 완료: {target} vs {control}', flush=True)
    R['meta'] = pd.DataFrame(META)
    by_lag, tbl, m = lag_scan(s, X, '미성형', 'matched')
    R['lag'], R['lag_table'], R['lag_meta'] = by_lag, tbl, m
    R['meta'].to_csv(f'{out_dir}/meta.csv', index=False, encoding='utf-8-sig')
    R['profile'].to_csv(f'{out_dir}/profile_short.csv', index=False, encoding='utf-8-sig')
    by_lag.to_csv(f'{out_dir}/lag.csv', index=False, encoding='utf-8-sig')
    return R


# ====================================================================== 그림
def plot_windows(R, target='미성형', top=6, w=15, control='matched', save=None):
    """상위 변수들: 대상 샷을 0으로 맞춘 앞뒤 ±w샷 궤적 (Run 중앙값·IQR로 표준화), 회색 = 정상 무작위 기준점"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    s, X = R['shots'], R['X']
    t = R[f'scan_{target}_{control}']
    feats = list(t['feature'].head(top))
    idx = np.flatnonzero(((s['type'] == target) & s['eval']).values)
    rng = np.random.default_rng(0)
    ctrl = rng.choice(np.flatnonzero(((s['type'] == '정상') & s['eval']).values), size=min(60, len(s)), replace=False)
    fig, axes = plt.subplots(2, (top + 1) // 2, figsize=(5 * ((top + 1) // 2), 7.5), squeeze=False)
    for ax, f in zip(axes.ravel(), feats):
        x = X[f]
        z = (x - x.groupby(s['run_id']).transform('median')) / \
            x.groupby(s['run_id']).transform(lambda v: max(v.quantile(0.75) - v.quantile(0.25), 1e-9))
        for set_, col, lw in ((ctrl, '#c9c8c2', 0.6), (idx, '#1f4e79' if target == '미성형' else '#e67e22', 1.4)):
            for i in set_:
                run, q = s['run_id'].iloc[i], s['seq'].iloc[i]
                win = s[(s['run_id'] == run) & s['seq'].between(q - w, q + w)]
                ax.plot(win['seq'] - q, z.loc[win.index].clip(-10, 10), color=col, lw=lw, alpha=0.8)
        ax.axvline(0, color='k', lw=0.8)
        r = t[t['feature'] == f].iloc[0]
        d, pf = r["Cliff's δ"], r['보정 p(FWER)']
        ax.set_title(f'{f}\nδ={d:+.2f}, 보정 p={pf:.3f}', loc='left', fontsize=8)
        ax.set_xlabel(f'{target} 샷 기준 상대 위치')
    fig.suptitle(f'RG3 {target} 샷 전후 궤적 (진한 선 = {target}, 회색 = 정상 무작위 60샷)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


def plot_null(R, save=None):
    """관측 최대 |δ| vs 무작위(같은 Run 안 라벨 섞기) 최대 |δ| 95% 지점"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    m = R['meta']
    lab = m['대상'] + ' vs ' + m['비교군'].map({'matched': '시간맞춘 정상', 'all': '정상 전체'})
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(m))
    ax.barh(y, m['관측 최대 |δ|'], color=['#1f4e79' if t == '미성형' else '#e67e22' for t in m['대상']], height=0.5)
    ax.scatter(m['무작위 최대 |δ| 95% 지점'], y, color='k', marker='|', s=400, label='무작위 최대값 95% 지점')
    ax.set_yticks(y); ax.set_yticklabels(lab)
    ax.set_xlabel("가장 잘 가르는 변수의 |Cliff's δ|")
    ax.set_title('변수 ~140개 중 최고값이 우연 수준을 넘는가 (막대가 검은 선을 넘어야 신호)', loc='left', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
