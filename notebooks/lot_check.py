"""
lot_check.py — 생산계획일(PART_FACT_PLAN_DATE)을 로트로 써도 되는지 점검 (CN7·RG3)

입력
  base/base_rows.csv                     00_base_labeled 출력 (행 단위, plan_date·run_id·split 포함)
  outputs/final_model/{p}_scores.csv     (선택) 최종 모델 점수 → 로트별 경보율
  1차 labeled_data.csv                   (선택) PART_FACT_PLAN_DATE 원본 형식·시리얼 확인

점검 항목
  A1 날짜 경계 규칙   : 'h시 이후 생산 = 다음 날 작업분' 하나로 모든 샷이 설명되나
  A2 로트 간 겹침     : 같은 설비에서 두 로트가 번갈아 생산되지 않나 (CN7·RG3 함께)
  A3 경계 = 생산 정지 : 로트가 바뀌는 지점에 60분 넘는 공백이 있나
  B  Run ↔ 로트       : Run 하나가 두 로트에 걸치지 않나
  C  공정 동질성      : 로트 안 Run끼리 조건(특히 금형·호퍼 온도)이 비슷한가
  D  품질 (참고)      : 로트마다 불량률이 다른가 (순열검정)
  E  최종 모델 (참고) : 로트마다 경보율이 기대(1−q)와 비슷한가

라벨은 D에서만 쓴다. 코드는 결과를 보기 전에 기준(60분, η² 0.5, q 0.99)을 고정했다.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from plot_style import set_style, short, INK, SUB, MUTED, ACCENT

PRODUCTS = ('cn7', 'rg3')
RUN_GAP_MIN = 60                     # 00·final_model과 같은 Run 기준
INITIAL = '초기허용불량'
STATE_VARS = ['Mold_Temperature_3', 'Hopper_Temperature', 'Plasticizing_Position']
EVENT_VARS = ['Max_Injection_Speed', 'Filling_Time', 'Max_Injection_Pressure',
              'Cushion_Position', 'Average_Back_Pressure', 'Barrel_Temperature_4']
KEY_VARS = STATE_VARS + EVENT_VARS
ETA_LIMIT = 0.5                      # 로트 안 Run 간 η² 이 이상이면 '조건이 갈림'
Q_ALARM = 0.99                       # 최종 모델 경보 기준 (기대 경보율 1%)
N_PERM, SEED = 5000, 42
raw = lambda v: f'{v}_raw'


# ====================================================================== 1. 데이터
def load_lot_shots(base_dir='.', model_dir='./outputs/final_model', products=PRODUCTS):
    """base_rows → 샷 1행. lot = 제품_생산계획일. 반환: (샷 df, LH/RH 계획일 불일치 샷 수)"""
    R = pd.read_csv(f'{base_dir}/base/base_rows.csv', parse_dates=['TimeStamp', 'plan_date'], low_memory=False)
    R = R[R['product'].isin(products) & R['split'].isin(['2차', '추가'])].copy()   # 이상로그·10샷 미만 Run 제외
    R['plan_date'] = R['plan_date'].dt.normalize()
    R['Reason'] = R['Reason'].fillna('정상')

    g = R.groupby('shot_key')
    n_mismatch = int((g['plan_date'].nunique() > 1).sum())          # 0이어야 정상
    agg = {'product': 'first', 'TimeStamp': 'first', 'plan_date': 'first',
           'run_id': 'first', 'split': 'first', 'PassOrFail': 'max'}
    agg.update({raw(v): 'first' for v in KEY_VARS if raw(v) in R.columns})
    S = g.agg(agg)
    S['initial'] = g['Reason'].agg(lambda s: (s == INITIAL).any())
    S['y'] = ((S['PassOrFail'] == 1) & ~S['initial']).astype(int)    # 공정 불량 (초기허용 제외)
    S = S.reset_index().sort_values(['product', 'TimeStamp']).reset_index(drop=True)
    S['lot'] = S['product'] + '_' + S['plan_date'].dt.strftime('%m-%d')

    # (선택) 최종 모델 점수
    parts = []
    for p in products:
        f = f'{model_dir}/{p}_scores.csv'
        if os.path.exists(f):
            parts.append(pd.read_csv(f, parse_dates=['TimeStamp'], usecols=['TimeStamp', 'score', 'pct'])
                           .assign(product=p))
    if parts:
        S = S.merge(pd.concat(parts, ignore_index=True), on=['product', 'TimeStamp'], how='left')
    return S, n_mismatch


# ====================================================================== 2. 로트 요약
def lot_summary(S):
    rows = []
    for lot, d in S.groupby('lot', sort=False):
        d = d.sort_values('TimeStamp')
        gap = d['TimeStamp'].diff().dt.total_seconds() / 60
        rows.append({'product': d['product'].iloc[0], 'lot': lot,
                     '시작': d['TimeStamp'].min(), '끝': d['TimeStamp'].max(),
                     '길이(시간)': (d['TimeStamp'].max() - d['TimeStamp'].min()).total_seconds() / 3600,
                     '샷': len(d), 'Run 수': d['run_id'].nunique(),
                     'Run': ', '.join(sorted(d['run_id'].unique())),
                     '내부 최대 공백(분)': gap.max(),
                     f'{RUN_GAP_MIN}분 초과 공백': int((gap > RUN_GAP_MIN).sum()),
                     '공정불량': int(d['y'].sum()), '불량률': d['y'].mean(), '초기허용': int(d['initial'].sum())})
    return pd.DataFrame(rows)


# ====================================================================== A1. 날짜 경계 규칙
def offset_table(S):
    """생산계획일 − 실제 날짜(일)별 생산 시각 범위"""
    off = (S['plan_date'] - S['TimeStamp'].dt.normalize()).dt.days
    return (S.assign(차이_일=off, 시=S['TimeStamp'].dt.hour)
             .groupby(['product', '차이_일'])['시'].agg(샷='size', 최소='min', 최대='max').reset_index())


def cutoff_rule(S):
    """규칙: h시 이후 생산 → 다음 날 작업분. h = 1~24 (24 = 전부 같은 날)"""
    base = S['TimeStamp'].dt.normalize()
    hour = S['TimeStamp'].dt.hour + S['TimeStamp'].dt.minute / 60
    rows = []
    for h in range(1, 25):
        pred = base + pd.to_timedelta((hour >= h).astype(int), unit='D')
        rows.append({'기준 시각(시)': h, '일치율': (pred == S['plan_date']).mean(),
                     '불일치 샷': int((pred != S['plan_date']).sum())})
    return pd.DataFrame(rows)


def best_cutoff(S):
    c = cutoff_rule(S)
    best = c['일치율'].max()
    hs = c.loc[c['일치율'] == best, '기준 시각(시)']
    return best, int(hs.min()), int(hs.max())


# ====================================================================== A2. 로트 간 시간 겹침
def lot_overlap(LOT):
    L = LOT.sort_values('시작').reset_index(drop=True)
    rows = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            a, b = L.loc[i], L.loc[j]
            ov = min(a['끝'], b['끝']) - max(a['시작'], b['시작'])
            if ov > pd.Timedelta(0):
                rows.append({'lot A': a['lot'], 'lot B': b['lot'], '겹침(분)': ov.total_seconds() / 60})
    return pd.DataFrame(rows, columns=['lot A', 'lot B', '겹침(분)'])


def lot_switches(S):
    """설비 전체 시간순으로 로트가 바뀐 횟수 (겹침이 없으면 로트 수 − 1)"""
    a = S.sort_values('TimeStamp')['lot']
    return int((a != a.shift()).sum() - 1), S['lot'].nunique() - 1


# ====================================================================== A3. 로트 경계 = 생산 공백?
def boundary_table(S):
    rows = []
    for p, d in S.groupby('product'):
        d = d.sort_values('TimeStamp').reset_index(drop=True)
        gap = d['TimeStamp'].diff().dt.total_seconds() / 60
        for i in np.flatnonzero((d['lot'] != d['lot'].shift()).values)[1:]:
            rows.append({'product': p, '이전 lot': d.loc[i - 1, 'lot'], '다음 lot': d.loc[i, 'lot'],
                         '이전 마지막 샷': d.loc[i - 1, 'TimeStamp'], '다음 첫 샷': d.loc[i, 'TimeStamp'],
                         '공백(분)': gap.iloc[i], 'Run도 바뀜': d.loc[i, 'run_id'] != d.loc[i - 1, 'run_id']})
    return pd.DataFrame(rows, columns=['product', '이전 lot', '다음 lot', '이전 마지막 샷', '다음 첫 샷',
                                       '공백(분)', 'Run도 바뀜'])


# ====================================================================== B. Run ↔ 로트
def run_lot_table(S):
    CT = pd.crosstab(S['run_id'], S['lot'])
    run_purity = CT.max(axis=1) / CT.sum(axis=1)                 # 1 = Run 전체가 한 로트 안
    lot_runs = (CT > 0).sum(axis=0)                               # 로트 안 Run 수
    return CT, run_purity, lot_runs


# ====================================================================== C. 공정 동질성
def eta2(x, grp):
    x = pd.Series(x, dtype=float)
    grp = pd.Series(np.asarray(grp), index=x.index)
    m = x.notna()
    x, grp = x[m], grp[m]
    tot = ((x - x.mean()) ** 2).sum()
    if tot == 0:
        return np.nan
    return ((x.groupby(grp).transform('mean') - x.mean()) ** 2).sum() / tot


def eta_table(S, vars_=KEY_VARS):
    """변동 중 로트 평균 차이 / Run 평균 차이가 차지하는 비율"""
    rows = []
    for p, d in S.groupby('product'):
        for v in vars_:
            if raw(v) not in d:
                continue
            rows.append({'product': p, '변수': short(v), '구분': '상태' if v in STATE_VARS else '이벤트',
                         'η² (로트)': eta2(d[raw(v)], d['lot']), 'η² (Run)': eta2(d[raw(v)], d['run_id'])})
    return pd.DataFrame(rows)


def within_lot_runs(S, vars_=KEY_VARS):
    """Run이 2개 이상인 로트: 로트 안 Run끼리 조건이 다른가"""
    rows = []
    for lot, d in S.groupby('lot'):
        if d['run_id'].nunique() < 2:
            continue
        for v in vars_:
            if raw(v) not in d:
                continue
            groups = [x[raw(v)].dropna().values for _, x in d.groupby('run_id')]
            try:
                pk = stats.kruskal(*groups).pvalue
            except ValueError:                                    # 값이 전부 같으면
                pk = np.nan
            rows.append({'lot': lot, '변수': short(v), '구분': '상태' if v in STATE_VARS else '이벤트',
                         'Run별 평균': d.groupby('run_id')[raw(v)].mean().round(2).to_dict(),
                         'Run 간 η² (로트 안)': eta2(d[raw(v)], d['run_id']), 'Kruskal p': pk})
    return pd.DataFrame(rows, columns=['lot', '변수', '구분', 'Run별 평균', 'Run 간 η² (로트 안)', 'Kruskal p'])


# ====================================================================== D. 품질: 그룹 간 불량률 차이
def group_rate_test(d, col, n_perm=N_PERM, seed=SEED):
    """카이제곱 통계량을 라벨 순열(같은 제품 안)로 검정 → 불량 수가 적어도 쓸 수 있음"""
    y = d['y'].values.astype(float)
    codes, uniq = pd.factorize(d[col])
    if y.sum() == 0 or len(uniq) < 2:
        return {'그룹 수': len(uniq), '카이제곱': np.nan, 'p(순열)': np.nan, '불량 최다 그룹 비중': np.nan}
    n = np.bincount(codes, minlength=len(uniq)).astype(float)

    def chi2(yv):
        k = np.bincount(codes, weights=yv, minlength=len(uniq))
        pr = yv.mean()
        e1, e0 = n * pr, n * (1 - pr)
        return (((k - e1) ** 2) / e1 + (((n - k) - e0) ** 2) / e0).sum()

    obs = chi2(y)
    rng = np.random.default_rng(seed)
    null = np.array([chi2(rng.permutation(y)) for _ in range(n_perm)])
    k = np.bincount(codes, weights=y, minlength=len(uniq))
    return {'그룹 수': len(uniq), '카이제곱': obs, 'p(순열)': (np.sum(null >= obs) + 1) / (n_perm + 1),
            '불량 최다 그룹 비중': k.max() / k.sum()}


def quality_table(S):
    rows = []
    for p, d in S.groupby('product'):
        for col in ['lot', 'run_id']:
            rows.append({'product': p, '단위': col, '샷': len(d), '불량': int(d['y'].sum()),
                         **group_rate_test(d, col)})
    return pd.DataFrame(rows)


# ====================================================================== E. 최종 모델 경보율 (참고)
def alarm_by_lot(S, q=Q_ALARM):
    """양품 샷만: 로트별 pct ≥ q 비율. 기대값 1−q에서 크게 벗어나면 로트 단위 수준 이동"""
    if 'pct' not in S:
        return pd.DataFrame()
    g = S[(S['y'] == 0) & ~S['initial'] & S['pct'].notna()]
    t = g.groupby('lot').agg(양품=('pct', 'size'), 경보=('pct', lambda x: int((x >= q).sum())),
                             pct_중앙값=('pct', 'median'))
    t['경보율'] = t['경보'] / t['양품']
    t['기대'] = 1 - q
    # 이항검정: 경보율이 기대와 다른가
    t['이항 p'] = [stats.binomtest(int(a), int(n), 1 - q).pvalue for a, n in zip(t['경보'], t['양품'])]
    return t.reset_index()


# ====================================================================== 종합 판정
def verdict(S, LOT, OV, BD, run_purity, WITHIN, QT):
    best, h0, h1 = best_cutoff(S)
    sw, sw_ok = lot_switches(S)
    n_cont = int((BD['공백(분)'] <= RUN_GAP_MIN).sum())
    st = WITHIN[WITHIN['구분'] == '상태']
    n_split = int((st['Run 간 η² (로트 안)'] >= ETA_LIMIT).sum())
    rows = [
        {'기준': 'A1 날짜 경계가 한 규칙으로 설명됨', '결과': f'일치율 {best:.4f} (기준 {h0}~{h1}시)',
         '판정': '통과' if best == 1 else '확인'},
        {'기준': 'A2 로트끼리 시간 겹침 없음 (설비 공통)', '결과': f'겹침 {len(OV)}쌍, 전환 {sw}회 (겹침 없으면 {sw_ok}회)',
         '판정': '통과' if len(OV) == 0 else '실패'},
        {'기준': f'A3 로트 경계에 {RUN_GAP_MIN}분 넘는 정지', '결과': f'경계 {len(BD)}개 중 연속 생산 중 경계 {n_cont}개',
         '판정': '통과' if n_cont == 0 else '확인'},
        {'기준': 'B Run이 한 로트 안에만 있음', '결과': f'두 로트에 걸친 Run {int((run_purity < 1).sum())}개',
         '판정': '통과' if (run_purity == 1).all() else '확인'},
        {'기준': f'C 로트 안 Run 간 상태 변수 η² < {ETA_LIMIT}',
         '결과': f'η² ≥ {ETA_LIMIT}인 (로트, 변수) {n_split} / {len(st)}' if len(st) else 'Run 2개 이상인 로트 없음',
         '판정': '해당 없음' if len(st) == 0 else ('통과' if n_split == 0 else '확인')},
        {'기준': 'D 로트 간 불량률 차이 (참고)',
         '결과': '; '.join(f"{r['product']} p={r['p(순열)']:.3f}" for _, r in QT[QT['단위'] == 'lot'].iterrows()),
         '판정': '참고'},
    ]
    return pd.DataFrame(rows)


# ====================================================================== 그림
def plot_lot_timeline(S, save=None):
    """로트별 생산 구간 — x축 = 생산계획일 0시 기준 경과 시간(음수 = 전날 밤)"""
    set_style()
    S = S.copy()
    S['rel_h'] = (S['TimeStamp'] - S['plan_date']).dt.total_seconds() / 3600
    run_col = [ACCENT, '#eb6834', '#4a3aa7']
    prods = [p for p in PRODUCTS if (S['product'] == p).any()]
    fig, axes = plt.subplots(1, len(prods), figsize=(6.5 * len(prods), 4.2), sharex=True, squeeze=False)
    for ax, p in zip(axes[0], prods):
        d = S[S['product'] == p]
        lots = sorted(d['lot'].unique())
        for i, lot in enumerate(lots):
            gl = d[d['lot'] == lot]
            for j, (r, gr) in enumerate(gl.groupby('run_id')):
                yy = i + 0.14 * j
                ax.hlines(yy, gr['rel_h'].min(), gr['rel_h'].max(), lw=6, color=run_col[j % 3])
                ax.text(gr['rel_h'].max() + 0.2, yy, r.split('_')[-1], fontsize=7, va='center', color=SUB)
            f = gl[gl['y'] == 1]
            ax.scatter(f['rel_h'], np.full(len(f), i), s=16, color='#d03b3b', zorder=3)
        ax.axvline(0, color=MUTED, ls='--', lw=1)
        ax.set_yticks(range(len(lots))); ax.set_yticklabels(lots, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('생산계획일 0시 기준 경과 시간 (h, 음수 = 전날 밤)')
        ax.set_title(f'{p.upper()} 로트별 생산 구간 (막대 색 = Run, 빨간 점 = 공정 불량)', loc='left',
                     fontsize=9, color=INK)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    return fig


# ====================================================================== (선택) 1차 원본
def check_primary(primary_path, products=PRODUCTS):
    """PART_FACT_PLAN_DATE 원본 형식, (있으면) 시리얼이 계획일 안에서 시간순으로 증가하는지"""
    if not os.path.exists(primary_path):
        return None
    want = {'PART_NAME', 'EQUIP_NAME', 'PART_FACT_PLAN_DATE', 'PART_FACT_SERIAL', 'TimeStamp'}
    P = pd.read_csv(primary_path, usecols=lambda c: c in want, low_memory=False)
    P = P[P['PART_NAME'].str[:3].str.lower().isin(products)]
    out = {'원본 값 예시': P['PART_FACT_PLAN_DATE'].drop_duplicates().head(8).tolist()}
    if 'PART_FACT_SERIAL' in P.columns:
        P['TimeStamp'] = pd.to_datetime(P['TimeStamp'])
        P = P.sort_values('TimeStamp')
        mono = P.groupby(['PART_NAME', 'PART_FACT_PLAN_DATE'])['PART_FACT_SERIAL'].apply(
            lambda s: pd.to_numeric(s, errors='coerce').dropna().is_monotonic_increasing)
        out['계획일 안 시리얼 시간순 증가 비율'] = round(float(mono.mean()), 3)
    return out


# ====================================================================== 한 번에
def run_all(base_dir='.', model_dir='./outputs/final_model', out_dir='./outputs/lot', primary_path=None):
    os.makedirs(out_dir, exist_ok=True)
    S, n_mismatch = load_lot_shots(base_dir, model_dir)
    LOT = lot_summary(S)
    OV = lot_overlap(LOT)
    BD = boundary_table(S)
    CT, run_purity, lot_runs = run_lot_table(S)
    ETA = eta_table(S)
    WITHIN = within_lot_runs(S)
    QT = quality_table(S)
    ALARM = alarm_by_lot(S)
    V = verdict(S, LOT, OV, BD, run_purity, WITHIN, QT)
    res = {'S': S, 'n_mismatch': n_mismatch, 'LOT': LOT, 'OFFSET': offset_table(S), 'CUT': cutoff_rule(S),
           'OV': OV, 'BD': BD, 'CT': CT, 'run_purity': run_purity, 'lot_runs': lot_runs,
           'ETA': ETA, 'WITHIN': WITHIN, 'QT': QT, 'ALARM': ALARM, 'VERDICT': V,
           'PRIMARY': check_primary(primary_path) if primary_path else None}
    for name in ['LOT', 'OFFSET', 'CUT', 'OV', 'BD', 'ETA', 'WITHIN', 'QT', 'ALARM', 'VERDICT']:
        if len(res[name]):
            res[name].to_csv(f'{out_dir}/lot_{name.lower()}.csv', index=False, encoding='utf-8-sig')
    CT.to_csv(f'{out_dir}/lot_run_crosstab.csv', encoding='utf-8-sig')
    return res
