"""
daily_check.py — CN7: unlabeled 날짜별 '온도 교란 언덕 수'와 '일별 불량 수(ERR_FACT_QTY)' 비교

데이터 (linked_unlabeled_cn7.csv)
  - ERR_FACT_QTY: 생산계획일(PART_FACT_PLAN_DATE) × 캐비티(LH/RH)별 불량 수 → 날짜별 합계 사용
  - 샷 날짜도 TimeStamp가 아니라 PART_FACT_PLAN_DATE 기준 (불량 수와 같은 날짜 기준, 자정 넘는 근무 반영)

언덕 기준 (29번에서 정한 그대로)
  - 구간 점수 E = 호퍼온도 추세↑·금형온도↓·호퍼온도 편차↑ 성분의 최근 5샷 평균
  - 임계값 = 정상 기준(reference)에서 재가동 후 15샷을 뺀 E의 95% 분위수 (정상 오경보율 5%)
  - 언덕(에피소드) = 같은 Run에서 연속 경보 묶음, 재가동 후 15샷 이내 경보는 제외
  - 비교용: 최종 모델 경보(pct ≥ 0.999) 수도 같이 계산

주의
  - 날짜별로 기록된 샷 수가 실제 생산량보다 적은 날이 많음 (몇 샷만 기록된 날에도 불량 수는 큼)
    → 생산 샷이 MIN_SHOTS 이상인 날만 비교
  - ERR_FACT_QTY는 모든 불량 유형 합계(가스만이 아님)일 수 있음 → '관련성' 수준으로만 해석
  - labeled 기간 날짜도 포함 (표에 표시)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from final_model import raw, ALL_RAW, FinalModel, make_features, load_labeled_shots
from thermal_episode import _episode_score
from thermal_cn7 import load as load_cn7, add_since_restart

ZERO_MODE_BARREL = 100
RUN_GAP_MIN, MIN_CYCLE_SEC, CYCLE_DEV_SEC = 60, 50, 2
N_EXCL = 15
Q = 0.95
MIN_SHOTS = 300


def load_daily_errors(p='cn7', base_dir='.'):
    u = pd.read_csv(f'{base_dir}/linked/linked_unlabeled_{p}.csv', parse_dates=['TimeStamp'], low_memory=False)
    u['plan_date'] = pd.to_datetime(u['PART_FACT_PLAN_DATE'].astype(str).str[:10])
    e = u.groupby(['plan_date', 'cavity'])['ERR_FACT_QTY'].first().unstack()
    e.columns = [f'불량_{c}' for c in e.columns]
    e['불량_합계'] = e.sum(axis=1, min_count=1)
    return u, e


def production_shots(u):
    """샷 1행으로 → 0 모드 제거 → 끼어든 이상 기록 제거 → Run 구분 (최종 모델과 같은 규칙, 기간·labeled 제거 없음)"""
    s = u.drop_duplicates('TimeStamp').sort_values('TimeStamp').reset_index(drop=True)
    s['zero'] = s[raw('Barrel_Temperature_1')] < ZERO_MODE_BARREL
    allday = s.groupby('plan_date').agg(기록_샷=('TimeStamp', 'size'), 영모드_비율=('zero', 'mean'))
    s = s[~s['zero']].reset_index(drop=True)
    gp = s['TimeStamp'].diff().dt.total_seconds()
    gn = -s['TimeStamp'].diff(-1).dt.total_seconds()
    day_med = s.groupby(s['TimeStamp'].dt.date)[raw('Cycle_Time')].transform('median')
    odd = (np.fmin(gp, gn) < MIN_CYCLE_SEC) & ((s[raw('Cycle_Time')] - day_med).abs() > CYCLE_DEV_SEC)
    s = s[~odd].reset_index(drop=True)
    gap = s['TimeStamp'].diff().dt.total_seconds()
    s['run_id'] = 'U' + (gap.isna() | (gap > RUN_GAP_MIN * 60)).cumsum().astype(str).str.zfill(4)
    keep = ['run_id', 'TimeStamp', 'plan_date'] + [raw(v) for v in ALL_RAW]
    return s[keep], allday


def build(p='cn7', base_dir='.', model_dir='./outputs/final_model'):
    """날짜별 표 + 샷별 점수"""
    model = FinalModel.load(f'{model_dir}/{p}_final_model.pkl')
    lab, ref = load_cn7(p, base_dir, model_dir)                            # 29번과 같은 정상 기준
    hs = ref['hopper_slope20'].dropna()
    med, iqr = hs.median(), max(hs.quantile(0.75) - hs.quantile(0.25), 1e-6)
    th = float(ref.loc[ref['since_restart'] > N_EXCL, 'E'].quantile(Q))

    u, err = load_daily_errors(p, base_dir)
    s, allday = production_shots(u)
    f = make_features(s, model.steps, model.fr_step)
    f = add_since_restart(_episode_score(f, med, iqr))
    f['pct'] = model.pct(model.score(f))
    f['al'] = (f['E'] >= th) & (f['since_restart'] > N_EXCL)
    prev = f.groupby('run_id', sort=False)['al'].shift(fill_value=False)
    f['ep_start'] = f['al'] & ~prev
    f['final_alarm'] = f['pct'] >= 0.999

    d = f.groupby('plan_date').agg(생산_샷=('TimeStamp', 'size'), 언덕_수=('ep_start', 'sum'),
                                   온도경보_샷=('al', 'sum'), 최종모델_경보=('final_alarm', 'sum'),
                                   금형온도_중앙=(raw('Mold_Temperature_3'), 'median'),
                                   호퍼온도_중앙=(raw('Hopper_Temperature'), 'median'))
    d = allday.join(d, how='left').join(err, how='left')
    d['생산_샷'] = d['생산_샷'].fillna(0).astype(int)
    d['1000샷당_언덕'] = 1000 * d['언덕_수'] / d['생산_샷'].replace(0, np.nan)
    d['1000샷당_최종경보'] = 1000 * d['최종모델_경보'] / d['생산_샷'].replace(0, np.nan)
    d['불량률(불량/(생산샷×2))'] = d['불량_합계'] / (2 * d['생산_샷'].replace(0, np.nan))
    lab_days = set(pd.to_datetime(load_labeled_shots(p, base_dir)['TimeStamp']).dt.normalize())
    d['labeled 기간'] = d.index.isin(lab_days)
    d['비교 대상'] = (d['생산_샷'] >= MIN_SHOTS) & d['불량_합계'].notna()
    info = {'임계값 E': th, '정상기준 hopper_slope20 중앙값': med, 'IQR': iqr}
    return d.reset_index().rename(columns={'plan_date': '날짜'}), f, info


def _perm_p(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    r0 = spearmanr(x, y).correlation
    rs = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(n)])
    return r0, float(np.mean(np.abs(rs) >= abs(r0)))


def correlations(d):
    t = d[d['비교 대상']]
    rows = []
    for xname in ['1000샷당_언덕', '언덕_수', '1000샷당_최종경보', '금형온도_중앙', '호퍼온도_중앙']:
        for yname in ['불량_합계', '불량률(불량/(생산샷×2))']:
            ok = t[[xname, yname]].dropna()
            if len(ok) < 4:
                continue
            r, p = _perm_p(ok[xname].values, ok[yname].values)
            rows.append({'x (날짜별)': xname, 'y (날짜별)': yname, '날짜 수': len(ok), 'Spearman ρ': r, '순열 p': p})
    return pd.DataFrame(rows)


def plot(d, save=None):
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    t = d[d['비교 대상']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, x, title in [(axes[0], '1000샷당_언덕', '온도 교란 언덕 (q=0.95, 재가동 15샷 제외)'),
                         (axes[1], '1000샷당_최종경보', '최종 모델 경보 (pct ≥ 0.999)')]:
        col = np.where(t['labeled 기간'], '#e67e22', '#1f4e79')
        ax.scatter(t[x], t['불량률(불량/(생산샷×2))'], c=col, s=40)
        for _, r in t.iterrows():
            ax.annotate(r['날짜'].strftime('%m-%d'), (r[x], r['불량률(불량/(생산샷×2))']), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel(f'1,000샷당 {title}'); ax.set_ylabel('일별 불량률 (불량 / 생산샷×2)')
        ax.set_title(title, loc='left', fontsize=10)
    axes[0].text(0.01, 0.98, '주황 = labeled 기간 날짜', transform=axes[0].transAxes, fontsize=7, va='top')
    fig.suptitle(f'CN7 날짜별 비교 (생산 샷 {MIN_SHOTS}개 이상인 날)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
