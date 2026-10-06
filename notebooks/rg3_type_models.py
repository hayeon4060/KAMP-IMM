"""
rg3_type_models.py — 35번: RG3 유형별 탐지 모델 (가스 / 미성형) 학습·테스트

34번 스캔 결과를 모델로 옮겨 검증
  - 가스: 쿠션위치 직전 10샷 std가 147개 변수 보정 뒤에도 유의 (보정 p 0.025)
  - 미성형: 유의한 변수 없음. 약한 경향 = 계량시간 편차 ↑, 호퍼온도 변화 ↓

⚠ 34번은 labeled 전체를 보고 변수를 골랐음 → 그 변수로 같은 데이터를 평가하면 낙관적
   → 'nested_top3'(각 Run을 뺄 때마다 남은 Run으로만 스캔해 변수 선택)가 정직한 추정치

데이터
  - labeled: base/rg3_shots.csv (평가 대상 = split≠제외, 초기허용 제외)  가스 22 / 미성형 10 / 정상 595
  - 정상 기준: unlabeled reference (최종 모델과 같은 규칙, 라벨 0개) → 비지도 학습·경보 임계값(백분위)에 사용
  - 변수: 34번 변수 은행 중 '실시간 가능한 것'만 (미래 포함 변환 제외)

모델 (대상별, 모두 사전 고정)
  [가스]
    rule_cushion_std   : 쿠션위치 직전 10샷 std 하나 (값이 클수록 의심) — 학습 없음
    maha_std4          : 하연님 std4(배럴2·호퍼·쿠션·배럴5 직전 10샷 std) Mahalanobis, 정상 reference로 학습 — 비지도
    rf_std4            : std4 RandomForest (가스 vs 정상), Run 하나씩 빼고 검증 — 지도 (하연님 방식)
    nested_top3        : 학습 Run에서만 스캔 → |δ| 상위 3개 변수 → 방향 맞춘 robust z 평균 — 지도(중첩)
  [미성형]
    rule_plast_dev     : 계량시간 직전 20샷 편차 하나 (클수록 의심) — 학습 없음 (계량 지연 가설)
    maha_short6        : 31번 SHORT6(계량시간 편차·변화, 배압·계량위치·쿠션·충전시간 편차) Mahalanobis — 비지도
    rf_short6          : SHORT6 RandomForest (미성형 vs 정상), Run 하나씩 빼고 — 지도
    nested_top3        : 위와 같은 중첩 방식
  [공통 기준] 최종 모델(rg3 Mahalanobis, outputs/final_model/rg3_scores.csv 있으면), 무작위

평가 (대상 vs 정상, 다른 불량 유형은 제외)
  AP·기준선·배수, AP 순열 p(같은 Run 안 라벨 섞기 1,000번), 전체·Run별 상위 10% 포착,
  정상 reference 백분위 q=0.99 / 0.995 경보의 TP·FP·정밀도 (임계값은 라벨 미사용)
"""
import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

from final_model import build_reference, load_labeled_shots, SEED, CLIP
from rg3_short_scan import load as load_labeled, feature_bank, _ranks, _auc_from_ranks, STOP_GAP_SEC

STD4 = ['Barrel_Temperature_2|직전10샷std', 'Hopper_Temperature|직전10샷std',
        'Cushion_Position|직전10샷std', 'Barrel_Temperature_5|직전10샷std']
SHORT6 = ['Plasticizing_Time|편차20', 'Plasticizing_Time|직전변화', 'Average_Back_Pressure|편차20',
          'Plasticizing_Position|편차20', 'Cushion_Position|편차20', 'Filling_Time|편차20']
RULES = {'가스': ('rule_cushion_std', 'Cushion_Position|직전10샷std', +1),
         '미성형': ('rule_plast_dev', 'Plasticizing_Time|편차20', +1)}
SETS = {'가스': ('std4', STD4), '미성형': ('short6', SHORT6)}
TOPK = 3
QS = (0.99, 0.995)
N_PERM = 1000


# ====================================================================== 데이터
def add_context(d):
    d = d.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    g = d.groupby('run_id', sort=False)
    d['seq'] = g.cumcount() + 1
    gap = g['TimeStamp'].diff().dt.total_seconds()
    d['prev_gap_sec'] = gap
    restart = gap.isna() | (gap > STOP_GAP_SEC)
    seg = restart.groupby(d['run_id']).cumsum()
    d['since_restart'] = d.groupby([d['run_id'], seg]).cumcount() + 1
    return d


def load_all(p='rg3', base_dir='.', verbose=True):
    s = load_labeled(p, base_dir)
    Xs = feature_bank(s)
    Xs = Xs[[c for c in Xs.columns if '미래' not in c]]                 # 실시간 가능한 변수만
    ref, log = build_reference(p, load_labeled_shots(p, base_dir), base_dir)
    ref = add_context(ref)
    Xr = feature_bank(ref).reindex(columns=Xs.columns)
    if verbose:
        print(f'[{p}] labeled {len(s):,}샷 | 정상 reference {len(ref):,}샷 | 실시간 변수 {Xs.shape[1]}개')
        print('  평가 대상:', s.loc[s['eval'], 'type'].value_counts().to_dict())
    return s, Xs, ref, Xr


# ====================================================================== 점수 도구
class RobustZ:
    """reference 중앙값·IQR로 표준화 → ±CLIP → NaN=0"""

    def fit(self, X):
        X = np.asarray(X, float)
        self.med = np.nanmedian(X, 0)
        q75, q25 = np.nanpercentile(X, [75, 25], 0)
        sd = np.nanstd(X, 0)
        self.iqr = np.where(q75 - q25 > 1e-9, q75 - q25, np.where(sd > 1e-9, sd, 1.0))
        return self

    def __call__(self, X):
        Z = (np.asarray(X, float) - self.med) / self.iqr
        return np.nan_to_num(np.clip(Z, -CLIP, CLIP), nan=0.0)


def maha_scores(Xr, Xs):
    z = RobustZ().fit(Xr)
    lw = LedoitWolf().fit(z(Xr))
    return np.sqrt(lw.mahalanobis(z(Xs))), np.sqrt(lw.mahalanobis(z(Xr)))


def to_pct(ref_scores, s):
    r = np.sort(np.asarray(ref_scores, float))
    r = r[np.isfinite(r)]
    return np.searchsorted(r, np.asarray(s, float), side='right') / (len(r) + 1)


def _rf():
    return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, n_jobs=-1,
                                  class_weight='balanced_subsample', random_state=SEED)


def _masks(s, target):
    ev = s['eval'].values
    t = s['type'].values
    pos = ev & (t == target)
    neg = ev & (t == '정상')
    return pos, neg


def select_topk(s, X, rows, target, k=TOPK):
    """rows(학습 샷)에서 대상 vs 정상 AUC 스캔 → |δ| 상위 k개 (변수, 방향)"""
    pos, neg = _masks(s, target)
    m = rows & (pos | neg)
    R = _ranks(X[m].reset_index(drop=True))
    auc = _auc_from_ranks(R, pos[m])
    order = np.argsort(-np.abs(auc - 0.5))[:k]
    return [(X.columns[i], 1 if auc[i] > 0.5 else -1, float(2 * auc[i] - 1)) for i in order]


# ====================================================================== 모델별 점수 (labeled 점수, reference 점수)
def score_models(s, Xs, ref, Xr, target, base_dir='.', model_dir='./outputs/final_model'):
    """반환: {모델: (labeled 점수, reference 점수 or None)}, 중첩 선택 기록"""
    out = {}
    runs = s['run_id'].values
    pos, neg = _masks(s, target)

    # 1) 규칙 하나 (학습 없음)
    name, col, sign = RULES[target]
    out[name] = (sign * Xs[col].values, sign * Xr[col].values)

    # 2) 비지도 Mahalanobis (정상 reference 학습)
    sname, cols = SETS[target]
    sl, sr = maha_scores(Xr[cols].values, Xs[cols].values)
    out[f'maha_{sname}'] = (sl, sr)

    # 3) 지도 RF (Run 하나씩 빼고)  4) 중첩 top-k
    rf_l = np.full(len(s), np.nan)
    nt_l = np.full(len(s), np.nan)
    rf_r = np.zeros(len(ref))
    nt_r = np.zeros(len(ref))
    picks = []
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    ur = pd.unique(runs)
    for r in ur:
        te = runs == r
        tr = ~te
        y_tr = pos[tr & (pos | neg)]
        if y_tr.sum() >= 2:
            Xtr = Xs.loc[tr & (pos | neg), cols].values
            imp_f = imp.fit(Xtr)
            m = _rf().fit(imp_f.transform(Xtr), y_tr.astype(int))
            rf_l[te] = m.predict_proba(imp_f.transform(Xs.loc[te, cols].values))[:, 1]
            rf_r += m.predict_proba(imp_f.transform(Xr[cols].values))[:, 1] / len(ur)
        sel = select_topk(s, Xs, tr, target)
        fcols = [c for c, _, _ in sel]
        signs = np.array([sg for _, sg, _ in sel])
        z = RobustZ().fit(Xr[fcols].values)
        nt_l[te] = (z(Xs.loc[te, fcols].values) * signs).mean(1)
        nt_r += (z(Xr[fcols].values) * signs).mean(1) / len(ur)
        picks.append({'대상': target, '뺀 Run': r, '학습 대상 샷': int(y_tr.sum()),
                      **{f'선택{i + 1}': f'{c} ({"↑" if sg > 0 else "↓"}, δ={d:+.2f})' for i, (c, sg, d) in enumerate(sel)}})
    out[f'rf_{sname}'] = (np.nan_to_num(rf_l, nan=0.0), rf_r)
    out['nested_top3'] = (nt_l, nt_r)

    # 5) 기준: 최종 모델
    f = f'{model_dir}/rg3_scores.csv'
    if os.path.exists(f):
        fm = pd.read_csv(f, parse_dates=['TimeStamp'])[['TimeStamp', 'score']].drop_duplicates('TimeStamp')
        sc = s[['TimeStamp']].merge(fm, on='TimeStamp', how='left')['score'].values
        out['final_maha (기준)'] = (np.nan_to_num(sc, nan=0.0), None)
    return out, pd.DataFrame(picks)


# ====================================================================== 평가
def _perm_ap(y, sc, runs, n=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = average_precision_score(y, sc)
    groups = [np.flatnonzero(runs == r) for r in pd.unique(runs)]
    null = np.empty(n)
    for b in range(n):
        yp = y.copy()
        for g in groups:
            yp[g] = rng.permutation(y[g])
        null[b] = average_precision_score(yp, sc)
    return obs, float(null.mean()), (1 + (null >= obs).sum()) / (n + 1)


def _clean(x):
    """결측·무한대(Run 첫 샷처럼 계산 불가)는 '가장 덜 의심'으로: 유한값 최솟값 − 1"""
    x = np.asarray(x, float).copy()
    ok = np.isfinite(x)
    x[~ok] = (x[ok].min() - 1.0) if ok.any() else 0.0
    return x


def evaluate(s, target, scores):
    pos, neg = _masks(s, target)
    m = pos | neg
    y = pos[m].astype(int)
    runs = s['run_id'].values[m]
    rows = []
    for name, (sl, sr) in scores.items():
        sl = _clean(sl)
        sc = sl[m] + np.random.default_rng(0).random(m.sum()) * 1e-9
        ap, null_mean, p = _perm_ap(y, sc, runs)
        k = int(np.ceil(0.1 * len(sc)))
        top = np.zeros(len(sc), bool); top[np.argsort(-sc)[:k]] = True
        run_hit = 0
        for r in pd.unique(runs):
            idx = np.flatnonzero(runs == r)
            kk = int(np.ceil(0.1 * len(idx)))
            run_hit += int(y[idx[np.argsort(-sc[idx])[:kk]]].sum())
        row = {'대상': target, '모델': name, '대상 샷': int(y.sum()), '정상 샷': int((1 - y).sum()),
               'AP': ap, '기준선(불량률)': y.mean(), 'AP/기준선': ap / y.mean(), '순열 p (vs 무작위)': p,
               '유의': 'O' if p < 0.05 else '-',
               '전체 상위10%': f'{int((top & (y == 1)).sum())}/{int(y.sum())}',
               'Run별 상위10%': f'{run_hit}/{int(y.sum())}'}
        if sr is not None:
            pc = to_pct(sr, sl[m])
            for q in QS:
                a = pc >= q
                tp, fp = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
                row[f'q{q} 경보'] = int(a.sum())
                row[f'q{q} TP'] = tp
                row[f'q{q} FP'] = fp
                row[f'q{q} 정밀도'] = tp / (tp + fp) if tp + fp else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(p='rg3', base_dir='.', model_dir='./outputs/final_model', out_dir='./outputs/rg3_type_models', verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    s, Xs, ref, Xr = load_all(p, base_dir, verbose)
    R = {'shots': s}
    for target in ('가스', '미성형'):
        sc, picks = score_models(s, Xs, ref, Xr, target, base_dir, model_dir)
        ev = evaluate(s, target, sc)
        R[f'eval_{target}'], R[f'picks_{target}'], R[f'scores_{target}'] = ev, picks, sc
        ev.to_csv(f'{out_dir}/eval_{target}.csv', index=False, encoding='utf-8-sig')
        picks.to_csv(f'{out_dir}/nested_picks_{target}.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame({'TimeStamp': s['TimeStamp'], 'run_id': s['run_id'], 'type': s['type'], 'eval': s['eval'],
                      **{k: v[0] for k, v in sc.items()}}).to_csv(f'{out_dir}/scores_{target}.csv', index=False,
                                                                   encoding='utf-8-sig')
        if verbose:
            print(f'  {target} 완료', flush=True)
    return R


# ====================================================================== 그림
def plot_pr(R, save=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    s = R['shots']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, target in zip(axes, ('가스', '미성형')):
        pos, neg = _masks(s, target)
        m = pos | neg
        y = pos[m].astype(int)
        for name, (sl, _) in R[f'scores_{target}'].items():
            sl = _clean(sl)
            pr, rc, _ = precision_recall_curve(y, sl[m])
            ap = average_precision_score(y, sl[m])
            ax.step(rc, pr, where='post', lw=1.6 if 'final' not in name else 1.0,
                    ls='--' if 'final' in name else '-', label=f'{name} (AP {ap:.3f})')
        ax.axhline(y.mean(), color='k', lw=0.8, ls=':', label=f'무작위 ({y.mean():.3f})')
        ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_ylim(0, 1.02)
        ax.set_title(f'RG3 {target} vs 정상 — PR 곡선', loc='left', fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
