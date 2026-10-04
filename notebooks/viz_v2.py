"""
viz_v2.py — v2 결과 진단 그래프

A. 판정·순위 (oof.csv만 있으면 됨)
   plot_score_threshold : 정상 vs 불량 유형별 점수 분포 + 임계값(오경보율 1 / 0.5 / 0.1%) → 몇 개가 넘는지
   plot_pr              : Precision-Recall 곡선 + 임계값 운영점 + 무작위 기준선
   plot_inspection      : 검사 비율 vs 잡은 불량 비율 (무작위·상한 같이) → R@k를 곡선으로
   plot_score_timeline  : Run별 샷 순번 × 점수, 임계값, 불량 위치
B. 변수 수준 (feature csv + reference 필요)
   deviation_heatmap    : 불량·오탐 샷이 어떤 변수에서 정상 생산(reference) 대비 얼마나 벗어났나
   univariate_auc       : 변수 하나만으로 불량을 얼마나 가르나 (+부트스트랩 CI) → '의미 있는 변수' 1차 판단
   group_ablation       : 이상탐지에서 변수 묶음을 빼면/그 묶음만 쓰면 성능이 어떻게 변하나 → 묶음의 기여
   logit_coef_stability : 통합 logit 계수가 LORO fold마다 같은 방향인가 → 우연인지 안정적인 신호인지

점수 축: 'surprise' = −log10(1 − pct). 2 = 정상 샷의 1%만 이보다 높음, 3 = 0.1%
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score

from plot_style import set_style, short, INK, SUB, MUTED, NORMAL, TYPE_COL, TYPE_ORDER, DIV
from model_v2 import RefDetector, to_ref_scale, make_sup, _X, SEED
from loro_eval import recall_at_k_by_run

QS = (0.99, 0.995, 0.999)
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']          # 방법 비교용 (고정 순서, 최대 4개)


def surprise(pct):
    return -np.log10(1 - np.clip(np.asarray(pct, float), 0, 1 - 1e-9))


def _type(o):
    t = np.where(o['y'] == 0, '정상', o['Reason'])
    return pd.Series(t, index=o.index)


def _sel(oof, method, product):
    o = oof[(oof['method'] == method) & (oof['product'] == product)].copy()
    o['유형'] = _type(o)
    o['S'] = surprise(o['pct'])
    return o


# ====================================================================== A. 판정·순위
def plot_score_threshold(oof, method, product, qs=QS, save=None):
    """정상/유형별 점수 분포 + 임계값선. 오른쪽 표: 임계값마다 넘은 샷 수 (정상=오탐, 불량=탐지)"""
    set_style()
    o = _sel(oof, method, product)
    cats = ['정상'] + [t for t in TYPE_ORDER if (o['유형'] == t).any()]
    rng = np.random.default_rng(0)
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [2.3, 1]})
    for i, t in enumerate(cats):
        v = o.loc[o['유형'] == t, 'S']
        ax.scatter(i + rng.uniform(-0.25, 0.25, len(v)), v, s=8 if t == '정상' else 46,
                   color=TYPE_COL[t], alpha=0.45 if t == '정상' else 1,
                   edgecolor='none' if t == '정상' else 'white', lw=1.3, zorder=3 if t != '정상' else 2)
        ax.boxplot(v, positions=[i], widths=0.6, showfliers=False, medianprops=dict(color=INK),
                   boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    rows = []
    for q, ls in zip(qs, [':', '--', '-']):
        thr = surprise(q)
        ax.axhline(thr, color=INK, lw=0.9, ls=ls, zorder=1)
        ax.text(len(cats) - 0.45, thr, f' pct≥{q}\n (오경보율 {100 * (1 - q):g}%)', va='center', fontsize=8,
                color=SUB)
        r = {'임계값': f'pct≥{q}'}
        for t in cats:
            m = o['유형'] == t
            r[t] = f"{int((o.loc[m, 'pct'] >= q).sum())}/{int(m.sum())}"
        rows.append(r)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f'{t}\n(n={int((o["유형"] == t).sum())})' for t in cats])
    ax.set_xlim(-0.6, len(cats) + 0.4)
    ax.set_ylabel('이상 정도  −log10(1−pct)')
    ax.set_title(f'{product.upper()} · {method} — 정상과 불량이 임계값 기준으로 얼마나 갈리나', loc='left')
    tb = pd.DataFrame(rows).set_index('임계값')
    axt.axis('off')
    tab = axt.table(cellText=tb.values, rowLabels=tb.index, colLabels=tb.columns, loc='center', cellLoc='center')
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.6)
    axt.set_title('임계값을 넘은 샷 (정상 = 오탐, 불량 = 탐지)', fontsize=9, color=SUB)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()
    return tb


def plot_pr(oof, methods, product, qs=QS, save=None):
    """PR 곡선. 점 = pct 임계값 운영점. 회색 점선 = 무작위(불량률)"""
    set_style()
    fig, ax = plt.subplots(figsize=(6.8, 5))
    base = None
    for col, m in zip(SERIES, methods[:4]):
        o = _sel(oof, m, product)
        y, s = o['y'].values, o['score'].values
        p, r, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.step(r, p, where='post', color=col, lw=2, label=f'{m}  (AP {ap:.3f})')
        for q, mk in zip(qs, ['o', 's', 'D']):
            pred = o['pct'].values >= q
            if pred.sum():
                ax.scatter((pred & (y == 1)).sum() / y.sum(), (pred & (y == 1)).sum() / pred.sum(), marker=mk,
                           s=60, color=col, edgecolor='white', lw=1.5, zorder=4)
        base = y.mean()
    ax.axhline(base, color=MUTED, ls='--', lw=1)
    ax.text(0.99, base, f'무작위 = 불량률 {base:.3f}', ha='right', va='bottom', fontsize=8, color=SUB)
    for q, mk in zip(qs, ['o', 's', 'D']):
        ax.scatter([], [], marker=mk, color=MUTED, label=f'운영점 pct≥{q}')
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall (잡은 불량 / 전체 불량)'); ax.set_ylabel('Precision (경보 중 실제 불량)')
    ax.set_title(f'{product.upper()} — Precision-Recall', loc='left')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def _recall_at(o, k, pos, group_col):
    g = o.copy()
    g['run_id'] = g[group_col] if group_col else 'ALL'
    h, t = recall_at_k_by_run(g, k, pos)
    return h / t if t else np.nan


def plot_inspection(oof, methods, product, group_col='run_id', ks=None, save=None):
    """
    x = 묶음(group_col)마다 상위 k% 검사, y = 잡은 불량 비율
    group_col: 'run_id' / 'plan_date' (oof에 컬럼 추가 필요) / None(전체 기간 한 묶음)
    회색 대각선 = 무작위, 검정 점선 = 상한(완벽한 순위)
    """
    set_style()
    ks = ks if ks is not None else np.r_[0.01, np.arange(0.02, 0.31, 0.02)]
    fig, ax = plt.subplots(figsize=(6.8, 5))
    o0 = _sel(oof, methods[0], product)
    pos0 = o0['y'] == 1
    ceil = []
    for k in ks:
        g = o0.copy(); g['grp'] = g[group_col] if group_col else 'ALL'
        hit = sum(min(int(x['y'].sum()), max(1, int(np.ceil(k * len(x))))) for _, x in g.groupby('grp'))
        ceil.append(hit / pos0.sum())
    ax.plot(ks * 100, ceil, color=INK, ls=':', lw=1.4, label='상한 (완벽한 순위)')
    ax.plot(ks * 100, ks, color=MUTED, ls='--', lw=1, label='무작위')
    for col, m in zip(SERIES, methods[:4]):
        o = _sel(oof, m, product)
        ax.plot(ks * 100, [_recall_at(o, k, o['y'] == 1, group_col) for k in ks], color=col, lw=2,
                marker='o', ms=4, label=m)
    for k in (5, 10):
        ax.axvline(k, color=MUTED, lw=0.6)
    ax.set_xlabel(f'검사 비율 (%) — {group_col or "전체 기간"} 단위 상위 k%')
    ax.set_ylabel('잡은 불량 비율 (Recall@k)')
    ax.set_ylim(0, 1.02)
    ax.set_title(f'{product.upper()} — 검사량 대비 불량 포착', loc='left')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def plot_score_timeline(oof, method, product, q=0.995, save=None):
    """Run별 점수 흐름 — 임계값선, 불량(색 점), 오탐(×)"""
    set_style()
    o = _sel(oof, method, product)
    runs = sorted(o['run_id'].unique())
    thr = surprise(q)
    fig, axes = plt.subplots(len(runs), 1, figsize=(13, 1.9 * len(runs)))
    for ax, r in zip(np.atleast_1d(axes), runs):
        g = o[o['run_id'] == r].sort_values('shot_seq')
        ax.plot(g['shot_seq'], g['S'], color=MUTED, lw=0.9, zorder=1)
        ax.axhline(thr, color=INK, ls='--', lw=0.9)
        for t in TYPE_ORDER:
            h = g[g['유형'] == t]
            ax.scatter(h['shot_seq'], h['S'], s=46, color=TYPE_COL[t], edgecolor='white', lw=1.3, zorder=3,
                       label=t)
        fp = g[(g['y'] == 0) & (g['pct'] >= q)]
        ax.scatter(fp['shot_seq'], fp['S'], marker='x', s=40, color=INK, lw=1.2, zorder=4, label='오탐')
        tp = int(((g['y'] == 1) & (g['pct'] >= q)).sum())
        ax.set_title(f"{r} — 불량 {int(g['y'].sum())}, 탐지 {tp}, 오탐 {len(fp)}", loc='left', fontsize=9)
    h, l = [], []
    for ax in np.atleast_1d(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc='upper right', ncol=4, fontsize=8)
    fig.suptitle(f'{product.upper()} · {method} — 점수 타임라인 (파선 = pct≥{q})', x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save:
        fig.savefig(save)
    plt.show()


# ====================================================================== B. 변수 수준
def ref_z(d, ref, cols):
    """reference(정상 생산) 중앙값·IQR 기준 robust z — 이상탐지 입력과 같은 척도"""
    X, R = d[cols].astype(float), ref[cols].astype(float)
    med = R.median()
    iqr = (R.quantile(.75) - R.quantile(.25)).replace(0, np.nan).fillna(R.std()).replace(0, 1)
    return (X - med) / iqr


def deviation_heatmap(d, ref, cols, oof, method, product, q=0.995, max_fp=15, save=None):
    """
    행 = 불량 샷 + 오탐 상위 max_fp개, 열 = 변수, 색 = 정상 생산 대비 z (±8에서 색 고정)
    → 탐지된 불량은 어떤 변수가 튀었는지, 놓친 불량은 왜 평범했는지, 오탐은 뭐 때문인지
    """
    set_style()
    o = _sel(oof, method, product)[['run_id', 'shot_seq', 'y', 'pct']]       # Reason은 d 쪽 사용
    dd = d.merge(o, left_on=['run_id', 'shot_seq_in_run'], right_on=['run_id', 'shot_seq'], how='inner')
    Z = ref_z(dd, ref, cols)
    det = dd['pct'] >= q
    fp = dd[(dd['y'] == 0) & det].nlargest(max_fp, 'pct').index
    rows = list(dd[dd['y'] == 1].sort_values(['Reason', 'run_id', 'shot_seq']).index) + list(fp)
    M = Z.loc[rows].fillna(0).values
    lab = [f"{dd.at[i, 'run_id']} #{dd.at[i, 'shot_seq']}  "
           f"{dd.at[i, 'Reason'] if dd.at[i, 'y'] == 1 else '오탐'}{'  ●' if det[i] and dd.at[i, 'y'] == 1 else ''}"
           for i in rows]
    fig, ax = plt.subplots(figsize=(0.42 * len(cols) + 4, 0.3 * len(rows) + 2.4))
    im = ax.imshow(M, cmap=DIV, vmin=-8, vmax=8, aspect='auto')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([short(c) for c in cols], rotation=60, ha='right',
                                                        fontsize=8)
    nd = int((dd['y'] == 1).sum())
    if len(fp):
        ax.axhline(nd - 0.5, color=INK, lw=1.2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label='정상 생산 대비 z (±8 고정)')
    ax.set_title(f'{product.upper()} · {method} — 불량(● = pct≥{q}로 탐지)과 오탐은 어떤 변수에서 벗어났나',
                 loc='left', fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()
    return pd.DataFrame(M, index=lab, columns=[short(c) for c in cols])


def univariate_auc(d, cols, n_boot=300, seed=SEED):
    """변수 하나로 불량 vs 정상 AUC (+95% CI). 방향 무관하게 보려고 |AUC−0.5|도 같이"""
    rng = np.random.default_rng(seed)
    groups = {'전체불량': d['PassOrFail'] == 1, '가스': d['Reason'] == '가스',
              '미성형계열': d['Reason'].isin(['미성형', '가스+미성형'])}
    rows = []
    for gname, pos in groups.items():
        if pos.sum() < 2:
            continue
        ev = pos | (d['PassOrFail'] == 0)
        y = pos[ev].astype(int).values
        ip, ineg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        for c in cols:
            x = d.loc[ev, c].values.astype(float)
            ok = np.isfinite(x)
            if ok.sum() < 10 or np.ptp(x[ok]) == 0:
                continue
            x = np.where(ok, x, np.nanmedian(x))
            auc = roc_auc_score(y, x)
            bs = [roc_auc_score(y[i], x[i]) for i in
                  (np.r_[rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))] for _ in range(n_boot))]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append({'불량군': gname, '변수': c, '이름': short(c), 'n불량': len(ip), 'AUC': auc,
                         'lo': lo, 'hi': hi, '|AUC-0.5|': abs(auc - 0.5), '유의': (lo > 0.5) or (hi < 0.5)})
    return pd.DataFrame(rows).sort_values(['불량군', '|AUC-0.5|'], ascending=[True, False])


def plot_univariate_auc(t, product, top=15, save=None):
    """불량군별 상위 변수 AUC 점 + CI. 0.5선을 안 걸치면(채운 점) 변수 하나로도 의미 있는 차이"""
    set_style()
    gs = list(t['불량군'].unique())
    fig, axes = plt.subplots(1, len(gs), figsize=(5.2 * len(gs), 0.32 * top + 1.8))
    for ax, g in zip(np.atleast_1d(axes), gs):
        s = t[t['불량군'] == g].head(top).iloc[::-1]
        y = np.arange(len(s))
        col = TYPE_COL.get(g, TYPE_COL['불량'])
        ax.hlines(y, s['lo'], s['hi'], color='#d6d5cc', lw=2.5)
        ax.scatter(s['AUC'], y, s=44, color=np.where(s['유의'], col, 'white'), edgecolor=col, lw=1.5, zorder=3)
        ax.axvline(0.5, color=SUB, lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(s['이름'], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_title(f"{g} (n={int(s['n불량'].iloc[0])})", loc='left', fontsize=10)
        ax.set_xlabel('AUC (0.5 = 무관, 채운 점 = CI가 0.5 안 걸침)')
    fig.suptitle(f'{product.upper()} — 변수 하나만으로 불량을 얼마나 가르나', x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save:
        fig.savefig(save)
    plt.show()


def group_ablation(d, ref, groups, kind='knn'):
    """
    groups: {'묶음 이름': [열...]} — 기준 = 전체 묶음 합집합
    행: 전체 / −묶음(빼기) / 묶음만. AP·hit@10%(Run 단위)·유형별 hit
    ※ 이상탐지는 reference로만 학습 → 라벨은 평가에만 쓰임. 그래도 이 표를 보고 묶음을 고르면 선택 과적합이 생기니
      확정은 중첩 LORO로 하거나, 가설 근거가 있는 묶음만 채택할 것
    """
    allc = list(dict.fromkeys(sum(groups.values(), [])))
    sets = {'전체': allc}
    for g, cols in groups.items():
        rest = [c for c in allc if c not in cols]
        if rest:
            sets[f'− {g}'] = rest
        sets[f'{g}만'] = cols
    rows = []
    y = d['PassOrFail'].values
    for name, cols in sets.items():
        det = RefDetector(kind, cols).fit(ref)
        sc, _ = to_ref_scale(det.raw_score(d), det.s_ref)
        g = pd.DataFrame({'run_id': d['run_id'].values, 'score': sc, 'y': y, 'Reason': d['Reason'].values})
        h, t = recall_at_k_by_run(g, 0.10, g['y'] == 1)
        r = {'설정': name, '변수 수': len(cols), 'AP': average_precision_score(y, sc), 'hit@10%': f'{h}/{t}',
             'R@10%': h / t}
        for tn, rs in {'가스': ['가스'], '미성형계열': ['미성형', '가스+미성형']}.items():
            pos = g['Reason'].isin(rs)
            if pos.sum():
                hh, tt = recall_at_k_by_run(g, 0.10, pos)
                r[f'hit10_{tn}'] = f'{hh}/{tt}'
        rows.append(r)
    t = pd.DataFrame(rows)
    t['ΔAP(vs 전체)'] = t['AP'] - t.loc[t['설정'] == '전체', 'AP'].iloc[0]
    return t


def plot_group_ablation(t, product, kind='knn', save=None):
    set_style()
    s = t[t['설정'] != '전체'].iloc[::-1]
    base = t.loc[t['설정'] == '전체', 'AP'].iloc[0]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(s) + 1.6))
    col = np.where(s['설정'].str.startswith('−'), '#eb6834', '#2a78d6')
    ax.barh(y, s['AP'], color=col, height=0.6)
    ax.axvline(base, color=INK, ls='--', lw=1)
    ax.text(base, len(s) - 0.3, f' 전체 AP {base:.3f}', fontsize=8, color=SUB)
    for yi, (ap, h) in enumerate(zip(s['AP'], s['hit@10%'])):
        ax.text(ap, yi, f'  {ap:.3f} · hit {h}', va='center', fontsize=8, color=SUB)
    ax.set_yticks(y); ax.set_yticklabels(s['설정'], fontsize=9)
    ax.set_xlabel('AP')
    ax.set_title(f'{product.upper()} · {kind} — 변수 묶음 기여 (주황 = 그 묶음 뺌, 파랑 = 그 묶음만)', loc='left',
                 fontsize=10)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()


def logit_coef_stability(FEAT, cols, balance_products=True):
    """통합 logit을 LORO fold마다 학습 → 표준화 계수 분포. 부호 일치율 높고 0에서 멀면 안정적인 신호"""
    A = pd.concat([FEAT[p].assign(product=p) for p in FEAT], ignore_index=True)
    rows = []
    for r in A['run_id'].unique():
        tr = A[A['run_id'] != r]
        w = None
        if balance_products:
            cnt = tr['product'].map(tr['product'].value_counts())
            w = (len(tr) / tr['product'].nunique() / cnt).values
        m = make_sup('logit')
        m.fit(_X(tr, cols), tr['PassOrFail'].values, logisticregression__sample_weight=w)
        rows.append(pd.Series(m.named_steps['logisticregression'].coef_[0], index=cols, name=r))
    C = pd.DataFrame(rows)
    summ = pd.DataFrame({'이름': [short(c) for c in cols], 'median': C.median(), 'min': C.min(), 'max': C.max(),
                         '부호일치율': (np.sign(C) == np.sign(C.median())).mean()})
    return C, summ.reindex(summ['median'].abs().sort_values(ascending=False).index)


def plot_coef_stability(C, summ, top=15, save=None):
    set_style()
    s = summ.head(top).iloc[::-1]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(s) + 1.6))
    for yi, c in enumerate(s.index):
        v = C[c].values
        ax.scatter(v, np.full(len(v), yi), s=14, color=MUTED, alpha=0.7, zorder=2)
        col = '#e34948' if s.at[c, 'median'] > 0 else '#2a78d6'
        ax.scatter(s.at[c, 'median'], yi, s=60, color=col, edgecolor='white', lw=1.5, zorder=3)
    ax.axvline(0, color=SUB, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  ({r:.0%})" for n, r in zip(s['이름'], s['부호일치율'])], fontsize=8)
    ax.set_xlabel('표준화 계수 (회색 = fold별, 큰 점 = 중앙값;  빨강 + = 불량 위험↑)')
    ax.set_title('통합 logit 계수 안정성 — 괄호 = fold 간 부호 일치율', loc='left', fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save)
    plt.show()
