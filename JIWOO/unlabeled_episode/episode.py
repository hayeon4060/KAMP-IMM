"""
episode.py — 비라벨 우진2 CN7·RG3 전체를 최종 모델(dev26 + 국소 Mahalanobis k=300)로 점수화하고
10/16형 '전조 → 사건' 과정을 찾는다 (정의: RESULT.md 상단, 실행 전 고정).
출력: outputs/*.csv (git 제외), fig/similar_cn7.png
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'unsup_search'))
sys.path.insert(0, str(ROOT.parent / 'final' / 'src'))
import common as C  # noqa: E402
import slices as SL  # noqa: E402
import models as M  # noqa: E402

OUT = ROOT / 'outputs'
FIG = ROOT / 'fig'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
K = 300
COLS = C.FEAT_DEV + C.FEAT_REL                              # dev26
F_SET = ['dv_Max_Injection_Speed', 'dv_Injection_Time', 'dv_Filling_Time', 'dv_rel_fill_speed']
P_SET = ['dv_rel_plast', 'dv_Barrel_Temperature_1', 'dv_Barrel_Temperature_2', 'dv_Barrel_Temperature_3',
         'dv_Hopper_Temperature']
PRE_WIN = 30
KO = {'Injection_Time': '사출시간', 'Filling_Time': '충전시간', 'Plasticizing_Time': '계량시간', 'Cycle_Time': '사이클',
      'Clamp_Close_Time': '형체시간', 'Cushion_Position': '쿠션', 'Plasticizing_Position': '계량위치',
      'Max_Injection_Speed': '사출속도', 'Max_Screw_RPM': '최대RPM', 'Max_Injection_Pressure': '사출압',
      'Max_Switch_Over_Pressure': '절환압', 'Average_Back_Pressure': '배압', 'Hopper_Temperature': '호퍼',
      'Mold_Temperature_3': '금형3', 'Mold_Temperature_4': '금형4', 'rel_fill_speed': 'R1', 'rel_plast': 'R2',
      'rel_hold': 'R3', 'rel_sw_ratio': 'R4', 'rel_rest': 'R5'}
KO.update({f'Barrel_Temperature_{i}': f'배럴{i}' for i in range(1, 7)})
ko = lambda c: KO.get(c[3:], c)                             # noqa: E731
log = C.log


# ====================================================================== 점수 (국소 Mahalanobis, 자기 제외)
def _score_chunk(A, nb_idx, Q, self_pos):
    s = np.empty(len(Q))
    T = np.empty_like(Q)
    for i, q in enumerate(Q):
        nb = nb_idx[i]
        if self_pos[i] >= 0:
            nb = nb[nb != self_pos[i]]
        nb = nb[:K]
        lw = LedoitWolf().fit(A[nb])
        z = q - lw.location_
        t = z * (lw.precision_ @ z)
        T[i] = t
        s[i] = np.sqrt(max(t.sum(), 0.0))
    return s, T


def score(A, Q, self_pos, n_jobs=15):
    nn = NearestNeighbors(n_neighbors=K + 1).fit(A)
    idx = nn.kneighbors(Q, return_distance=False)
    parts = np.array_split(np.arange(len(Q)), n_jobs)
    res = Parallel(n_jobs=n_jobs)(delayed(_score_chunk)(A, idx[p], Q[p], self_pos[p]) for p in parts)
    s = np.concatenate([r[0] for r in res])
    T = np.vstack([r[1] for r in res])
    ratio = T / np.where(np.abs(T.sum(1, keepdims=True)) > 1e-12, T.sum(1, keepdims=True), 1.0)
    return s, ratio


# ====================================================================== 구간·패턴
def segments(flag, run):
    """같은 Run에서 연속 True 묶음 → [(시작, 끝 포함)]"""
    out, i, n = [], 0, len(flag)
    while i < n:
        if flag[i]:
            j = i
            while j + 1 < n and flag[j + 1] and run[j + 1] == run[i]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def top_k(vec, k=3):
    o = np.argsort(-vec)[:k]
    return [COLS[i] for i in o]


def analyse(d, s, R, q95, q99, q999, product, ref=None):
    """d: 쇼트 표(Run→시간 정렬, dv 열 포함). 반환: 경보 구간 표, 유사도 후보 표"""
    run = d['run_id'].values
    run_start = d.groupby('run_id').cumcount().values
    dv = d[COLS].values
    top1 = np.array([COLS[i] for i in np.argmax(R, 1)])
    pre_flag = (s > q95) & np.isin(top1, P_SET)

    def precursor(a):
        lo = a - min(PRE_WIN, run_start[a])
        w = np.arange(lo, a)
        return int(pre_flag[w].sum()), w

    rows = []
    for a, b in segments(s > q999, run):
        mc = R[a:b + 1].mean(0)
        t3 = top_k(mc)
        nF = sum(v in F_SET for v in t3)
        per_shot = int(sum(sum(v in F_SET for v in top_k(R[i])) >= 2 for i in range(a, b + 1)))
        npre, _ = precursor(a)
        L = b - a + 1
        kind = ('사건' if L >= 3 else '단발 사건') if nF >= 2 else ('경보 3연속(패턴 불일치)' if L >= 3 else '')
        rows.append({'product': product, 'start': d['TimeStamp'].iloc[a], 'end': d['TimeStamp'].iloc[b],
                     'run_id': run[a], 'len': L, 'kind': kind, 'max_score': float(s[a:b + 1].max()),
                     'top3': ', '.join(f'{ko(v)} {mc[COLS.index(v)] * 100:.0f}% ({dv[a:b + 1, COLS.index(v)].mean():+.1f}σ)'
                                       for v in t3),
                     'F_in_top3': nF, 'shots_F_pattern': per_shot, 'precursor_shots': npre,
                     'precursor': npre >= 2, 'i0': a, 'i1': b})
    seg = pd.DataFrame(rows)
    # 전조형 구간(사건 없음): 전조 쇼트끼리 30쇼트 이내로 이어진 묶음 중 2개 이상, 뒤 30쇼트 안에 경보 구간 없음
    pre_idx = np.flatnonzero(pre_flag)
    clusters, cur = [], []
    for i in pre_idx:
        if cur and (run[i] != run[cur[-1]] or i - cur[0] >= PRE_WIN):
            clusters.append(cur)
            cur = []
        cur.append(i)
    if cur:
        clusters.append(cur)
    alarm_starts = set(seg['i0']) if len(seg) else set()
    lonely = 0
    for c in clusters:
        if len(c) >= 2 and not any((c[-1] < a <= c[-1] + PRE_WIN) and run[a] == run[c[-1]] for a in alarm_starts):
            lonely += 1
    # 유사도 후보 (99% 연속 묶음)
    sims = []
    if ref is not None:
        cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))   # noqa: E731
        for a, b in segments(s > q99, run):
            mc, md = R[a:b + 1].mean(0), np.nan_to_num(dv[a:b + 1]).mean(0)
            sE = 0.5 * cos(mc, ref['E_c']) + 0.5 * cos(md, ref['E_d'])
            _, w = precursor(a)
            w = w[s[w] > q95]
            sP = cos(R[w].mean(0), ref['P_c']) if len(w) else 0.0
            L = b - a + 1
            sims.append({'product': product, 'start': d['TimeStamp'].iloc[a], 'run_id': run[a], 'len': L,
                         'n_alarm': int((s[a:b + 1] > q999).sum()), 'max_score': float(s[a:b + 1].max()),
                         's_E': sE, 's_P': sP, 'pre_q95_shots': len(w),
                         'similarity': 0.5 * sE + 0.3 * sP + 0.2 * min(L, 3) / 3,
                         'top3': ', '.join(f'{ko(v)} {mc[COLS.index(v)] * 100:.0f}% ({md[COLS.index(v)]:+.1f}σ)'
                                           for v in top_k(mc)), 'i0': a, 'i1': b})
    return seg, pd.DataFrame(sims), lonely, top1


# ====================================================================== 로트 불량
def lot_table():
    U = pd.read_csv(C.DATA / 'unlabeled_data.csv', low_memory=False,
                    usecols=['TimeStamp', 'EQUIP_NAME', 'PART_NAME', 'PART_FACT_PLAN_DATE', 'PART_FACT_SERIAL',
                             'ERR_FACT_QTY'])
    U = U[U['EQUIP_NAME'] == C.EQUIP_W2].copy()
    U['prod'] = U['PART_NAME'].str[:3]
    U['side'] = U['PART_NAME'].str[-2:]
    U['lot'] = U['PART_FACT_PLAN_DATE'].astype(str).str[:10] + '_' + U['PART_FACT_SERIAL'].astype(str)
    U['TimeStamp'] = pd.to_datetime(U['TimeStamp'])
    U['day'] = U['TimeStamp'].dt.date
    lot_err = U.groupby(['prod', 'lot', 'side'])['ERR_FACT_QTY'].first().unstack('side')
    day_lots = U.groupby(['prod', 'day'])['lot'].unique()
    day = {}
    for (pr, dy), lots in day_lots.items():
        e = lot_err.loc[pr].reindex(lots)
        day[(pr, dy)] = {'LH': e['LH'].sum(), 'RH': e['RH'].sum(), 'n_lots': len(lots)}
    shot_lot = U.drop_duplicates(['prod', 'TimeStamp'])[['prod', 'TimeStamp', 'lot']]
    return lot_err, pd.DataFrame(day).T, shot_lot


# ====================================================================== 그림
def figure(panels, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT.parent / 'final' / 'src'))
    import plot_font
    plot_font.setup(9)
    col = {'사건형(F)': '#2a78d6', '전조형(P)': '#eb6834', '기타': '#9a9a96'}
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.3 * len(panels)), sharey=True)
    for ax, pn in zip(axes, panels):
        x = np.arange(len(pn['s'])) - pn['offset']
        g = np.where(np.isin(pn['top1'], F_SET), '사건형(F)', np.where(np.isin(pn['top1'], P_SET), '전조형(P)', '기타'))
        ax.plot(x, pn['s'], color='#c3c2b7', lw=1, zorder=1)
        for k, c in col.items():
            m = g == k
            ax.scatter(x[m], pn['s'][m], s=22, color=c, label=k, zorder=3, edgecolors='white', linewidths=0.6)
        if 'defect' in pn:
            ax.scatter(x[pn['defect']], pn['s'][pn['defect']], s=70, facecolors='none', edgecolors='#e34948',
                       linewidths=1.2, label='라벨 불량 쇼트', zorder=4)
        ax.axhline(pn['q999'], color='#4a4a48', lw=1, ls='--')
        ax.axhline(pn['q95'], color='#4a4a48', lw=0.8, ls=':')
        ax.text(x[0], pn['q999'], ' 경보선(99.9%)', va='bottom', fontsize=8, color='#4a4a48')
        ax.text(x[0], pn['q95'], ' 상위 5% 선', va='bottom', fontsize=8, color='#4a4a48')
        ax.set_yscale('log')
        ax.set_title(pn['title'], loc='left', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#ebeae4', lw=0.6)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, [f'기여 1위: {x}' if x != '라벨 불량 쇼트' else x for x in lab], loc='upper center', ncol=4,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    axes[-1].set_xlabel('구간 첫 쇼트 기준 쇼트 위치 (같은 Run)')
    for ax in axes:
        ax.set_ylabel('이상 점수')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ====================================================================== 실행
def main():
    F = C.build_cache()
    LAB = C.build_labeled_feats()
    lot_err, day_err, shot_lot = lot_table()
    summary, all_seg, all_sim, ref, panels_data = {}, [], [], None, {}
    for p in ('cn7', 'rg3'):
        P = p.upper()
        d = F[(F['equip'] == C.EQUIP_W2) & (F['prod'] == P)].sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
        train = (d['TimeStamp'] < SL.LAB_START[p]).values
        X = d[COLS].replace([np.inf, -np.inf], np.nan).values.astype(float)
        pr = M.Prep().fit(X[train])
        A, Q = pr(X[train]), pr(X)
        self_pos = np.full(len(d), -1)
        self_pos[train] = np.arange(train.sum())
        log(f'{p}: 비라벨 {len(d):,}쇼트 (학습 {train.sum():,}) 점수화')
        s, R = score(A, Q, self_pos)
        q95, q99, q999 = np.quantile(s[train], [0.95, 0.99, 0.999])
        # 양성 대조 (라벨 쇼트, 학습 쇼트가 아니므로 자기 제외 없음)
        lab = LAB[p].sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
        Xl = lab[COLS].replace([np.inf, -np.inf], np.nan).values.astype(float)
        sl, Rl = score(A, pr(Xl), np.full(len(lab), -1))
        if p == 'cn7':
            ev_ts = pd.to_datetime(['2020-10-16 05:55:21', '2020-10-16 05:56:22', '2020-10-16 05:57:20'])
            ei = np.flatnonzero(lab['TimeStamp'].isin(ev_ts).values)
            a = ei[0]
            run_start = lab.groupby('run_id').cumcount().values
            w = np.arange(a - min(PRE_WIN, run_start[a]), a)
            w = w[sl[w] > q95]
            ref = {'E_c': Rl[ei].mean(0), 'E_d': np.nan_to_num(lab.loc[ei, COLS].values).mean(0), 'P_c': Rl[w].mean(0)}
        seg_l, sim_l, lonely_l, top1_l = analyse(lab, sl, Rl, q95, q99, q999, f'{p}_라벨', ref)
        seg, sim, lonely, top1 = analyse(d, s, R, q95, q99, q999, p, ref)
        # 기간별 경보
        alarm = s > q999
        per = pd.Series(np.where(train, '학습 기간 안', '학습 기간 밖(라벨 시작 이후)'))
        st = pd.DataFrame({'per': per, 'alarm': alarm, 'top5': s > q95}).groupby('per').agg(
            shots=('alarm', 'size'), alarm=('alarm', 'sum'), alarm_rate=('alarm', 'mean'), top5_rate=('top5', 'mean'))
        mon = pd.DataFrame({'m': d['TimeStamp'].dt.to_period('M').astype(str), 'alarm': alarm}).groupby('m').agg(
            shots=('alarm', 'size'), alarm=('alarm', 'sum'), rate=('alarm', 'mean'))
        summary[p] = {'n': len(d), 'n_train': int(train.sum()), 'q95': q95, 'q99': q99, 'q999': q999,
                      'period': st.reset_index().to_dict('records'), 'month': mon.reset_index().to_dict('records'),
                      'alarm_segments': len(seg), 'alarm_3plus': int((seg['len'] >= 3).sum()) if len(seg) else 0,
                      'lonely_precursor_clusters': lonely,
                      'positive_control_segments': seg_l.drop(columns=['i0', 'i1']).to_dict('records') if len(seg_l) else [],
                      'positive_control_lonely': lonely_l}
        # 로트 불량 (사건·단발 사건·유사도 상위)
        def lot_info(ts):
            dy = pd.Timestamp(ts).date()
            de = day_err.loc[P] if P in day_err.index.get_level_values(0) else None
            row = {}
            if de is not None and dy in de.index:
                for sd in ('LH', 'RH'):
                    oth = de.drop(index=dy)[sd].astype(float)
                    v = float(de.loc[dy, sd])
                    row[f'day_{sd}'] = v
                    row[f'other_days_mean_{sd}'] = float(oth.mean())
                    row[f'day_pct_{sd}'] = float((oth < v).mean())
            lt = shot_lot[(shot_lot['prod'] == P) & (shot_lot['TimeStamp'] == pd.Timestamp(ts))]['lot']
            if len(lt):
                le = lot_err.loc[P]
                for sd in ('LH', 'RH'):
                    row[f'lot_{sd}'] = float(le.loc[lt.iloc[0], sd])
                    row[f'all_lots_mean_{sd}'] = float(le[sd].mean())
            return row
        if len(seg):
            seg = pd.concat([seg, pd.DataFrame([lot_info(t) for t in seg['start']])], axis=1)
        if len(sim):
            sim = sim.sort_values('similarity', ascending=False).reset_index(drop=True)
            sim = pd.concat([sim, pd.DataFrame([lot_info(t) for t in sim['start']])], axis=1)
        all_seg.append(seg)
        all_sim.append(sim)
        panels_data[p] = (d, s, top1, q95, q999, sim)
        if p == 'cn7':
            panels_data['ref'] = (lab, sl, top1_l, q95, q999, ei[0])
        log(f'{p}: 경보선 {q999:.2f}, 경보 구간 {len(seg)}개 (3연속 이상 {summary[p]["alarm_3plus"]}), '
            f'사건 {int((seg["kind"] == "사건").sum()) if len(seg) else 0}, '
            f'단발 {int((seg["kind"] == "단발 사건").sum()) if len(seg) else 0}, 전조형 구간(사건 없음) {lonely}')
        log(f'  양성 대조(라벨): ' + '; '.join(f'{r["start"]:%m-%d %H:%M:%S} 길이 {r["len"]} {r["kind"] or "패턴 불일치"} '
                                         f'전조 {r["precursor_shots"]}' for r in summary[p]['positive_control_segments']))
    SEG = pd.concat(all_seg, ignore_index=True)
    SIM = pd.concat(all_sim, ignore_index=True)
    SEG.drop(columns=['i0', 'i1']).to_csv(OUT / 'alarm_segments.csv', index=False, encoding='utf-8-sig')
    SIM.drop(columns=['i0', 'i1']).to_csv(OUT / 'similarity_rank.csv', index=False, encoding='utf-8-sig')
    json.dump(summary, open(OUT / 'summary.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    # 그림: 10/16 기준 + CN7 유사도 상위 3
    lab, sl, t1l, q95, q999, a = panels_data['ref']
    rs = lab['run_id'].values == lab['run_id'].values[a]
    idx = np.flatnonzero(rs)
    lo, hi = max(idx[0], a - 40), min(idx[-1], a + 15)
    defect = (lab['PassOrFail'].values[lo:hi + 1] == 1)
    panels = [{'s': sl[lo:hi + 1], 'top1': t1l[lo:hi + 1], 'offset': a - lo, 'q95': q95, 'q999': q999,
               'defect': defect, 'title': f'기준: 10/16 라벨 사건 (05:55:21 시작, 빨간 원 = 라벨 불량 쇼트)'}]
    d, s, top1, q95, q999, sim = panels_data['cn7']
    for r in sim.head(3).itertuples():
        a = r.i0
        idx = np.flatnonzero(d['run_id'].values == r.run_id)
        lo, hi = max(idx[0], a - 40), min(idx[-1], a + 15)
        panels.append({'s': s[lo:hi + 1], 'top1': top1[lo:hi + 1], 'offset': a - lo, 'q95': q95, 'q999': q999,
                       'title': f'유사도 {r.similarity:.2f}: {pd.Timestamp(r.start):%Y-%m-%d %H:%M:%S} '
                                f'(길이 {r.len}, 경보 {r.n_alarm}쇼트, 직전 상위5% {r.pre_q95_shots}쇼트)'})
    figure(panels, FIG / 'similar_cn7.png')
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 90)
    print(SEG.drop(columns=['i0', 'i1']).to_string(index=False))
    for p in ('cn7', 'rg3'):
        print(f'\n== {p} 유사도 상위 10')
        print(SIM[SIM['product'] == p].head(10).drop(columns=['i0', 'i1']).round(3).to_string(index=False))
    log('완료')


if __name__ == '__main__':
    main()
