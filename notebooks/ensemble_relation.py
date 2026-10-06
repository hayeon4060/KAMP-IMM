"""
ensemble_relation.py — 36번: 수치를 올릴 수 있는 두 가지 새 방법 (공정 데이터만 사용, 모두 사전 고정)

왜 이 두 가지인가
  ① 공식 2차 CN7 AP 0.373 ≈ 미성형 6행을 1~6위에 올렸을 때의 몫(6/17 = 0.353) + 가스 약간
  → 미성형은 이미 거의 다 잡음. 더 올리려면 '가스'를 잡아야 함 (LH/RH 쌍 제약까지 넣은 이론 상한 ≈ 0.80)

A. 두 전문가 앙상블 — 유형마다 다른 신호를 보는 점수를 따로 만들고 max로 합침 (가중치 튜닝 없음)
   - 미성형 전문가 = 최종 모델 pct (샷 순간 이벤트 편차)
   - 가스 전문가 (제품별로 다름 — CN7과 RG3는 따로 봄)
       CN7: 온도 교란 점수 E (호퍼 기울기↑ + 금형 편차↓ + 0.5·호퍼 편차↑, 최근 5샷 평균), 재가동 후 15샷 제외 (29번 규칙)
       RG3: 흔들림 std4 (배럴2·호퍼·쿠션·배럴5 직전 10샷 std) Mahalanobis (하연님 신호 + 34번 확인)
   - 합치기: max(미성형 pct, 가스 pct) → 정상 reference에서 같은 합치기 점수의 백분위로 다시 보정 (경보율 유지)
   ⚠ CN7 E 규칙의 방향은 R1을 보고 정했음 → CN7 개선분은 사후 확인 성격. RG3 std4는 하연님 사전 가설

B. 변수 간 관계 이탈 (가상계측 잔차) — 새로운 비지도 방식
   - 정상 reference(앞 80% Run)에서 원값 21개 각각을 '같은 샷의 나머지 20개 + 자기 직전 2샷 + 자기 직전 20샷 중앙값'으로 예측
     (HistGradientBoosting 회귀, 변수마다 1개)
   - 잔차 = 실제 − 예측 → 검증 reference(뒤 20% Run)의 robust 척도로 표준화 → ±20
   - 점수: 잔차 Mahalanobis(B_rel_maha), 잔차 최대 |z|(B_rel_max) / 샷마다 잔차가 가장 큰 변수도 기록(설명용)
   - 생각: 가스(수분·휘발분)는 각 변수는 정상 범위라도 '같은 속도·온도에서 사출압·쿠션이 평소와 다른' 관계 이탈로 나타날 수 있음
           (최종 모델의 흐름저항 = 사출압/속도 하나를 21개 변수 전체 관계로 확장)

비교 (모두 같은 평가 함수: ① 공식 2차 행 / ② labeled 전체 샷, 초기허용 제외 / q 임계값 / 최종 대비 ΔAP)
  final | A_gas_expert | A_max(final, gas) | B_rel_maha | B_rel_max | B_max(final, B_rel_maha) | AB_max(final, gas, B_rel_maha)
  + ① 상위 k%에 들어온 행을 유형별(미성형/가스)로 분리 → '가스가 들어왔는가'를 직접 확인
⚠ 비교 대상이 7개 → 하나가 우연히 좋아 보일 수 있음. ΔAP 95% CI가 0을 넘는지로 판단
"""
import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingRegressor

from final_model import (ALL_RAW, raw, SEED, CLIP, TYPE_GROUPS, load_labeled_shots, load_official_rows, resolution,
                         apply_exclusions)
from seq_ae import ensure_inputs, split_reference, evaluate, delta_ap, eval_mask, to_pct

STOP_GAP_SEC, N_EXCL, E_WIN = 5 * 60, 15, 5
STD4 = ['Barrel_Temperature_2', 'Hopper_Temperature', 'Cushion_Position', 'Barrel_Temperature_5']
BASE = 'final (최종)'


# ====================================================================== 데이터
def load(p, base_dir='.', model_dir='./outputs/final_model'):
    """reference(학습 데이터, 원값+17변수+최종 점수) / labeled(원값+17변수+최종 점수), Run→시간 순"""
    ensure_inputs(p, base_dir, model_dir)
    ref = pd.read_csv(f'{model_dir}/{p}_train_features.csv', parse_dates=['TimeStamp'])
    lab = pd.read_csv(f'{model_dir}/{p}_scores.csv', parse_dates=['TimeStamp'])
    lab['Reason'] = lab['Reason'].fillna('정상')
    lab = apply_exclusions(lab, p)                                   # RG3 R6 등 평가 제외 Run
    rcols = [raw(v) for v in ALL_RAW if raw(v) in ref.columns]
    shots = load_labeled_shots(p, base_dir)[['TimeStamp'] + rcols]
    lab = lab.merge(shots, on='TimeStamp', how='left')
    ref = ref.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    lab = lab.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    for d in (ref, lab):
        g = d.groupby('run_id', sort=False)
        gap = g['TimeStamp'].diff().dt.total_seconds()
        restart = gap.isna() | (gap > STOP_GAP_SEC)
        seg = restart.groupby(d['run_id']).cumsum()
        d['since_restart'] = d.groupby([d['run_id'], seg]).cumcount() + 1
        d['seq'] = g.cumcount() + 1
    return ref, lab, rcols


# ====================================================================== A. 가스 전문가
def gas_expert_cn7(ref, lab):
    """온도 교란 점수 E (29번 규칙). 재가동 후 15샷 이내는 판단 보류(점수 최저)"""
    hs = ref['hopper_slope20'].dropna()
    med, iqr = hs.median(), max(hs.quantile(0.75) - hs.quantile(0.25), 1e-6)

    def E(d):
        c = (((d['hopper_slope20'] - med) / iqr).fillna(0).clip(0, 10)
             + (-d['dev_Mold_Temperature_3'].fillna(0)).clip(0, 10)
             + 0.5 * d['dev_Hopper_Temperature'].fillna(0).clip(0, 20))
        e = c.groupby(d['run_id'], sort=False).transform(lambda s: s.rolling(E_WIN, min_periods=1).mean())
        return e.where(d['since_restart'] > N_EXCL, -np.inf).values

    return E(ref), E(lab)


def _std10(d, v):
    return d[raw(v)].groupby(d['run_id'], sort=False).transform(lambda s: s.rolling(10, min_periods=3).std())


def _robust_fit(X):
    med = np.nanmedian(X, 0)
    q75, q25 = np.nanpercentile(X, [75, 25], 0)
    sd = np.nanstd(X, 0)
    iqr = np.where(q75 - q25 > 1e-9, q75 - q25, np.where(sd > 1e-9, sd, 1.0))
    return lambda Z: np.nan_to_num(np.clip((Z - med) / iqr, -CLIP, CLIP), nan=0.0)


def gas_expert_rg3(ref, lab):
    """흔들림 std4 Mahalanobis (정상 reference 학습)"""
    Xr = np.column_stack([_std10(ref, v) for v in STD4])
    Xl = np.column_stack([_std10(lab, v) for v in STD4])
    z = _robust_fit(Xr)
    lw = LedoitWolf().fit(z(Xr))
    return np.sqrt(lw.mahalanobis(z(Xr))), np.sqrt(lw.mahalanobis(z(Xl)))


# ====================================================================== B. 관계 이탈 (가상계측 잔차)
def _design(d, rcols, v):
    """변수 v 예측 입력: 같은 샷의 나머지 원값 + v의 직전 1·2샷 + v의 직전 20샷 중앙값 (모두 같은 Run 과거만)"""
    x = d[raw(v)]
    g = x.groupby(d['run_id'], sort=False)
    X = d[[c for c in rcols if c != raw(v)]].copy()
    X['self_lag1'] = g.shift(1)
    X['self_lag2'] = g.shift(2)
    X['self_med20'] = g.transform(lambda s: s.shift(1).rolling(20, min_periods=3).median())
    return X


class RelationModel:
    def fit(self, ref, is_val, rcols, verbose=False):
        self.rcols = rcols
        self.models, self.scale, self.targets = {}, {}, []
        tr = ~is_val
        for v in [c.replace('_raw', '') for c in rcols]:
            y = ref[raw(v)]
            if y[tr].nunique() < 3:
                continue
            X = _design(ref, rcols, v)
            ok = tr & y.notna().values
            m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=20, random_state=SEED)
            m.fit(X[ok], y[ok])
            res = (y - m.predict(X))[is_val & y.notna().values]
            mad = 1.4826 * np.median(np.abs(res - np.median(res)))
            self.scale[v] = max(mad, resolution(y), 1e-6)
            self.models[v] = m
            self.targets.append(v)
            if verbose:
                print(f'    관계 모델 {v}: 검증 잔차 척도 {self.scale[v]:.4g}')
        Zv = self.residual_z(ref[is_val])
        self.lw = LedoitWolf().fit(Zv)
        return self

    def residual_z(self, d):
        Z = np.column_stack([((d[raw(v)] - self.models[v].predict(_design(d, self.rcols, v))) / self.scale[v]).values
                             for v in self.targets])
        return np.nan_to_num(np.clip(Z, -CLIP, CLIP), nan=0.0)

    def scores(self, d):
        Z = self.residual_z(d)
        top = np.array(self.targets)[np.abs(Z).argmax(1)]
        return np.sqrt(self.lw.mahalanobis(Z)), np.abs(Z).max(1), top, Z


# ====================================================================== 평가 보조
def frame1_by_type(p, lab, score, base_dir='.', ks=(1, 5, 10)):
    """① 공식 2차 행: 상위 k%에 들어온 불량 행을 유형별로"""
    rows = load_official_rows(p, base_dir)
    s = rows['TimeStamp'].map(pd.Series(np.asarray(score, float), index=lab['TimeStamp'].values)).fillna(-np.inf).values
    o = np.lexsort((np.random.RandomState(0).rand(len(s)), -s))
    gas = rows['Reason'].isin(TYPE_GROUPS['가스']).values & (rows['PassOrFail'] == 1).values
    sh = rows['Reason'].isin(TYPE_GROUPS['미성형']).values & (rows['PassOrFail'] == 1).values
    out = {}
    for k in ks:
        top = o[:max(1, int(len(s) * k / 100))]
        out[f'① 상위{k}% 미성형행'] = f'{int(sh[top].sum())}/{int(sh.sum())}'
        out[f'① 상위{k}% 가스행'] = f'{int(gas[top].sum())}/{int(gas.sum())}'
    return out


def _fuse(*pcts):
    return np.max(np.column_stack(pcts), axis=1)


# ====================================================================== 실행
def run(products=('cn7', 'rg3'), base_dir='.', model_dir='./outputs/final_model', out_dir='./outputs/ensemble_relation',
        verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    F1, F2, TH, SUM, SCORES = [], [], [], [], {}
    for p in products:
        ref, lab, rcols = load(p, base_dir, model_dir)
        is_val = split_reference(ref)
        if verbose:
            print(f'[{p}] reference {len(ref):,}샷 (관계모델 학습 {int((~is_val).sum()):,} / 검증 {int(is_val.sum()):,}) '
                  f'| labeled {len(lab):,}샷')

        # 최종 모델 (미성형 전문가)
        fin_l, fin_r = lab['score'].values, ref['score'].values
        pf_l, pf_r = lab['pct'].values, ref['pct'].values

        # A. 가스 전문가
        g_r, g_l = gas_expert_cn7(ref, lab) if p == 'cn7' else gas_expert_rg3(ref, lab)
        g_ref_ok = g_r[np.isfinite(g_r)]
        pg_l, pg_r = to_pct(np.sort(g_ref_ok), g_l), to_pct(np.sort(g_ref_ok), g_r)

        # B. 관계 이탈
        rm = RelationModel().fit(ref, is_val, rcols, verbose=False)
        bm_l, bx_l, top_l, _ = rm.scores(lab)
        bm_r, bx_r, _, _ = rm.scores(ref[is_val])
        pbm_l, pbm_r = to_pct(np.sort(bm_r), bm_l), to_pct(np.sort(bm_r), bm_r)
        pbx_l, pbx_r = to_pct(np.sort(bx_r), bx_l), to_pct(np.sort(bx_r), bx_r)

        # 합치기: labeled 점수 / reference 점수(검증 Run) → 합친 점수의 reference 백분위로 보정
        v = is_val
        cand = {
            BASE: (fin_l, pf_l),
            'A_gas_expert': (np.where(np.isfinite(g_l), g_l, np.nanmin(g_ref_ok) - 1), pg_l),
            'A_max(final,gas)': _calib(_fuse(pf_l, pg_l), _fuse(pf_r[v], pg_r[v])),
            'B_rel_maha': (bm_l, pbm_l),
            'B_rel_max': (bx_l, pbx_l),
            'B_max(final,B)': _calib(_fuse(pf_l, pbm_l), _fuse(pf_r[v], pbm_r)),
            'AB_max(final,gas,B)': _calib(_fuse(pf_l, pg_l, pbm_l), _fuse(pf_r[v], pg_r[v], pbm_r)),
        }
        out = lab[['TimeStamp', 'run_id', 'seq', 'Reason', 'PassOrFail']].copy()
        out['eval'] = eval_mask(lab).values
        out['B_최대잔차변수'] = top_l
        g0 = None
        for name, (sc, pc) in cand.items():
            f1, f2, th, g = evaluate(p, lab, sc, pc, name, base_dir)
            F1.append(f1); F2.append(f2); TH.append(th)
            if name == BASE:
                g0 = g
            dl = None if name == BASE else delta_ap(g['y'].values, g['score'].values, g0['score'].values)
            SUM.append(_row(p, name, f1, f2, th, dl, frame1_by_type(p, lab, sc, base_dir)))
            out[f'{name}|pct'] = pc
        SCORES[p] = out
        out.to_csv(f'{out_dir}/{p}_shot_scores.csv', index=False, encoding='utf-8-sig')
        if verbose:
            print(f'  [{p}] 완료', flush=True)
    R = {'summary': pd.DataFrame(SUM), 'frame1': pd.DataFrame(F1), 'frame2': pd.DataFrame(F2),
         'threshold': pd.concat(TH, ignore_index=True), 'scores': SCORES}
    for k in ('summary', 'frame1', 'frame2', 'threshold'):
        R[k].to_csv(f'{out_dir}/{k}.csv', index=False, encoding='utf-8-sig')
    return R


def _calib(fused_lab, fused_ref):
    """합친 점수 → 정상 reference(검증 Run)의 같은 합친 점수 분포 대비 백분위"""
    return fused_lab, to_pct(np.sort(fused_ref), fused_lab)


def _row(p, name, f1, f2, th, dl, bytype):
    t = th.set_index('q')
    r = {'제품': p, '모델': name, '① AP': f1['AP'], '① 상위5%': f1['상위5%'], '① 상위10%': f1['상위10%'], **bytype,
         '② AP': f2['AP'], '② AP 95%CI': f'[{f2["AP_lo"]:.3f}, {f2["AP_hi"]:.3f}]',
         '② AP_가스': f2.get('AP_가스', np.nan), '② AP_미성형': f2.get('AP_미성형', np.nan),
         'Run별 상위10% 가스': f2.get('Run별 상위10%_가스', ''), 'Run별 상위10% 미성형': f2.get('Run별 상위10%_미성형', ''),
         'q0.99 경보': int(t.loc[0.99, '경보 수']), 'q0.99 경보 비율': float(t.loc[0.99, '경보 비율']),
         'q0.99 가스': int(t.loc[0.99, '가스 탐지']), 'q0.99 미성형': int(t.loc[0.99, '미성형계열 탐지']),
         'q0.999 경보': int(t.loc[0.999, '경보 수']), 'q0.999 TP': int(t.loc[0.999, 'TP']),
         'q0.999 가스': int(t.loc[0.999, '가스 탐지']), 'q0.999 미성형': int(t.loc[0.999, '미성형계열 탐지'])}
    if dl is not None:
        r.update({'ΔAP(②, vs 최종)': dl[0], 'ΔAP 95%CI': f'[{dl[1]:+.3f}, {dl[2]:+.3f}]', 'P(최종보다 좋음)': dl[3],
                  '유의하게 좋음': 'O' if dl[1] > 0 else '-'})
    return r


# ====================================================================== 그림
def plot_run(R, p='cn7', models=None, save=None):
    """가스가 가장 많은 Run의 샷별 pct — 미성형·가스 전문가와 합친 점수가 어디서 올라오는지"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    d = R['scores'][p]
    gas = d[d['Reason'].isin(TYPE_GROUPS['가스'])]
    if gas.empty:
        print('가스 샷 없음'); return
    run = gas['run_id'].mode().iloc[0]
    r = d[d['run_id'] == run].reset_index(drop=True)
    models = models or [BASE, 'A_gas_expert', 'B_rel_maha', 'AB_max(final,gas,B)']
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for mname, c, lw in zip(models, ['k', '#e67e22', '#2a9d8f', '#1f4e79'], [1.4, 1.1, 1.1, 1.6]):
        ax.plot(r['seq'], r[f'{mname}|pct'], color=c, lw=lw, label=mname)
    for reasons, c, lab_ in ((TYPE_GROUPS['가스'], '#e67e22', '가스'), (TYPE_GROUPS['미성형'], '#1f4e79', '미성형계열')):
        idx = r.loc[r['Reason'].isin(reasons), 'seq']
        ax.scatter(idx, np.full(len(idx), 1.03), color=c, marker='v', s=24, label=lab_)
    ax.set_ylim(0, 1.07); ax.set_xlabel(f'{run} 안 샷 순서'); ax.set_ylabel('정상 대비 백분위')
    ax.set_title(f'{p.upper()} 가스 집중 Run — 전문가별 점수', loc='left', fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
