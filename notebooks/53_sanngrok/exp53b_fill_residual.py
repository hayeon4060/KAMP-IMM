import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))); _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))), '37_search'))  # notebooks 폴더에서 실행: python 53_sanngrok/exp53b_fill_residual.py
"""
53-B. sanngrok의 '정상 공정관계 잔차'(P7)가 우리 최종 모델(M_maha_free)과 겹치는지

잔차 계산 (sanngrok NormalRelations와 같은 식, 단 '라벨 없이')
  - 정상 기준 = 우리 최종 모델 학습 데이터(unlabeled 정상 생산 샷, outputs/final_model/{p}_train_features.csv)
    (sanngrok은 labeled 과거 날짜의 정상 행으로 적합 → 우리는 라벨을 아예 안 씀)
  - fill    : 충전시간 ~ 사출속도·사출압·전환압·계량위치·배럴2·배럴4·금형3·금형4  (Ridge α=10, 표준화)
  - plast   : 계량시간 ~ 스크류RPM·계량위치·배럴2·배럴4·호퍼 (Max_Back_Pressure는 21개 원값에 없어 제외)
  - cushion : 쿠션위치 ~ 계량위치·충전시간·사출압·전환압
  - switch  : 전환압 ~ 사출압·사출속도·충전시간·배럴2·금형3
  res = z(실제) − 예측 (z는 정상 기준 평균·표준편차). +면 '정상 관계보다 큼'
평가: 37번 harness (② labeled 전체·샷·초기허용 제외, RG3 R6 제외)
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from harness import load, evaluate, topk_counts, circ_null
from final_model import raw, INITIAL

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
HERE = _o.path.dirname(_o.path.abspath(__file__))
REL = {
    'fill': ('Filling_Time', ['Max_Injection_Speed', 'Max_Injection_Pressure', 'Max_Switch_Over_Pressure',
                              'Plasticizing_Position', 'Barrel_Temperature_2', 'Barrel_Temperature_4',
                              'Mold_Temperature_3', 'Mold_Temperature_4']),
    'plast': ('Plasticizing_Time', ['Max_Screw_RPM', 'Plasticizing_Position', 'Barrel_Temperature_2',
                                    'Barrel_Temperature_4', 'Hopper_Temperature']),
    'cushion': ('Cushion_Position', ['Plasticizing_Position', 'Filling_Time', 'Max_Injection_Pressure',
                                     'Max_Switch_Over_Pressure']),
    'switch': ('Max_Switch_Over_Pressure', ['Max_Injection_Pressure', 'Max_Injection_Speed', 'Filling_Time',
                                            'Barrel_Temperature_2', 'Mold_Temperature_3']),
}


def residuals(ref, lab):
    out = {}
    R = ref[[raw(c) for c in set(sum([[t] + i for t, i in REL.values()], []))]].replace([np.inf, -np.inf], np.nan)
    med = R.median(); R = R.fillna(med)
    L = lab[R.columns].replace([np.inf, -np.inf], np.nan).fillna(med)
    for k, (t, inp) in REL.items():
        inp = [raw(c) for c in inp if R[raw(c)].std() > 0]
        mu, sd = R[raw(t)].mean(), max(R[raw(t)].std(), 1e-6)
        sc = StandardScaler().fit(R[inp])
        rg = Ridge(alpha=10).fit(sc.transform(R[inp]), (R[raw(t)] - mu) / sd)
        out[f'res_{k}'] = ((L[raw(t)] - mu) / sd - rg.predict(sc.transform(L[inp]))).values
        out[f'res_{k}_ref'] = ((R[raw(t)] - mu) / sd - rg.predict(sc.transform(R[inp]))).values
    return out


def pct_in(ref_s, s):
    r = np.sort(ref_s[np.isfinite(ref_s)])
    return np.searchsorted(r, s, side='right') / (len(r) + 1)


def budget_first(lab, s_first, alarm, s_rest):
    """경보 샷(alarm)을 맨 위에, 나머지는 s_rest 순 (반복 21과 같은 방식)"""
    r = pd.Series(s_rest).rank(pct=True).values
    return np.where(alarm, 2.0 + pd.Series(s_first).rank(pct=True).values, r)


def main():
    rows, detail, overlap = [], [], []
    for p in ('cn7', 'rg3'):
        ref, lab, rc = load(p)
        R = residuals(ref, lab)
        fin = lab['score'].values
        S = {'final (M_maha_free)': fin}
        for k in REL:
            S[f'res_{k} (+)'] = R[f'res_{k}']
            S[f'|res_{k}|'] = np.abs(R[f'res_{k}'])
        S['max|res| 4개'] = np.max(np.abs(np.c_[[R[f'res_{k}'] for k in REL]]), axis=0)
        # 결합: 라벨 없이 정한 q0.999 경보 몫을 최종 모델이 먼저 쓰고, 나머지 예산을 잔차 순위로
        alarm = lab['pct'].values >= 0.999
        S['final q0.999 경보 → res_fill(+)'] = budget_first(lab, fin, alarm, R['res_fill'])
        S['final q0.999 경보 → |res_fill|'] = budget_first(lab, fin, alarm, np.abs(R['res_fill']))
        S['rank max(final, res_fill+)'] = np.maximum(pd.Series(fin).rank(pct=True).values,
                                                     pd.Series(R['res_fill']).rank(pct=True).values)
        for n, s in S.items():
            rows.append(evaluate(p, lab, s, n))

        # --- 겹침: 불량 샷별 순위, 상위 10% 집합 교집합, Spearman
        m = lab['eval'].values
        rk = lambda s: pd.Series(s[m]).rank(ascending=False, pct=True).values
        fr, rr, ar = rk(fin), rk(R['res_fill']), rk(np.abs(R['res_fill']))
        k10 = np.ceil(0.10 * m.sum())
        top_f, top_r = fr <= k10 / m.sum(), rr <= k10 / m.sum()
        typ = lab['type'].values[m]
        overlap.append({'제품': p, 'Spearman(final, res_fill)': pd.Series(fin[m]).corr(pd.Series(R['res_fill'][m]), method='spearman'),
                        'Spearman(final, |res_fill|)': pd.Series(fin[m]).corr(pd.Series(np.abs(R['res_fill'][m])), method='spearman'),
                        '상위10% 샷 수': int(top_f.sum()), '두 점수 상위10% 공통 샷': int((top_f & top_r).sum()),
                        '무작위 기대 공통': round(top_f.sum() * top_r.sum() / m.sum(), 1),
                        '불량 중 final만': int((top_f & ~top_r & (typ != '정상')).sum()),
                        '불량 중 res_fill만': int((~top_f & top_r & (typ != '정상')).sum()),
                        '불량 중 둘 다': int((top_f & top_r & (typ != '정상')).sum())})
        d = lab.loc[m, ['run_id', 'TimeStamp', 'type', 'Reason']].assign(
            final_순위=fr, res_fill_순위=rr, abs_res_fill_순위=ar, res_fill=R['res_fill'][m],
            dev_Filling_Time=lab.loc[m, 'dev_Filling_Time'].values, 충전시간=lab.loc[m, raw('Filling_Time')].values)
        detail.append(d[d['type'] != '정상'].assign(제품=p))

        # 초기허용불량 샷(평가 제외)도 참고로: 두 점수가 어디에 두는지 (labeled 전체 기준 순위)
        ini = (lab['Reason'] == INITIAL).values & (lab['split'] != '제외').values
        if ini.any():
            allm = (lab['split'] != '제외').values
            fa = pd.Series(fin[allm]).rank(ascending=False, pct=True).values
            ra = pd.Series(R['res_fill'][allm]).rank(ascending=False, pct=True).values
            print(f'\n[{p}] 초기허용불량 샷 순위 (labeled 전체 중 상위 %):')
            print(pd.DataFrame({'TimeStamp': lab.loc[allm, 'TimeStamp'].values[ini[allm]],
                                'final': fa[ini[allm]].round(4), 'res_fill': ra[ini[allm]].round(4),
                                '충전시간': lab.loc[allm, raw('Filling_Time')].values[ini[allm]]}).to_string(index=False))

    T = pd.DataFrame(rows).drop(columns='_c')
    print('\n=== 1. 단독·결합 성능 (② 틀) ==='); print(T.round(3).to_string(index=False))
    print('\n=== 2. final vs res_fill 겹침 ==='); print(pd.DataFrame(overlap).round(3).to_string(index=False))
    D = pd.concat(detail)
    print('\n=== 3. 불량 샷별 순위 (평가 대상 중 상위 %, 0.10 이하면 상위 10% 안) ===')
    print(D.round(3).to_string(index=False))
    T.to_csv(_o.path.join(HERE, 'exp53b_scores.csv'), index=False, encoding='utf-8-sig')
    D.to_csv(_o.path.join(HERE, 'exp53b_defect_ranks.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(overlap).to_csv(_o.path.join(HERE, 'exp53b_overlap.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
