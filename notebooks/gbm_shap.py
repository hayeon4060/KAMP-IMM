"""
gbm_shap.py — 33번: 머신러닝(지도) + SHAP (2022 우수작의 'AutoML·CatBoost 비교 + SHAP 해석'을 우리 데이터에 맞게)

2022와 다르게 한 것
  - PyCaret 무작위 교차검증 X → LH/RH 같은 샷·같은 Run의 이웃 샷이 학습/평가에 섞여 성능이 부풀려짐
    대신 Run 하나씩 빼고 검증(LORO): 빼 둔 Run은 학습에 전혀 안 들어감
  - 하이퍼파라미터는 사전 고정 (라벨로 튜닝 X) — 불량이 15~32개라 튜닝하면 바로 과적합
  - 지표: Accuracy·F1 대신 AP(PR-AUC), 상위 k%, Run별 상위 10% (불량률 1% 미만이라 Accuracy는 의미 없음)

데이터
  - outputs/final_model/{p}_scores.csv (labeled 샷 + 파생변수 17개 + 최종 모델 점수) ← 20b 결과
  - base/{p}_shots.csv 의 원값 21개 (TimeStamp로 병합)
  - 학습·평가 대상: split ≠ '제외', 초기허용불량 제외 (최종 모델 ② 틀과 같음). 점수는 빼 둔 Run의 모든 샷에 매김
  - 타깃: 불량 전체(가스+미성형) = PassOrFail. 유형별 성능은 평가에서 따로 봄

실험 조건 (2022의 '파생변수 유무 / 불균형 처리 / 모델 비교'의 대응)
  - 변수: free17(최종 모델과 같은 파생변수) / raw21(원값) / free17+raw21
  - 불균형: weight(클래스 가중치) / smote (imbalanced-learn 있을 때)
  - 모델: LightGBM / CatBoost / RandomForest
  ⚠ 학습 Run에 불량이 2개 미만이면 그 Run은 학습 불가 → 점수 0(=판단 불가)으로 두고 표에 표시
  ⚠ CN7 가스 11건은 모두 한 Run(R1)에 있음 → R1을 빼면 학습에 가스가 0개 → 가스는 원리상 못 배움 (그 자체가 결과)

SHAP (2022의 SHAP 중요도·워터폴 대응)
  - 선택한 설정으로 학습 가능한 labeled 전체에 다시 학습 → TreeExplainer
  - 전체 중요도(평균 |SHAP|), 유형별(가스·미성형 샷의 평균 SHAP 방향), 샷 하나의 워터폴
  - ⚠ 전체 데이터로 학습한 설명 = '모델이 무엇을 보고 있나'의 기술(記述). 성능 근거는 LORO 표만 사용
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from final_model import (FEATURES, ALL_RAW, raw, INITIAL, SEED, TYPE_GROUPS, load_labeled_shots,
                         load_official_rows, eval_frame1, eval_frame2, apply_exclusions)
from seq_ae import delta_ap, ensure_inputs

RAW = [raw(v) for v in ALL_RAW]
FEATURE_SETS = {'free17': list(FEATURES), 'raw21': RAW, 'free17+raw21': list(FEATURES) + RAW}
MODELS = ('lgbm', 'catboost', 'rf')
IMBALANCE = ('weight', 'smote')
MIN_POS = 2
BASELINE = 'M_maha_free (최종, 비지도)'


# ====================================================================== 데이터
def load(p, base_dir='.', model_dir='./outputs/final_model'):
    ensure_inputs(p, base_dir, model_dir)                      # 없으면 저장된 최종 모델로 생성
    f = f'{model_dir}/{p}_scores.csv'
    d = pd.read_csv(f, parse_dates=['TimeStamp']).rename(columns={'score': 'maha_score', 'pct': 'maha_pct'})
    d['Reason'] = d['Reason'].fillna('정상')
    d = apply_exclusions(d, p)                                       # RG3 R6 등 평가 제외 Run
    shots = load_labeled_shots(p, base_dir)
    d = d.merge(shots[['TimeStamp'] + [c for c in RAW if c in shots.columns]], on='TimeStamp', how='left')
    d = d.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    d['eval'] = ((d['split'] != '제외') if 'split' in d.columns else True) & (d['Reason'] != INITIAL)
    d['y'] = d['PassOrFail'].fillna(0).astype(int)
    d['type'] = np.select([d['Reason'].isin(TYPE_GROUPS['가스']), d['Reason'].isin(TYPE_GROUPS['미성형'])],
                          ['가스', '미성형'], '정상')
    return d


def available():
    out = {}
    for name, mod in (('lightgbm', 'lightgbm'), ('catboost', 'catboost'), ('imblearn', 'imblearn'), ('shap', 'shap')):
        try:
            __import__(mod)
            out[name] = True
        except ImportError:
            out[name] = False
    return out


# ====================================================================== 모델
def make_model(name, imbalance):
    w = imbalance == 'weight'
    if name == 'lgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              class_weight='balanced' if w else None, random_state=SEED, verbose=-1)
    if name == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=400, depth=4, learning_rate=0.05, random_seed=SEED, verbose=0,
                                  auto_class_weights='Balanced' if w else None, allow_writing_files=False)
    if name == 'rf':
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3, n_jobs=-1,
                                      class_weight='balanced_subsample' if w else None, random_state=SEED)
    raise ValueError(name)


def fit_one(name, imbalance, X, y):
    """반환: (모델, 결측 대치기 or None). RF·SMOTE는 결측을 못 다뤄서 중앙값 대치"""
    imp = None
    if name == 'rf' or imbalance == 'smote':
        imp = SimpleImputer(strategy='median', keep_empty_features=True).fit(X)
        X = imp.transform(X)
    if imbalance == 'smote':
        from imblearn.over_sampling import SMOTE
        k = int(min(5, y.sum() - 1))
        X, y = SMOTE(k_neighbors=max(1, k), random_state=SEED).fit_resample(X, y)
    m = make_model(name, imbalance)
    m.fit(X, y)
    return m, imp


def predict(m, imp, X):
    if imp is not None:
        X = imp.transform(X)
    return m.predict_proba(X)[:, 1]


def loro_scores(d, cols, name, imbalance):
    """Run 하나씩 빼고 학습 → 빼 둔 Run의 모든 샷 점수. 반환: (점수, 학습 불가 Run 목록)"""
    s = np.zeros(len(d))
    skipped = []
    X_all = d[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)
    for r in d['run_id'].unique():
        te = (d['run_id'] == r).values
        tr = (~te) & d['eval'].values
        y = d['y'].values[tr]
        if y.sum() < MIN_POS or y.sum() == len(y):
            skipped.append(r)
            continue
        m, imp = fit_one(name, imbalance, X_all[tr], y)
        s[te] = predict(m, imp, X_all[te])
    return s, skipped


# ====================================================================== 평가
def evaluate(p, d, score, name, base_dir='.'):
    f1 = eval_frame1(p, load_official_rows(p, base_dir), pd.Series(score, index=d['TimeStamp'].values))
    m = d['eval'].values
    g = pd.DataFrame({'run_id': d['run_id'].values[m], 'y': d['y'].values[m], 'Reason': d['Reason'].values[m],
                      'score': np.asarray(score)[m]})
    f2 = eval_frame2(p, g)
    f1['모델'] = f2['모델'] = name
    return f1, f2, g


def run(products=('cn7', 'rg3'), models=MODELS, feature_sets=tuple(FEATURE_SETS), imbalance=IMBALANCE,
        base_dir='.', model_dir='./outputs/final_model', out_dir='./outputs/gbm_shap', verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    av = available()
    models = [m for m in models if (m != 'lgbm' or av['lightgbm']) and (m != 'catboost' or av['catboost'])]
    imbalance = [i for i in imbalance if i != 'smote' or av['imblearn']]
    if verbose:
        print('라이브러리:', av, '| 사용 모델:', models, '| 불균형:', imbalance)
    F1, F2, SUM, SCORES = [], [], [], {}
    for p in products:
        d = load(p, base_dir, model_dir)
        runs = d[d['eval']].groupby('run_id').agg(샷=('y', 'size'), 불량=('y', 'sum'),
                                                   가스=('type', lambda t: int((t == '가스').sum())),
                                                   미성형=('type', lambda t: int((t == '미성형').sum())))
        if verbose:
            print(f'[{p}] labeled {len(d):,}샷 (평가 {int(d["eval"].sum()):,}, 불량 {int(d.loc[d["eval"], "y"].sum())})')
            print(runs.to_string())
        f1, f2, g0 = evaluate(p, d, d['maha_score'].values, BASELINE, base_dir)
        F1.append(f1); F2.append(f2)
        SUM.append(_row(p, BASELINE, None, None, None, f1, f2, None, []))
        out = d[['TimeStamp', 'run_id', 'Reason', 'y', 'eval', 'maha_score']].copy()
        for fs in feature_sets:
            cols = [c for c in FEATURE_SETS[fs] if c in d.columns]
            for mname in models:
                for imb in imbalance:
                    name = f'{mname}|{fs}|{imb}'
                    s, skipped = loro_scores(d, cols, mname, imb)
                    f1, f2, g = evaluate(p, d, s, name, base_dir)
                    F1.append(f1); F2.append(f2)
                    dl = delta_ap(g['y'].values, g['score'].values, g0['score'].values)
                    SUM.append(_row(p, name, mname, fs, imb, f1, f2, dl, skipped))
                    out[name] = s
                    if verbose:
                        print(f'  [{p}] {name}: ② AP {f2["AP"]:.3f} | ① AP {f1["AP"]:.3f}'
                              + (f' | 학습 불가 Run {skipped}' if skipped else ''), flush=True)
        SCORES[p] = out
        out.to_csv(f'{out_dir}/{p}_loro_scores.csv', index=False, encoding='utf-8-sig')
    R = {'summary': pd.DataFrame(SUM), 'frame1': pd.DataFrame(F1), 'frame2': pd.DataFrame(F2), 'scores': SCORES}
    for k in ('summary', 'frame1', 'frame2'):
        R[k].to_csv(f'{out_dir}/{k}.csv', index=False, encoding='utf-8-sig')
    return R


def _row(p, name, mname, fs, imb, f1, f2, dl, skipped):
    r = {'제품': p, '모델': name, '알고리즘': mname or 'maha', '변수': fs or 'free17', '불균형': imb or '-',
         '① AP': f1['AP'], '① 상위5%': f1['상위5%'], '① 상위10%': f1['상위10%'],
         '② AP': f2['AP'], '② AP 95%CI': f'[{f2["AP_lo"]:.3f}, {f2["AP_hi"]:.3f}]',
         '② AP_가스': f2.get('AP_가스', np.nan), '② AP_미성형': f2.get('AP_미성형', np.nan),
         'Run별 상위10%': f2['Run별 상위10%'],
         'Run별 상위10% 가스': f2.get('Run별 상위10%_가스', ''), 'Run별 상위10% 미성형': f2.get('Run별 상위10%_미성형', ''),
         '학습 불가 Run': ', '.join(map(str, skipped))}
    if dl is not None:
        r.update({'ΔAP(②, vs 최종)': dl[0], 'ΔAP 95%CI': f'[{dl[1]:+.3f}, {dl[2]:+.3f}]', 'P(최종보다 좋음)': dl[3],
                  '유의하게 좋음': 'O' if dl[1] > 0 else '-'})
    return r


# ====================================================================== SHAP
def shap_explain(p, model='lgbm', feature_set='free17+raw21', imbalance='weight', base_dir='.',
                 model_dir='./outputs/final_model', out_dir='./outputs/gbm_shap'):
    """학습 가능한 labeled 전체로 다시 학습 → SHAP. 반환: dict(중요도 표, 유형별 표, shap 값, X, 데이터)"""
    import shap
    os.makedirs(out_dir, exist_ok=True)
    d = load(p, base_dir, model_dir)
    cols = [c for c in FEATURE_SETS[feature_set] if c in d.columns]
    tr = d['eval'].values
    X = d.loc[tr, cols].replace([np.inf, -np.inf], np.nan)
    y = d.loc[tr, 'y'].values
    m, imp = fit_one(model, imbalance, X.values.astype(float), y)
    Xe = pd.DataFrame(imp.transform(X.values) if imp is not None else X.values, columns=cols, index=X.index)
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(Xe)
    if isinstance(sv, list):                                     # RF: [음성, 양성]
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                                              # (N, F, 2) 형태
        sv = sv[:, :, 1]
    t = d.loc[tr, 'type'].values
    imp_tbl = pd.DataFrame({'변수': cols, '평균 |SHAP| (전체)': np.abs(sv).mean(0),
                            '평균 SHAP (가스 샷)': sv[t == '가스'].mean(0) if (t == '가스').any() else np.nan,
                            '평균 SHAP (미성형 샷)': sv[t == '미성형'].mean(0) if (t == '미성형').any() else np.nan,
                            '평균 SHAP (정상 샷)': sv[t == '정상'].mean(0)}) \
        .sort_values('평균 |SHAP| (전체)', ascending=False).reset_index(drop=True)
    imp_tbl['구분'] = np.where(imp_tbl['변수'].isin(FEATURES), '파생(free)', '원값(raw)')
    tag = f'{p}_{model}_{feature_set}_{imbalance}'.replace('+', '_')
    imp_tbl.to_csv(f'{out_dir}/{tag}_shap_importance.csv', index=False, encoding='utf-8-sig')
    base = ex.expected_value
    if np.ndim(base) > 0:
        base = np.asarray(base).ravel()[-1]
    return {'importance': imp_tbl, 'shap_values': sv, 'X': Xe, 'data': d.loc[tr].reset_index(drop=True),
            'base_value': float(base), 'tag': tag, 'cols': cols}


def plot_shap(S, top=15, save_dir=None):
    """① 평균 |SHAP| 막대 (파생/원값 색 구분) ② beeswarm ③ 유형별 평균 SHAP 방향"""
    import matplotlib.pyplot as plt
    import shap
    try:
        from plot_style import set_style, short
        set_style()
    except ImportError:
        short = lambda c: c
    t = S['importance'].head(top).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.42 * top + 1.5))
    col = np.where(t['구분'] == '파생(free)', '#1f4e79', '#b5b4ad')
    axes[0].barh([short(c) for c in t['변수']], t['평균 |SHAP| (전체)'], color=col)
    axes[0].set_title(f'{S["tag"]} — 평균 |SHAP| (진한색 = 파생변수, 회색 = 원값)', loc='left', fontsize=10)
    y = np.arange(len(t))
    axes[1].barh(y - 0.2, t['평균 SHAP (가스 샷)'], height=0.4, color='#e67e22', label='가스 샷')
    axes[1].barh(y + 0.2, t['평균 SHAP (미성형 샷)'], height=0.4, color='#1f4e79', label='미성형 샷')
    axes[1].set_yticks(y); axes[1].set_yticklabels([short(c) for c in t['변수']])
    axes[1].axvline(0, color='k', lw=0.8)
    axes[1].set_title('유형별 평균 SHAP (+ = 불량 쪽으로 밀어올림)', loc='left', fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    if save_dir:
        fig.savefig(f'{save_dir}/{S["tag"]}_shap_bar.png', dpi=150, bbox_inches='tight')
    plt.show()
    shap.summary_plot(S['shap_values'], S['X'], feature_names=[short(c) for c in S['cols']], max_display=top,
                      show=False)
    if save_dir:
        plt.savefig(f'{save_dir}/{S["tag"]}_shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_waterfall(S, which='미성형', k=0, top=10, save_dir=None):
    """유형(가스/미성형) 샷 중 모델 점수가 k번째로 높은 샷의 워터폴"""
    import matplotlib.pyplot as plt
    import shap
    d = S['data']
    idx = np.flatnonzero(d['type'].values == which)
    if len(idx) == 0:
        print(f'{which} 샷 없음'); return
    order = idx[np.argsort(-S['shap_values'][idx].sum(1))]
    i = order[min(k, len(order) - 1)]
    e = shap.Explanation(values=S['shap_values'][i], base_values=S['base_value'], data=S['X'].iloc[i].values,
                         feature_names=S['cols'])
    shap.plots.waterfall(e, max_display=top, show=False)
    plt.title(f'{S["tag"]} — {which} 샷 {d["TimeStamp"].iloc[i]} ({d["run_id"].iloc[i]})', fontsize=9)
    if save_dir:
        plt.savefig(f'{save_dir}/{S["tag"]}_waterfall_{which}_{k}.png', dpi=150, bbox_inches='tight')
    plt.show()
