"""
alarm_explain.py — 경보 샷마다 '어느 변수가 튀었나 + 이상 원인 후보'를 보여주기 (작업자 의사결정 지원, 보고서 4장)

입력: 20a로 저장한 모델(outputs/final_model/{p}_final_model.pkl) + 20b가 저장한 점수(outputs/final_model/{p}_scores.csv)

1) 변수별 기여도 (Mahalanobis 분해)
   점수² = Σ_i c_i,   c_i = (x − μ)_i · [Σ⁻¹ (x − μ)]_i
   → 변수마다 '점수에 얼마나 보탰나'. 합 = 점수². 비율(c_i / 점수²)로 상위 변수를 뽑음
     (변수 간 상관 때문에 c_i가 음수인 변수도 있음 = 다른 변수와 같이 움직여 오히려 '정상스러움'을 보탠 경우)

2) 이상 원인 후보 (규칙, 사전 고정 — 검증된 유형 분류기가 아님)
   dev 값은 '같은 Run 직전 20샷 대비 몇 배 벗어났나'(robust z)
     충전 이상(미성형형)  : fill_anomaly ≤ −3  또는 (dev_충전시간 ≤ −3 이고 dev_사출속도 ≥ 3)
     쿠션 부족(재료 부족형): dev_쿠션 ≤ −3
     계량 지연형          : dev_계량시간 ≥ 3
     흐름저항 상승(점도형): dev_flow_res ≥ 3 또는 dev_사출압 ≥ 3
     열 상태 변화형       : |dev_금형온도| ≥ 3 또는 |dev_호퍼온도| ≥ 3
     그 외                : '기타 이상 (상위 기여 변수 참고)'
   여러 개 해당하면 모두 표시
"""
import numpy as np
import pandas as pd

from final_model import FinalModel, FEATURES, INITIAL

LABEL = {
    'dev_Max_Injection_Speed': '사출속도', 'dev_Filling_Time': '충전시간', 'dev_Max_Injection_Pressure': '사출압',
    'dev_Max_Switch_Over_Pressure': '전환압', 'dev_Average_Back_Pressure': '배압', 'dev_Cushion_Position': '쿠션위치',
    'dev_Plasticizing_Time': '계량시간', 'dev_Cycle_Time': '사이클', 'fill_anomaly': '충전 이상지표',
    'ev_max_abs_dev': '최대 편차', 'dev_Mold_Temperature_3': '금형온도', 'dev_Hopper_Temperature': '호퍼온도',
    'mold_rise': '금형온도 상승량', 'hopper_rise': '호퍼온도 상승량', 'mold_slope20': '금형온도 추세',
    'hopper_slope20': '호퍼온도 추세', 'dev_flow_res': '흐름저항',
}
TH = 3.0


def load(p, model_dir='./outputs/final_model'):
    model = FinalModel.load(f'{model_dir}/{p}_final_model.pkl')
    f = pd.read_csv(f'{model_dir}/{p}_scores.csv', parse_dates=['TimeStamp'])
    return model, f


def contributions(model, df):
    """반환: (기여도 DataFrame [샷 × 변수], 점수²)"""
    X = model._prep(df) - model.lw.location_
    PX = X @ model.lw.precision_
    C = X * PX
    return pd.DataFrame(C, columns=model.cols, index=df.index), C.sum(axis=1)


def cause_hint(r):
    g = lambda c: r.get(c, np.nan)
    out = []
    if g('fill_anomaly') <= -TH or (g('dev_Filling_Time') <= -TH and g('dev_Max_Injection_Speed') >= TH):
        out.append('충전 이상(미성형형)')
    if g('dev_Cushion_Position') <= -TH:
        out.append('쿠션 부족(재료 부족형)')
    if g('dev_Plasticizing_Time') >= TH:
        out.append('계량 지연형')
    if g('dev_flow_res') >= TH or g('dev_Max_Injection_Pressure') >= TH:
        out.append('흐름저항 상승(점도형)')
    if abs(g('dev_Mold_Temperature_3')) >= TH or abs(g('dev_Hopper_Temperature')) >= TH:
        out.append('열 상태 변화형')
    return ' + '.join(out) if out else '기타 이상 (상위 기여 변수 참고)'


def _fmt_top(row_c, row_v, d2, k=3):
    top = row_c.sort_values(ascending=False).head(k)
    return ', '.join(f'{LABEL.get(c, c)}({row_v[c]:+.1f}, {100 * v / d2:.0f}%)' for c, v in top.items())


def alarm_table(p, q=0.999, model_dir='./outputs/final_model', eval_only=True, k=3):
    """
    pct ≥ q 경보 샷 목록 (점수 높은 순 = 검사 우선순위)
    eval_only=True: 평가 범위(제외 구간·초기허용불량 뺀 샷)만
    상위 기여 변수 표기: 이름(dev 값, 점수² 중 비율)
    """
    model, f = load(p, model_dir)
    if eval_only:
        f = f[(f['split'] != '제외') & (f['Reason'] != INITIAL)]
    a = f[f['pct'] >= q].sort_values('score', ascending=False).copy()
    if a.empty:
        return a
    C, d2 = contributions(model, a)
    a['검사 순위'] = np.arange(1, len(a) + 1)
    a['실제'] = np.where(a['PassOrFail'] == 1, '불량(' + a['Reason'].astype(str) + ')', '정상')
    a['이상 원인 후보'] = a.apply(cause_hint, axis=1)
    a['상위 기여 변수 (dev, 비율)'] = [_fmt_top(C.loc[i], a.loc[i], d2[j], k) for j, i in enumerate(a.index)]
    return a[['검사 순위', 'TimeStamp', 'run_id', 'score', 'pct', '실제', '이상 원인 후보', '상위 기여 변수 (dev, 비율)']]


def hint_vs_actual(p, q=0.99, model_dir='./outputs/final_model'):
    """경보 샷의 '원인 후보' × '실제 라벨' 교차표 (후보가 실제 유형과 맞는 경향이 있는지 점검)"""
    t = alarm_table(p, q, model_dir)
    if t.empty:
        return t
    return pd.crosstab(t['이상 원인 후보'], t['실제'], margins=True, margins_name='합계')


def plot_contrib(p, timestamp, model_dir='./outputs/final_model', save=None):
    """경보 샷 하나의 변수별 기여도 막대 + 해당 변수의 dev 값"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    model, f = load(p, model_dir)
    r = f[f['TimeStamp'] == pd.Timestamp(timestamp)]
    C, d2 = contributions(model, r)
    s = (C.iloc[0] / d2[0] * 100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(s) + 1.4))
    col = ['#c0392b' if v > 0 else '#9db9d5' for v in s.values]
    ax.barh([LABEL.get(c, c) for c in s.index], s.values, color=col)
    for i, c in enumerate(s.index):
        ax.text(s.values[i], i, f'  dev {r.iloc[0][c]:+.1f}', va='center', fontsize=8)
    row = r.iloc[0]
    ax.set_xlabel('점수² 중 기여 비율 (%)')
    ax.set_title(f'{p.upper()} {row["TimeStamp"]:%Y-%m-%d %H:%M:%S} · pct {row["pct"]:.4f} · '
                 f'{"불량(" + str(row["Reason"]) + ")" if row["PassOrFail"] == 1 else "정상"} · '
                 f'후보: {cause_hint(row)}', loc='left', fontsize=9)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
