import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))  # notebooks 폴더에서 실행: python 53_sanngrok/exp53c_cn7_expand.py
"""
53-C. CN7 운영 규칙 '불량 발견 시 주변 확장' — 앞뒤 몇 샷이 좋은가 (0·3·5·7·10·15·20)
  - 49·52번 시뮬레이션(active_learning.simulate) 그대로: 100샷 묶음마다 검사 10%, 시간 순, 검사한 샷만 라벨 공개
  - 기본 점수 = CN7 통합 모델 (q0.999 경보 먼저 + std4 흔들림)
  - 확장 샷은 같은 묶음 예산 안에서 씀 → 모든 설정의 총 검사 샷 수 동일
  - 확장 이득의 출처를 보려고, 불량을 '첫 발견 이후 찾은 수'로 Run별 분해
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import active_learning as AL
import semi_final as SF
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
HERE = _o.path.dirname(_o.path.abspath(__file__))

rows, eps = [], []
D0 = AL.prepare('cn7')
for frame in ('2차', '전체'):
    D = dict(D0, lab=SF.apply_frame(D0['lab'], 'cn7', frame))
    for nbr in (0, 3, 5, 7, 10, 15, 20):
        st = 'S1 고정 순위' if nbr == 0 else 'S2 고정 + 주변 확장'
        L = AL.simulate(D, st, 'combo', nbr=max(nbr, 1), seed=0)
        r = AL.summarize_log(D, L, st, 'combo', 0); r.update({'평가틀': frame, '확장 ±샷': nbr})
        rows.append(r)
        for e in AL.episode_view(D, L, st, 'combo', 0):
            e.update({'평가틀': frame, '확장 ±샷': nbr}); eps.append(e)
        # 확장으로 검사된 샷이 정상인 비율 (= 확장 비용)
T = pd.DataFrame(rows)[['평가틀', '확장 ±샷', '검사 샷', '찾은 불량', '전체 불량', 'Precision', 'Recall', 'F1', '가스', '가스 전체', '미성형', '미성형 전체']]
print(T.round(3).to_string(index=False))
E = pd.DataFrame(eps)[['평가틀', '확장 ±샷', 'Run', 'Run 불량', 'Run 검사 샷', '첫 불량까지 검사 샷', '첫 발견 이후 찾은 불량']]
print(E.to_string(index=False))
T.to_csv(_o.path.join(HERE, 'exp53c_cn7_expand.csv'), index=False, encoding='utf-8-sig')
E.to_csv(_o.path.join(HERE, 'exp53c_cn7_expand_episodes.csv'), index=False, encoding='utf-8-sig')


# ---------------------------------------------------------------- B. 예산 밖 '추가 검사'로 확장할 때
# 기본 검사(S1, 묶음마다 10%)는 그대로 두고, 불량이 나오면 같은 Run 앞뒤 ±nbr샷을 추가로 검사 (추가에서 또 불량이 나오면 거기서도 확장)
print('\n=== B. 예산 밖 추가 검사로 확장 ===')
rowsB = []
for frame in ('2차', '전체'):
    D = dict(D0, lab=SF.apply_frame(D0['lab'], 'cn7', frame))
    lab = D['lab']; ev = lab['eval'].values
    y = (lab['PassOrFail'].values == 1) & ev
    t = lab['type'].values; runs = lab['run_id'].values; seq = lab['seq'].values
    L0 = AL.simulate(D, 'S1 고정 순위', 'combo', seed=0)
    base_idx = set(L0['i'].astype(int))
    for nbr in (0, 3, 5, 10, 15, 20):
        seen = set(base_idx); queue = [i for i in L0['i'].astype(int) if y[i]]
        while queue and nbr > 0:
            i = queue.pop(0)
            nb = np.flatnonzero(ev & (runs == runs[i]) & (np.abs(seq - seq[i]) <= nbr))
            for j in nb:
                if j not in seen:
                    seen.add(j)
                    if y[j]:
                        queue.append(j)
        idx = np.array(sorted(seen))
        found = int(y[idx].sum()); extra = len(idx) - len(base_idx)
        gas = int(np.isin(t[idx], ['가스']).sum() + ((lab['Reason'].values[idx] == '가스+미성형')).sum())
        rowsB.append({'평가틀': frame, '확장 ±샷': nbr, '기본 검사': len(base_idx), '추가 검사': extra,
                      '총 검사': len(idx), '총 검사 비율': len(idx) / ev.sum(), '찾은 불량': found, '전체 불량': int(y.sum()),
                      '가스(+혼합)': gas, 'Precision': found / len(idx), '추가 검사로 더 찾은 불량': found - int(y[np.array(sorted(base_idx))].sum())})
B = pd.DataFrame(rowsB)
print(B.round(3).to_string(index=False))
B.to_csv(_o.path.join(HERE, 'exp53c_cn7_expand_extra.csv'), index=False, encoding='utf-8-sig')
