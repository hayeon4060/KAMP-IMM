"""
scope.py — 학습·평가 범위 필터
파생변수는 전체 시간 순서로 이미 계산돼 있으므로(현장과 동일), 모델 학습·평가 단계에서만 행을 뺀다.
"""
INITIAL = '초기허용불량'


def apply_scope(d, exclude_initial=True, exclude_transient=False):
    """
    exclude_initial  : 초기허용불량 샷 제거 (재가동 램프업 = 규칙으로 처리하는 대상)
    exclude_transient: 재가동 후 5샷(is_transient=1) 전체 제거 (과도구간은 규칙으로 처리한다고 가정)
    """
    d = d.copy()
    if exclude_initial:
        d = d[d['Reason'] != INITIAL]
    if exclude_transient and 'is_transient' in d.columns:
        d = d[d['is_transient'] == 0]
    return d.reset_index(drop=True)


def scope_tag(exclude_initial=True, exclude_transient=False):
    tag = 'noinit' if exclude_initial else 'all'
    return tag + ('_notrans' if exclude_transient else '')