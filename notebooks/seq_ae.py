"""
seq_ae.py — 32번: 시계열 딥러닝 이상탐지 (2022 우수작의 'LSTM → CNN-LSTM, window 실험'을 우리 데이터에 맞게)

왜 '오토인코더(정상만 학습)'인가
  - 불량 라벨이 CN7 15샷 / RG3 32샷뿐 → LSTM을 불량으로 지도학습하면 과적합이 뻔함
  - 정상 생산(unlabeled reference)은 수천 샷 → 정상 시퀀스를 복원하도록 학습하고, 복원이 안 되는 정도를 이상 점수로 사용
  - 최종 모델(Mahalanobis)과 같은 학습 데이터·같은 평가 → 차이는 '시퀀스(과거 W샷)를 통째로 보느냐'뿐

데이터 (최종 모델 결과 파일을 그대로 사용 → 20a·20b를 먼저 실행)
  - 학습: outputs/final_model/{p}_train_features.csv  (reference, 라벨 0개)
          Run 시간 순으로 앞 80% 학습 / 뒤 20% 검증(조기 종료 + 백분위 기준) — Run 단위로 자름
  - 테스트: outputs/final_model/{p}_scores.csv  (labeled 샷, 17개 변수 + Mahalanobis 점수)
  - 입력: 'free' = 최종 모델과 같은 파생변수 17개 / 'raw' = 원값 21개 (선택, 2022의 '파생변수 유무' 실험)
  - 스케일: 학습 구간 중앙값·IQR → ±20 클리핑 → NaN=0 (최종 모델과 동일)

창(window)
  - 샷 t의 입력 = 같은 Run의 직전 W샷(t 포함). Run 앞부분은 0(=정상 중앙값)으로 왼쪽 채움 → 미래 정보 없음
  - W = 10 / 20 / 30 / 50샷 (2022의 window 1~10분 실험의 대응). CN7 R1 가스 구간(25~58번째 샷)은 W=50이면 한 창에 들어감

모델 (설정은 결과 보기 전에 고정, 라벨로 튜닝하지 않음)
  - lstm_ae     : LSTM 인코더 → 잠재 16 → 반복 → LSTM 디코더
  - cnn_ae      : 1D-CNN 인코더 → 잠재 16 → 1D-CNN 디코더
  - cnn_lstm_ae : 1D-CNN(국소 패턴) → LSTM 인코더 → 잠재 16 → LSTM 디코더  (2022 최고 모델 구조의 오토인코더판)
  - 시드 3개 점수 평균 (딥러닝 결과 흔들림 완화)

점수 (둘 다 사전에 정함)
  - last : 창의 마지막 샷(지금 샷) 복원오차 → 샷 단위 이상 (미성형형)
  - win  : 창 전체 평균 복원오차 → 구간 단위 이상 (가스 에피소드 가설)
  - pct  : 검증 reference 점수 분포 대비 백분위 → q 임계값 경보 (최종 모델과 같은 방식)

평가: 최종 모델과 같은 함수 (① 공식 2차 행 / ② labeled 전체 샷, 초기허용 제외, 유형별 AP, q 임계값)
      + 최종 모델 대비 ΔAP (② 기준, 층화 부트스트랩 95% CI, P(더 좋음))
⚠ 설정이 24개(3모델 × 4창 × 2점수)라 하나가 우연히 좋아 보일 수 있음 → ΔAP 구간이 0을 넘는지로 판단
"""
import os
import time
import numpy as np
import pandas as pd

from final_model import (FEATURES, ALL_RAW, raw, INITIAL, SEED, CLIP, load_labeled_shots, load_official_rows,
                         eval_frame1, eval_frame2, threshold_table, apply_exclusions)

WINDOWS = (10, 20, 30, 50)
ARCHS = ('lstm_ae', 'cnn_ae', 'cnn_lstm_ae')
SCORE_TYPES = ('last', 'win')
SEEDS = (0, 1, 2)
VAL_FRAC = 0.2
EPOCHS, PATIENCE, BATCH, LR = 60, 8, 128, 1e-3
HIDDEN, LATENT = 64, 16
BASELINE = 'M_maha_free (최종)'


# ====================================================================== 데이터
def ensure_inputs(p, base_dir='.', model_dir='./outputs/final_model'):
    """
    {p}_train_features.csv / {p}_scores.csv 가 없으면 저장된 최종 모델(.pkl)로 다시 만든다.
    (train_features.csv는 final_model.py 최신판부터 저장 → 예전에 20a를 돌렸으면 없음)
    .pkl도 없으면 final_model.train()부터 실행. 결과는 20a·20b와 같은 파일·같은 값
    """
    from final_model import (FinalModel, train, build_reference, make_features, load_labeled_shots as _lls)
    f_tr, f_sc = f'{model_dir}/{p}_train_features.csv', f'{model_dir}/{p}_scores.csv'
    if os.path.exists(f_tr) and os.path.exists(f_sc):
        return
    pkl = f'{model_dir}/{p}_final_model.pkl'
    if not os.path.exists(pkl):
        print(f'[{p}] 최종 모델 없음 → final_model.train() 실행 (수 분)')
        train(base_dir=base_dir, products=(p,), out_dir=model_dir)
    model = FinalModel.load(pkl)
    shots = _lls(p, base_dir)
    if not os.path.exists(f_tr):
        print(f'[{p}] {os.path.basename(f_tr)} 없음 → 저장된 모델 기준값으로 학습 데이터 다시 생성 (수 분)')
        ref_shots, _ = build_reference(p, shots, base_dir)
        rf = make_features(ref_shots, model.steps, model.fr_step)
        tf = rf[['run_id', 'TimeStamp'] + [raw(v) for v in ALL_RAW if raw(v) in rf.columns] + list(FEATURES)].copy()
        tf['score'] = model.score(rf)
        tf['pct'] = model.pct(tf['score'])
        if len(tf) != model.n_ref:
            print(f'  ⚠ 샷 수 {len(tf):,} ≠ 모델 학습 샷 {model.n_ref:,} → 20a를 다시 실행하는 것을 권장')
        tf.sort_values(['run_id', 'TimeStamp']).to_csv(f_tr, index=False, encoding='utf-8-sig')
    if not os.path.exists(f_sc):
        print(f'[{p}] {os.path.basename(f_sc)} 없음 → labeled 점수 다시 계산')
        lf = make_features(shots, model.steps, model.fr_step)
        lf['score'] = model.score(lf)
        lf['pct'] = model.pct(lf['score'])
        keep = ['TimeStamp', 'run_id', 'split', 'PassOrFail', 'Reason', 'score', 'pct'] + list(FEATURES)
        lf[[c for c in keep if c in lf.columns]].to_csv(f_sc, index=False, encoding='utf-8-sig')


def load_data(p, inputs='free', base_dir='.', model_dir='./outputs/final_model'):
    """reference(학습) / labeled(테스트) 샷 표와 입력 열"""
    ensure_inputs(p, base_dir, model_dir)
    ref = pd.read_csv(f'{model_dir}/{p}_train_features.csv', parse_dates=['TimeStamp'])
    lab = pd.read_csv(f'{model_dir}/{p}_scores.csv', parse_dates=['TimeStamp'])
    lab = lab.rename(columns={'score': 'maha_score', 'pct': 'maha_pct'})
    lab['Reason'] = lab['Reason'].fillna('정상')
    lab = apply_exclusions(lab, p)                                   # RG3 R6 등 평가 제외 Run
    if inputs == 'free':
        cols = list(FEATURES)
    elif inputs == 'raw':
        cols = [raw(v) for v in ALL_RAW if raw(v) in ref.columns]
        shots = load_labeled_shots(p, base_dir)[['TimeStamp'] + cols]
        lab = lab.merge(shots, on='TimeStamp', how='left')
    else:
        raise ValueError(inputs)
    ref = ref.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    lab = lab.sort_values(['run_id', 'TimeStamp']).reset_index(drop=True)
    return ref, lab, cols


def split_reference(ref, val_frac=VAL_FRAC):
    """Run 시작 시각 순으로 앞쪽 Run = 학습, 뒤쪽 Run = 검증 (샷 수 기준 약 80/20, Run을 자르지 않음)"""
    start = ref.groupby('run_id')['TimeStamp'].min().sort_values()
    size = ref.groupby('run_id').size().reindex(start.index)
    cum = size.cumsum() / size.sum()
    val_runs = set(cum[cum > 1 - val_frac].index)
    is_val = ref['run_id'].isin(val_runs).values
    if is_val.all() or (~is_val).all():                       # Run이 1~2개뿐인 경우 대비
        is_val = np.zeros(len(ref), bool)
        is_val[int(len(ref) * (1 - val_frac)):] = True
    return is_val


class Scaler:
    """학습 구간 중앙값·IQR → ±CLIP → NaN=0 (FinalModel._prep과 같은 규칙)"""

    def fit(self, X):
        self.med = np.nanmedian(X, axis=0)
        q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
        sd = np.nanstd(X, axis=0)
        self.iqr = np.where(q75 - q25 > 1e-9, q75 - q25, np.where(sd > 1e-9, sd, 1.0))
        return self

    def transform(self, X):
        Z = (X - self.med) / self.iqr
        return np.nan_to_num(np.clip(Z, -CLIP, CLIP), nan=0.0).astype(np.float32)


def _X(df, cols):
    return df[cols].replace([np.inf, -np.inf], np.nan).values.astype(float)


def make_windows(Z, run_ids, W):
    """샷 t의 창 = 같은 Run의 [t−W+1 … t] (Run 앞부분은 0으로 왼쪽 채움). 반환 (N, W, F). 입력은 Run→시간 순 정렬"""
    N, F = Z.shape
    out = np.zeros((N, W, F), np.float32)
    run_ids = np.asarray(run_ids)
    change = np.r_[True, run_ids[1:] != run_ids[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], N]
    for a, b in zip(starts, ends):
        for t in range(a, b):
            lo = max(a, t - W + 1)
            out[t, W - (t + 1 - lo):] = Z[lo:t + 1]
    return out


# ====================================================================== 모델
def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError('PyTorch 필요: pip install torch  (CPU 버전으로 충분)') from e


def build_model(arch, F, W):
    torch = _torch()
    nn = torch.nn

    class LSTMAE(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.LSTM(F, HIDDEN, batch_first=True)
            s.to_z, s.from_z = nn.Linear(HIDDEN, LATENT), nn.Linear(LATENT, HIDDEN)
            s.dec = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
            s.out = nn.Linear(HIDDEN, F)

        def forward(s, x):
            _, (h, _) = s.enc(x)
            z = s.to_z(h[-1])
            y, _ = s.dec(s.from_z(z).unsqueeze(1).repeat(1, x.shape[1], 1))
            return s.out(y)

    class CNNAE(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(nn.Conv1d(F, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 16, 3, padding=1), nn.ReLU())
            s.to_z, s.from_z = nn.Linear(16 * W, LATENT), nn.Linear(LATENT, 16 * W)
            s.dec = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, F, 3, padding=1))

        def forward(s, x):                                    # x: (B, W, F)
            h = s.enc(x.transpose(1, 2))                      # (B, 16, W)
            z = s.to_z(h.flatten(1))
            h = s.from_z(z).view(-1, 16, x.shape[1])
            return s.dec(h).transpose(1, 2)

    class CNNLSTMAE(nn.Module):
        def __init__(s):
            super().__init__()
            s.conv = nn.Sequential(nn.Conv1d(F, 32, 3, padding=1), nn.ReLU())
            s.enc = nn.LSTM(32, HIDDEN, batch_first=True)
            s.to_z, s.from_z = nn.Linear(HIDDEN, LATENT), nn.Linear(LATENT, HIDDEN)
            s.dec = nn.LSTM(HIDDEN, HIDDEN, batch_first=True)
            s.out = nn.Linear(HIDDEN, F)

        def forward(s, x):
            c = s.conv(x.transpose(1, 2)).transpose(1, 2)
            _, (h, _) = s.enc(c)
            z = s.to_z(h[-1])
            y, _ = s.dec(s.from_z(z).unsqueeze(1).repeat(1, x.shape[1], 1))
            return s.out(y)

    return {'lstm_ae': LSTMAE, 'cnn_ae': CNNAE, 'cnn_lstm_ae': CNNLSTMAE}[arch]()


def fit_ae(arch, Xtr, Xva, seed, verbose=False):
    """정상 창만으로 복원 학습. 검증 손실 기준 조기 종료, 최적 가중치 복원"""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    N, W, F = Xtr.shape
    m = build_model(arch, F, W).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = torch.nn.MSELoss()
    tr = torch.tensor(Xtr, device=dev)
    va = torch.tensor(Xva, device=dev)
    best, best_state, wait, ep_done = np.inf, None, 0, 0
    g = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, BATCH):
            xb = tr[perm[i:i + BATCH].to(dev)]
            opt.zero_grad()
            loss = lossf(m(xb), xb)
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(np.mean([lossf(m(va[i:i + 1024]), va[i:i + 1024]).item() for i in range(0, len(va), 1024)]))
        ep_done = ep + 1
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.load_state_dict(best_state)
    if verbose:
        print(f'    {arch} seed{seed}: {ep_done} epoch, 검증 손실 {best:.4f}')
    return m, ep_done, best


def recon_scores(m, X):
    """(last, win) 복원오차"""
    torch = _torch()
    dev = next(m.parameters()).device
    last, win = [], []
    m.eval()
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            xb = torch.tensor(X[i:i + 1024], device=dev)
            e = ((m(xb) - xb) ** 2).cpu().numpy()
            last.append(e[:, -1, :].mean(1))
            win.append(e.mean((1, 2)))
    return {'last': np.concatenate(last), 'win': np.concatenate(win)}


def to_pct(s_ref_sorted, s):
    return np.searchsorted(s_ref_sorted, np.asarray(s, float), side='right') / (len(s_ref_sorted) + 1)


# ====================================================================== 평가
def eval_mask(lab):
    return ((lab['split'] != '제외') if 'split' in lab.columns else True) & (lab['Reason'] != INITIAL)


def evaluate(p, lab, score, pct, name, base_dir='.'):
    shot_score = pd.Series(np.asarray(score), index=lab['TimeStamp'].values)
    f1 = eval_frame1(p, load_official_rows(p, base_dir), shot_score)
    m = eval_mask(lab).values
    g = pd.DataFrame({'run_id': lab['run_id'].values[m], 'y': lab['PassOrFail'].values[m].astype(int),
                      'Reason': lab['Reason'].values[m], 'score': np.asarray(score)[m],
                      'pct': np.asarray(pct)[m]})
    f2 = eval_frame2(p, g)
    th = threshold_table(p, g)
    for d in (f1, f2):
        d['모델'] = name
    th.insert(0, '모델', name)
    return f1, f2, th, g


def delta_ap(y, s_new, s_base, n=1000, seed=SEED):
    """같은 샷을 층화 부트스트랩 → ΔAP(새 − 기준) 평균·95% CI·P(새 > 기준)"""
    from sklearn.metrics import average_precision_score as aps
    y, a, b = np.asarray(y), np.asarray(s_new), np.asarray(s_base)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(aps(y[i], a[i]) - aps(y[i], b[i]))
    d = np.asarray(d)
    return float(aps(y, a) - aps(y, b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


# ====================================================================== 실행
def run(products=('cn7', 'rg3'), archs=ARCHS, windows=WINDOWS, seeds=SEEDS, inputs=('free',),
        base_dir='.', model_dir='./outputs/final_model', out_dir='./outputs/seq_ae', verbose=True):
    """반환: summary(한눈 표), frame1, frame2, threshold, train_log, scores(제품별 샷 점수)"""
    torch = _torch()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(out_dir, exist_ok=True)
    F1, F2, TH, LOG, SUM, SCORES = [], [], [], [], [], {}
    for p in products:
        for inp in inputs:
            ref, lab, cols = load_data(p, inp, base_dir, model_dir)
            is_val = split_reference(ref)
            sc = Scaler().fit(_X(ref[~is_val], cols))
            Zr, Zl = sc.transform(_X(ref, cols)), sc.transform(_X(lab, cols))
            if verbose:
                print(f'[{p}|{inp}] reference 학습 {int((~is_val).sum()):,}샷 / 검증 {int(is_val.sum()):,}샷 '
                      f'| labeled {len(lab):,}샷 | 변수 {len(cols)}개')

            # 기준: 최종 모델 (같은 표에 같이)
            f1, f2, th, g0 = evaluate(p, lab, lab['maha_score'].values, lab['maha_pct'].values, BASELINE, base_dir)
            F1.append(f1); F2.append(f2); TH.append(th)
            out = lab[['TimeStamp', 'run_id', 'Reason', 'PassOrFail']].copy()
            out['eval'] = eval_mask(lab).values
            out[BASELINE] = lab['maha_pct'].values
            SUM.append(_summary_row(p, inp, BASELINE, None, None, f1, f2, th, None, np.nan))

            for W in windows:
                Xr = make_windows(Zr, ref['run_id'].values, W)
                Xl = make_windows(Zl, lab['run_id'].values, W)
                Xtr, Xva = Xr[~is_val], Xr[is_val]
                for arch in archs:
                    t0 = time.time()
                    acc = {k: {'va': 0, 'lab': 0} for k in SCORE_TYPES}
                    for sd in seeds:
                        m, ep, vl = fit_ae(arch, Xtr, Xva, sd, verbose)
                        sva, slab = recon_scores(m, Xva), recon_scores(m, Xl)
                        for k in SCORE_TYPES:
                            acc[k]['va'] = acc[k]['va'] + sva[k] / len(seeds)
                            acc[k]['lab'] = acc[k]['lab'] + slab[k] / len(seeds)
                        LOG.append({'제품': p, '입력': inp, '창': W, '모델': arch, 'seed': sd, 'epoch': ep,
                                    '검증 손실': vl})
                    sec = time.time() - t0
                    for k in SCORE_TYPES:
                        name = f'{arch}|W{W}|{k}'
                        pct = to_pct(np.sort(acc[k]['va']), acc[k]['lab'])
                        f1, f2, th, g = evaluate(p, lab, acc[k]['lab'], pct, name, base_dir)
                        F1.append(f1); F2.append(f2); TH.append(th)
                        dl = delta_ap(g['y'].values, g['score'].values, g0['score'].values)
                        SUM.append(_summary_row(p, inp, name, arch, W, f1, f2, th, dl, sec))
                        out[name] = pct
                    if verbose:
                        print(f'  [{p}|{inp}] {arch} W={W} 완료 ({sec:.0f}초)', flush=True)
            SCORES[(p, inp)] = out
            out.to_csv(f'{out_dir}/{p}_{inp}_shot_pct.csv', index=False, encoding='utf-8-sig')

    R = {'summary': pd.DataFrame(SUM), 'frame1': pd.DataFrame(F1), 'frame2': pd.DataFrame(F2),
         'threshold': pd.concat(TH, ignore_index=True), 'train_log': pd.DataFrame(LOG), 'scores': SCORES}
    for k in ('summary', 'frame1', 'frame2', 'threshold', 'train_log'):
        R[k].to_csv(f'{out_dir}/{k}.csv', index=False, encoding='utf-8-sig')
    return R


def _summary_row(p, inp, name, arch, W, f1, f2, th, dl, sec):
    t = th.set_index('q')
    r = {'제품': p, '입력': inp, '모델': name, 'arch': arch or 'maha', '창': W,
         '① AP': f1['AP'], '① 상위5%': f1['상위5%'], '① 상위10%': f1['상위10%'],
         '② AP': f2['AP'], '② AP 95%CI': f'[{f2["AP_lo"]:.3f}, {f2["AP_hi"]:.3f}]',
         '② AP_가스': f2.get('AP_가스', np.nan), '② AP_미성형': f2.get('AP_미성형', np.nan),
         'Run별 상위10% 가스': f2.get('Run별 상위10%_가스', ''), 'Run별 상위10% 미성형': f2.get('Run별 상위10%_미성형', ''),
         'q0.999 경보': int(t.loc[0.999, '경보 수']), 'q0.999 TP': int(t.loc[0.999, 'TP']),
         'q0.999 가스': int(t.loc[0.999, '가스 탐지']), 'q0.999 미성형': int(t.loc[0.999, '미성형계열 탐지']),
         'q0.99 경보 비율': float(t.loc[0.99, '경보 비율'])}
    if dl is not None:
        r.update({'ΔAP(②, vs 최종)': dl[0], 'ΔAP 95%CI': f'[{dl[1]:+.3f}, {dl[2]:+.3f}]', 'P(최종보다 좋음)': dl[3],
                  '유의하게 좋음': 'O' if dl[1] > 0 else '-'})
    r['학습 시간(초, 시드합)'] = sec
    return r


# ====================================================================== 그림
def plot_windows(R, products=('cn7', 'rg3'), inp='free', save=None):
    """창 크기별 AP (모델·점수별 선) + 최종 모델 기준선"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    S = R['summary']
    metrics = [('② AP', '② 전체 AP'), ('② AP_가스', '② 가스 AP'), ('② AP_미성형', '② 미성형 AP')]
    fig, axes = plt.subplots(len(products), 3, figsize=(15, 4.2 * len(products)), squeeze=False)
    for i, p in enumerate(products):
        s = S[(S['제품'] == p) & (S['입력'] == inp)]
        base = s[s['모델'] == BASELINE].iloc[0]
        for j, (col, title) in enumerate(metrics):
            ax = axes[i, j]
            for (arch, k), g in s[s['arch'] != 'maha'].assign(k=lambda d: d['모델'].str.split('|').str[2]) \
                    .groupby(['arch', 'k']):
                g = g.sort_values('창')
                ax.plot(g['창'], g[col], marker='o', ls='-' if k == 'win' else '--', label=f'{arch} · {k}')
            if pd.notna(base[col]):
                ax.axhline(base[col], color='k', lw=1.2, label='최종 모델 (Mahalanobis)')
            ax.set_title(f'{p.upper()} — {title}', loc='left', fontsize=10)
            ax.set_xlabel('창 크기 W (과거 샷 수)')
            ax.set_xticks(sorted(s['창'].dropna().unique()))
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()


def plot_r1(R, p='cn7', inp='free', models=None, save=None):
    """가스가 몰린 Run에서 샷 순서별 pct (최종 모델 vs 선택 모델) — 가스 에피소드가 '구간'으로 올라오는지"""
    import matplotlib.pyplot as plt
    try:
        from plot_style import set_style
        set_style()
    except ImportError:
        pass
    d = R['scores'][(p, inp)]
    run = d.loc[d['Reason'] == '가스', 'run_id'].mode()
    if run.empty:
        print('가스 샷 없음'); return
    r = d[d['run_id'] == run.iloc[0]].reset_index(drop=True)
    S = R['summary']
    if models is None:                                         # 창 50 win 점수 3종 + ② AP 최고 1개
        s = S[(S['제품'] == p) & (S['입력'] == inp) & (S['arch'] != 'maha')]
        models = list(s[(s['창'] == s['창'].max()) & s['모델'].str.endswith('|win')]['모델'])
        best = s.sort_values('② AP', ascending=False)['모델'].iloc[0]
        if best not in models:
            models.append(best)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = np.arange(1, len(r) + 1)
    ax.plot(x, r[BASELINE], color='k', lw=1.4, label='최종 모델')
    for mname in models:
        ax.plot(x, r[mname], lw=1, label=mname)
    for reason, c in (('가스', '#e67e22'), ('미성형', '#1f4e79'), ('가스+미성형', '#1f4e79')):
        idx = x[r['Reason'] == reason]
        ax.scatter(idx, np.full(len(idx), 1.02), color=c, s=22, marker='v', label=reason if len(idx) else None)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel(f'{run.iloc[0]} 안 샷 순서'); ax.set_ylabel('정상 대비 백분위 (pct)')
    ax.set_title(f'{p.upper()} 가스 집중 Run — 샷별 이상 백분위', loc='left', fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
    plt.show()
