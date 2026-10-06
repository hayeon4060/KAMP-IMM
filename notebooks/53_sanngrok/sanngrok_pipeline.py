"""
사출성형 CN7·RG3 최종검사 전 불량 예측 파이프라인
- 마스터 문서(사출_AI_통합마스터_가이드북_포렌식_전처리.md)의 검증 계약을 따른다.
  * labeled_data.csv 원본에서 완전 재수록만 제거, LH/RH 보존
  * CN7 / RG3 분리, side는 PART_NAME에서만 추출
  * 과거 날짜 → 다음 날짜 확장창 평가(주 평가), 모든 변환은 훈련 파티션에서만 적합
  * 임계값은 훈련자료 안의 event 그룹 3-fold 내부검증 F1 최대(동률이면 높은 값)
- 사용법: python molding_pipeline.py --data data/labeled_data.csv --out outputs
"""
import argparse, json, math, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")
SEED = 2026

SENSORS = [
    "Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time", "Clamp_Close_Time",
    "Cushion_Position", "Switch_Over_Position", "Plasticizing_Position", "Clamp_Open_Position",
    "Max_Injection_Speed", "Max_Screw_RPM", "Average_Screw_RPM", "Max_Injection_Pressure",
    "Max_Switch_Over_Pressure", "Max_Back_Pressure", "Average_Back_Pressure",
    *[f"Barrel_Temperature_{i}" for i in range(1, 8)], "Hopper_Temperature",
    *[f"Mold_Temperature_{i}" for i in range(1, 13)],
]
BARREL6 = [f"Barrel_Temperature_{i}" for i in range(1, 7)]
# 마스터 문서 12.2절의 정상 공정관계(P7)
RELATIONS = {
    "fill": ("Filling_Time", ["Max_Injection_Speed", "Max_Injection_Pressure", "Max_Switch_Over_Pressure",
                              "Plasticizing_Position", "Barrel_Temperature_2", "Barrel_Temperature_4",
                              "Mold_Temperature_3", "Mold_Temperature_4"]),
    "plast": ("Plasticizing_Time", ["Max_Screw_RPM", "Max_Back_Pressure", "Plasticizing_Position",
                                    "Barrel_Temperature_2", "Barrel_Temperature_4", "Hopper_Temperature"]),
    "cushion": ("Cushion_Position", ["Plasticizing_Position", "Filling_Time", "Max_Injection_Pressure",
                                     "Max_Switch_Over_Pressure"]),
    "switch": ("Max_Switch_Over_Pressure", ["Max_Injection_Pressure", "Max_Injection_Speed", "Filling_Time",
                                            "Barrel_Temperature_2", "Mold_Temperature_3"]),
}
# 모든 모델 입력에서 제외: 포렌식 9.8절의 평균 RPM 수치체계 변화(약 292→29, CN7 10/29·RG3 11/04부터).
# v1에서는 지도모델에 남아 있어 체계 변화일 전체가 강제 음성(점수 약 -1,600)이 되는 착시가 있었다(검증에서 발견).
REGIME_EXCLUDE = ["Average_Screw_RPM"]
ANOMALY_EXCLUDE = REGIME_EXCLUDE
MODEL_SENSORS = [c for c in SENSORS if c not in REGIME_EXCLUDE]


# ----------------------------------------------------------------------------- 데이터 준비
def prepare(path):
    raw = pd.read_csv(path)
    n_raw = len(raw)
    d = raw.drop_duplicates().copy()               # 완전히 같은 원본행(재수록)만 제거
    assert not d["_id"].duplicated().any(), "같은 ID에 다른 내용 → 감사 필요"
    d = d[(d.EQUIP_CD == "S14") & d.PART_NAME.str.match(r"^(CN7|RG3)")].copy()
    d["family"] = d.PART_NAME.str[:3]
    d["side"] = np.where(d.PART_NAME.str.contains("RH"), "RH", "LH")
    d["ts"] = pd.to_datetime(d.TimeStamp)
    d["date"] = d.ts.dt.strftime("%Y-%m-%d")
    d["event"] = d.EQUIP_CD + "|" + d.family + "|" + d.TimeStamp
    d["y"] = d.PassOrFail.map({"Y": 0, "N": 1}).astype(int)
    d = d.sort_values(["ts", "side"]).reset_index(drop=True)
    d = add_sequence_features(d)
    return d, n_raw


SEQ_K, SESSION_GAP_S = 10, 1800
SEQ_SENSORS = [c for c in SENSORS if c not in ANOMALY_EXCLUDE]


def add_sequence_features(d):
    """P10 순서 특징: 같은 제품군·같은 관측세션(공백 30분 초과 시 새 세션)에서
    '현재보다 앞선' 최대 10개 event의 센서 중앙값 대비 현재값의 차이.
    라벨은 쓰지 않으며 예측 시점에 이미 확보된 과거 센서값만 사용한다."""
    parts = []
    for fam, g in d.groupby("family"):
        ev = g.drop_duplicates("event")[["event", "ts", *SEQ_SENSORS]].reset_index(drop=True)
        ev["session"] = (ev.ts.diff().dt.total_seconds() > SESSION_GAP_S).cumsum()
        prior_med = (ev.groupby("session")[SEQ_SENSORS]
                     .transform(lambda s: s.shift(1).rolling(SEQ_K, min_periods=3).median()))
        dev = ev[SEQ_SENSORS] - prior_med
        dev.columns = [f"dev_{c}" for c in SEQ_SENSORS]
        dev["event"] = ev["event"]
        dev["seq_pos"] = ev.groupby("session").cumcount()   # 세션 내 몇 번째 관측인지(기동시각 아님)
        parts.append(dev)
    dev = pd.concat(parts)
    return d.merge(dev, on="event", how="left")


DEV_COLS = [f"dev_{c}" for c in SEQ_SENSORS]


# ----------------------------------------------------------------------------- 특징 블록
class Base:
    """훈련 파티션 기준: 비유한값 중앙값 대치, 상수열 제거, side 더미."""

    def __init__(self, cols=None, side=True):
        cols = MODEL_SENSORS if cols is None else cols
        self.cols, self.side = cols, side

    def fit(self, df):
        X = df[self.cols].replace([np.inf, -np.inf], np.nan)
        self.med = X.median()
        X = X.fillna(self.med)
        self.keep = [c for c in self.cols if X[c].nunique() > 1]
        return self

    def transform(self, df):
        X = df[self.cols].replace([np.inf, -np.inf], np.nan).fillna(self.med)[self.keep].copy()
        if self.side:
            X["side_RH"] = (df["side"] == "RH").astype(float).values
        return X


class NormalRelations:
    """P7: 정상(target=0) event 대표행으로 적합한 Ridge 관계 잔차 + 배럴온도 PCA 복원오차."""

    def fit(self, df, y):
        ref = df[y == 0].drop_duplicates("event")
        ref = ref[SENSORS].replace([np.inf, -np.inf], np.nan)
        self.med = ref.median()
        ref = ref.fillna(self.med)
        self.models = {}
        for k, (tgt, inp) in RELATIONS.items():
            inp = [c for c in inp if ref[c].std() > 0]
            mu, sd = ref[tgt].mean(), max(ref[tgt].std(), 1e-6)
            if not inp:
                self.models[k] = (tgt, inp, None, None, mu, sd)
                continue
            sc = StandardScaler().fit(ref[inp])
            rg = Ridge(alpha=10).fit(sc.transform(ref[inp]), (ref[tgt] - mu) / sd)
            self.models[k] = (tgt, inp, sc, rg, mu, sd)
        b = [c for c in BARREL6 if ref[c].std() > 0]
        self.bcols = b
        self.bsc = StandardScaler().fit(ref[b])
        self.pca = PCA(n_components=min(2, len(b))).fit(self.bsc.transform(ref[b]))
        return self

    def transform(self, df):
        X = df[SENSORS].replace([np.inf, -np.inf], np.nan).fillna(self.med)
        out = {}
        for k, (tgt, inp, sc, rg, mu, sd) in self.models.items():
            z = (X[tgt] - mu) / sd
            pred = rg.predict(sc.transform(X[inp])) if rg is not None else 0.0
            out[f"res_{k}"] = z - pred
            out[f"absres_{k}"] = np.abs(z - pred)
        Z = self.bsc.transform(X[self.bcols])
        out["temp_recon"] = ((Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2).mean(1)
        return pd.DataFrame(out, index=df.index)


def residual_features(tr, ytr, te):
    """훈련행 잔차는 event 3-fold 교차적합, 평가행은 훈련 전체 정상 참조로 변환."""
    R_tr = pd.DataFrame(index=tr.index, dtype=float)
    parts = []
    for a, b in GroupKFold(3).split(tr, ytr, tr["event"]):
        nr = NormalRelations().fit(tr.iloc[a], ytr[a])
        parts.append(nr.transform(tr.iloc[b]))
    R_tr = pd.concat(parts).loc[tr.index]
    R_te = NormalRelations().fit(tr, ytr).transform(te)
    return R_tr, R_te


# ----------------------------------------------------------------------------- 모델 정의
def seq_block(tr, te):
    """세션 초반(직전 이력 3개 미만)은 '변화 정보 없음'=0으로 두고, 훈련 표준편차로 스케일."""
    A, B = tr[DEV_COLS].fillna(0.0), te[DEV_COLS].fillna(0.0)
    keep = [c for c in DEV_COLS if A[c].std() > 0]
    sc = StandardScaler().fit(A[keep])
    A = pd.DataFrame(sc.transform(A[keep]), columns=keep, index=tr.index)
    B = pd.DataFrame(sc.transform(B[keep]), columns=keep, index=te.index)
    A["dev_maxabs"], B["dev_maxabs"] = A.abs().max(1), B.abs().max(1)
    return A, B


class SupervisedPipe:
    def __init__(self, est, scaler=None, resid=False, side=True, seq=False):
        self.est, self.scaler, self.resid, self.side, self.seq = est, scaler, resid, side, seq

    def fit_score(self, tr, ytr, te):
        base = Base(side=self.side).fit(tr)
        Xtr, Xte = base.transform(tr), base.transform(te)
        if self.resid:
            Rtr, Rte = residual_features(tr, ytr, te)
            Xtr, Xte = pd.concat([Xtr, Rtr], axis=1), pd.concat([Xte, Rte], axis=1)
        if self.scaler is not None:
            sc = clone(self.scaler).fit(Xtr)
            Xtr = pd.DataFrame(sc.transform(Xtr), index=tr.index)
            Xte = pd.DataFrame(sc.transform(Xte), index=te.index)
        if self.seq:
            Str, Ste = seq_block(tr, te)
            Xtr, Xte = pd.concat([Xtr, Str], axis=1), pd.concat([Xte, Ste], axis=1)
        m = clone(self.est).fit(np.asarray(Xtr), ytr)
        if hasattr(m, "decision_function"):
            return m.decision_function(np.asarray(Xte))
        return m.predict_proba(np.asarray(Xte))[:, 1]


class CatPipe(SupervisedPipe):
    def fit_score(self, tr, ytr, te):
        from catboost import CatBoostClassifier
        base = Base(side=self.side).fit(tr)
        Xtr, Xte = base.transform(tr), base.transform(te)
        if self.resid:
            Rtr, Rte = residual_features(tr, ytr, te)
            Xtr, Xte = pd.concat([Xtr, Rtr], axis=1), pd.concat([Xte, Rte], axis=1)
        m = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=10,
                               auto_class_weights="Balanced", random_seed=SEED, verbose=0,
                               thread_count=4, allow_writing_files=False)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]


class DAEPipe:
    """가이드북 2.3.1 잡음제거 오토인코더를 누수 없이 재구현.
    - 훈련 파티션의 양품 event 대표행만 사용, 하나의 스케일러를 훈련에서 적합(라벨별 정규화 금지)
    - 구조 d→15→5→15→d, 입력 30% 마스킹 잡음(원문의 Dropout 0.3 대응)
    - 점수 = log(복원 MSE). 확률이 아님."""

    def fit_score(self, tr, ytr, te):
        cols = [c for c in SENSORS if c not in ANOMALY_EXCLUDE]
        ref = tr[ytr == 0].drop_duplicates("event")
        base = Base(cols=cols, side=False).fit(ref)
        R, T = base.transform(ref), base.transform(te)
        sc = StandardScaler().fit(R)
        R, T = sc.transform(R), sc.transform(T)
        rng = np.random.default_rng(SEED)
        Xin = np.vstack([R * (rng.random(R.shape) > 0.3) for _ in range(5)])
        Xout = np.vstack([R] * 5)
        ae = MLPRegressor(hidden_layer_sizes=(15, 5, 15), activation="relu", learning_rate_init=1e-3,
                          max_iter=300, early_stopping=True, validation_fraction=0.2,
                          n_iter_no_change=10, random_state=SEED).fit(Xin, Xout)
        err = ((T - ae.predict(T)) ** 2).mean(1)
        return np.log(err + 1e-9)


class HybridPipe:
    """지도(P5+P7 로지스틱) 점수와 정상-only 이상점수(DAE)를 훈련 OOF 분포로 강건 표준화 후 평균."""

    def __init__(self, sup, ano, w=0.5):
        self.sup, self.ano, self.w = sup, ano, w

    def components(self, tr, ytr, te):
        return self.sup.fit_score(tr, ytr, te), self.ano.fit_score(tr, ytr, te)


MODELS = {
    "M0_LR_Std(P1,베이스라인)": SupervisedPipe(
        LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=4000, tol=1e-5,
                           random_state=SEED), scaler=StandardScaler()),
    "M1_RF(가이드북최고)": SupervisedPipe(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                               class_weight="balanced_subsample", n_jobs=8, random_state=SEED)),
    "M2_CatBoost": CatPipe(None),
    "M3_LR_Robust(P5)": SupervisedPipe(
        LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=4000, tol=1e-5,
                           random_state=SEED), scaler=RobustScaler()),
    "M4_LR_Robust+Resid(P5+P7)": SupervisedPipe(
        LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=4000, tol=1e-5,
                           random_state=SEED), scaler=RobustScaler(), resid=True),
    "M5_DAE(정상only)": DAEPipe(),
    "M7_LR_Robust+Resid+Seq(P5+P7+P10)": SupervisedPipe(
        LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=4000, tol=1e-5,
                           random_state=SEED), scaler=RobustScaler(), resid=True, seq=True),
}


# ----------------------------------------------------------------------------- 평가 도구
def threshold_f1(y, s):
    order = np.argsort(-s)
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    f1 = 2 * tp / (tp + fp + P)
    # 동점 점수 경계만 후보로
    last = np.r_[ss[1:] != ss[:-1], True]
    f1 = np.where(last, f1, -1)
    best = f1.max()
    idx = np.where(f1 == best)[0]
    return ss[idx.min()]  # 동률이면 더 높은 임계값


def robust_norm(ref):
    med = np.median(ref); iqr = np.subtract(*np.percentile(ref, [75, 25]))
    return lambda s: (s - med) / (iqr if iqr > 0 else (np.std(ref) or 1.0))


def expected_topk(y, s, frac=0.2):
    n = len(y); k = math.ceil(n * frac)
    order = np.argsort(-s); ss = s[order]
    cut = ss[k - 1]
    above = s > cut; tie = s == cut
    return y[above].sum() + (k - above.sum()) * y[tie].sum() / tie.sum()


def inner_oof(model, tr, ytr):
    """event 그룹 3-fold OOF 점수(하이브리드는 두 성분 반환)."""
    sgk = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.full(len(tr), np.nan); oof2 = np.full(len(tr), np.nan)
    for a, b in sgk.split(tr, ytr, tr["event"]):
        if isinstance(model, HybridPipe):
            s1, s2 = model.components(tr.iloc[a], ytr[a], tr.iloc[b])
            oof[b], oof2[b] = s1, s2
        else:
            oof[b] = model.fit_score(tr.iloc[a], ytr[a], tr.iloc[b])
    return oof, oof2


def forward_eval(df, fam, models):
    d = df[df.family == fam]
    dates = sorted(d.date.unique())
    rows, preds, splits = [], [], []
    for i in range(1, len(dates)):
        tr = d[d.date < dates[i]]; te = d[d.date == dates[i]]
        ytr, yte = tr.y.values, te.y.values
        assert tr.ts.max() < te.ts.min() and not set(tr.event) & set(te.event)
        status = "평가" if 0 < ytr.sum() < len(ytr) else "준비: 과거불량 없음"
        splits.append(dict(family=fam, date=dates[i], status=status, n_train=len(tr), pos_train=int(ytr.sum()),
                           n_test=len(te), pos_test=int(yte.sum())))
        if status != "평가":
            continue
        for name, m in models.items():
            t0 = time.time()
            oof, oof2 = inner_oof(m, tr, ytr)
            if isinstance(m, HybridPipe):
                n1, n2 = robust_norm(oof), robust_norm(oof2)
                comb_oof = m.w * n1(oof) + (1 - m.w) * n2(oof2)
                s1, s2 = m.components(tr, ytr, te)
                score = m.w * n1(s1) + (1 - m.w) * n2(s2)
                thr = threshold_f1(ytr, comb_oof); inner_ap = average_precision_score(ytr, comb_oof)
            else:
                thr = threshold_f1(ytr, oof); inner_ap = average_precision_score(ytr, oof)
                score = m.fit_score(tr, ytr, te)
            pred = (score >= thr).astype(int)
            ref = comb_oof if isinstance(m, HybridPipe) else oof
            shift_iqr = (np.median(score) - np.median(ref)) / max(np.subtract(*np.percentile(ref, [75, 25])), 1e-9)
            preds.append(pd.DataFrame(dict(model=name, family=fam, date=dates[i], _id=te["_id"].values,
                                           ts=te.TimeStamp.values, side=te.side.values,
                                           reason=te.Reason.values, y=yte, score=score, thr=thr, pred=pred,
                                           inner_ap=inner_ap, day_shift_iqr=shift_iqr)))
            print(f"  {fam} {dates[i]} {name:28s} TP={int((pred & yte).sum())} "
                  f"FP={int((pred & (1 - yte)).sum())} pos={yte.sum()} innerAP={inner_ap:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return pd.DataFrame(splits), pd.concat(preds, ignore_index=True)


def summarize(P):
    out = []
    for (fam, name), g in P.groupby(["family", "model"], sort=False):
        y, p, s = g.y.values, g.pred.values, g.score.values
        tp = int((p & y).sum()); fp = int((p & (1 - y)).sum()); fn = int(((1 - p) & y).sum())
        top = sum(expected_topk(h.y.values, h.score.values) for _, h in g.groupby("date"))
        n_insp = sum(math.ceil(len(h) * 0.2) for _, h in g.groupby("date"))
        day_ap = [average_precision_score(h.y, h.score) for _, h in g.groupby("date") if h.y.sum() > 0]
        out.append(dict(family=fam, model=name, n=len(y), pos=int(y.sum()), TP=tp, FP=fp, FN=fn,
                        precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn) if tp + fn else 0.0,
                        F1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
                        AP_pooled=average_precision_score(y, s), AP_posday_macro=float(np.mean(day_ap)),
                        top20_capture=round(top, 2), top20_inspected=n_insp,
                        random_top20_expect=round(y.sum() * n_insp / len(y), 2)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/labeled_data.csv")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df, n_raw = prepare(a.data)
    audit = df.groupby("family").agg(records=("y", "size"), defects=("y", "sum"), days=("date", "nunique"),
                                     events=("event", "nunique"))
    print(f"원본 {n_raw}행 → 재수록 제거·CN7/RG3·S14 {len(df)}행\n{audit}")
    audit.to_csv(out / "data_audit.csv", encoding="utf-8-sig")

    models = dict(MODELS)
    models["M6_Hybrid(M4+M5)"] = HybridPipe(MODELS["M4_LR_Robust+Resid(P5+P7)"], MODELS["M5_DAE(정상only)"])
    models["M8_Hybrid(M7+M5)"] = HybridPipe(MODELS["M7_LR_Robust+Resid+Seq(P5+P7+P10)"], MODELS["M5_DAE(정상only)"])
    S, P = [], []
    for fam in ["CN7", "RG3"]:
        s, p = forward_eval(df, fam, models)
        S.append(s); P.append(p)
    S, P = pd.concat(S), pd.concat(P, ignore_index=True)
    S.to_csv(out / "forward_splits.csv", index=False, encoding="utf-8-sig")
    P.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    summ = summarize(P)
    summ.to_csv(out / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))
    json.dump(dict(seed=SEED, sensors=SENSORS, anomaly_exclude=ANOMALY_EXCLUDE, relations=RELATIONS,
                   models=list(models)), open(out / "protocol.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
