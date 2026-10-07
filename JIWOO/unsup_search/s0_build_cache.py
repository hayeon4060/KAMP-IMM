"""0단계: 비라벨 전체 흐름 캐시 + 라벨 샷 파생변수 캐시"""
import common as C

if __name__ == '__main__':
    F = C.build_cache(n_jobs=14, force=True)
    print(F.groupby(['equip', 'prod']).agg(n=('run_id', 'size'), runs=('run_id', 'nunique'),
                                           t0=('TimeStamp', 'min'), t1=('TimeStamp', 'max'))
          .sort_values('n', ascending=False).to_string())
    L = C.build_labeled_feats(force=True)
    for p, f in L.items():
        C.log(f'라벨 {p}: {len(f):,}샷')
