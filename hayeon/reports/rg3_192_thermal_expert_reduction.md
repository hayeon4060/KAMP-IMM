# RG3 26개에서 추가 축소

동일583샷(정상566/가스17),5구간 제외 개발 비교. 기존열9개를실제로재학습해각구간의원본점수/임계값/단독예측모두재현한후열반복을축소했다. 압력은191에서재학습한단일RF/ET/LR3개,기본RF2예측과결합한것을사용했다. 패턴전문가12개→단일seed4개비교는47번의기존구간제외점수및그seed의학습내부임계값을사용했다(이번에전문가를새로학습하지않음).

               candidate  classifiers  TP  FP  FN  TN       F1
           none+thermal2            7   6   6  11 560 0.413793
           none+thermal3            8   7   7  10 559 0.451613
         seed42+thermal2           11   7   6  10 560 0.466667
        seed101+thermal2           11   7   6  10 560 0.466667
           none+thermal6           11   6   6  11 560 0.413793
          seed7+thermal2           11   6   6  11 560 0.413793
         seed42+thermal3           12   8   7   9 559 0.500000
        seed101+thermal3           12   8   7   9 559 0.500000
           none+thermal7           12   7   6  10 560 0.466667
          seed7+thermal3           12   7   7  10 559 0.451613
           none+thermal9           14   7   6  10 560 0.466667
         seed42+thermal6           15   7   6  10 560 0.466667
        seed101+thermal6           15   7   6  10 560 0.466667
          seed7+thermal6           15   6   6  11 560 0.413793
         seed42+thermal7           16   8   6   9 560 0.516129
        seed101+thermal7           16   8   6   9 560 0.516129
          seed7+thermal7           16   7   6  10 560 0.466667
         seed42+thermal9           18   8   6   9 560 0.516129
        seed101+thermal9           18   8   6   9 560 0.516129
          seed7+thermal9           18   7   6  10 560 0.466667
three_seed_mean+thermal2           19   7   6  10 560 0.466667
three_seed_mean+thermal3           20   8   7   9 559 0.500000
three_seed_mean+thermal6           23   7   6  10 560 0.466667
three_seed_mean+thermal7           24   8   6   9 560 0.516129
three_seed_mean+thermal9           26   8   6   9 560 0.516129

## 한계
독립최종검증이아니며평가결과를본축소후보선택이다. 전체조합을빈환경에서처음부터재학습한검증은아직아니다. 기존이력피처조건재사용,비라벨경보율미확인,모델파일미교체,commit/push없음. 분류기수에Ridge4전처리회귀는별도다.
