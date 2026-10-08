# CN7 비라벨 가스 후보 경보 비율

최종7개 지도 모델을 전체 공식605샷으로 새 학습하고 적용했다. 원래F1 .714의 구간별 모델을 그대로 적용한 것이 아니다.
기준은 기존 중첩 검증 각20블록의 임계값 중앙값으로 고정했다. 비라벨 결과를 보고 기준을 조정하지 않았다.
공식 비라벨35239행/S14만 사용한다. 정규화된 공식 파일은Timestamp 선별에만 사용하고 실제 모델에는 연결한원시 센서를 넣었다.
같은Timestamp의LH/RH를 한샷으로 처리한다. 이미 라벨이 있는606샷은 후보 비율에서 제외한다.
예전Run ID를 새Run에 직접 할당할 수 없어 배럴5/호퍼/금형4가 가장 비슷한 학습 정상의운전모드를 선택했다.
새Run의정답을 정상이라고 가정해 정규화하지 않았다. 참조정상은 라벨592샷만 사용한다.
쿼리는원시비라벨 과거5샷이력을사용하므로 기존희소라벨 이력과밀도가달라진다. 초기3샷 미만/센서충돌은보류한다.
정답이 없으므로 이하비율은가스경보비율이며 정확도/실제불량률이 아니다.

## 요약
{
  "official_rows": 35239,
  "official_unique_shots": 28744,
  "known_labeled_shots_excluded": 606,
  "unknown_shots": 28138,
  "unknown_sensor_conflict": 28,
  "unknown_history_unavailable": 370,
  "eligible_unknown_shots": 27768,
  "gas_candidate_shots": 7765,
  "candidate_percent_of_eligible": 27.963843272832037,
  "candidate_percent_of_all_unknown": 27.596133342810436,
  "mode_support_cut": 0.5017312249266509,
  "supported_eligible": 2582,
  "supported_candidates": 1491,
  "model": "full605 fit same7 heads; mode matching to nearest labeled normal in3 control variables; frozen median fold cuts",
  "history": "past5 query sensor history from rawS14, reset30min/product switch; original sparse label history retained for training",
  "caveat": "No quality ground truth, not precision/recall/F1. Full-fit predictions are not the original0.714 OOF ensemble.",
  "supported_candidate_percent": 57.745933384972886,
  "eligible_rows": 33464,
  "candidate_rows": 11736,
  "candidate_row_percent": 35.07052354769304
}

## 모델별(중복있음)
        model  candidate_shots   percent
  relation_ET                0  0.000000
     joint_LR             3862 13.908096
      load_LR             3250 11.704120
persistent_ET                3  0.010804
  transfer_LR             4737 17.059205
local_flow_RF               22  0.079228
  abs_flow_LR              251  0.903918

## 후보가많은Run
                          first                last  shots  candidates  supported  candidate_percent
history_run                                                                                         
216         2020-10-17 22:51:46 2020-10-19 23:23:48   3007        1992       1504          66.245427
23          2020-04-03 23:29:49 2020-04-04 23:28:43   1195        1195          0         100.000000
111         2020-09-07 03:10:06 2020-09-07 16:09:19    722         703          0          97.368421
174         2020-10-16 03:54:18 2020-10-17 01:31:56   1219         679        487          55.701395
54          2020-07-29 15:05:35 2020-07-30 01:40:40    561         561          0         100.000000
31          2020-05-27 08:23:16 2020-05-27 18:30:00    527         466          0          88.425047
170         2020-10-15 17:28:59 2020-10-15 23:26:18    357         352        352          98.599440
212         2020-10-17 17:21:15 2020-10-17 22:29:30    813         249         34          30.627306
121         2020-09-19 00:45:52 2020-09-19 07:49:54    385         222          0          57.662338
93          2020-08-25 23:26:40 2020-08-26 02:32:23    170         166          0          97.647059
196         2020-10-17 08:24:30 2020-10-17 10:27:05    324         123          0          37.962963
97          2020-08-27 23:27:18 2020-08-28 01:37:16    122         117          0          95.901639
204         2020-10-17 12:06:41 2020-10-17 13:56:25    290         109         24          37.586207
208         2020-10-17 14:24:39 2020-10-17 16:13:21    289          92         82          31.833910
188         2020-10-17 05:23:54 2020-10-17 07:12:38    289          76          0          26.297578

참조범위밖운전조건에서도 모델은큰점수를 낼수 있다. 지원범위내비율을따로 보고했지만 지원범위내라는것이정상/불량을보증하지는않는다.