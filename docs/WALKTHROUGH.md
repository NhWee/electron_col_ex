# 따라하기: 처음부터 끝까지 직접 돌려 보기

이 문서는 저장소를 처음 받은 사람이 **명령을 하나씩 실행하면서 각 단계가
무엇을 하고, 화면에 무엇이 나오고, 그것을 어떻게 읽어야 하는지**를 설명한다.
물리 배경은 `README.md`의 0절에, 결과 해석은 1~4절에 있으므로 여기서는
"실행"에 집중한다.

전체를 한 번에 돌리려면 `./run_all.sh` 하나면 된다 (약 10분). 아래는 그
스크립트가 하는 일을 단계별로 풀어 쓴 것이다.

---

## 0. 준비

```bash
git clone https://github.com/nhwee/electron_col_ex.git
cd electron_col_ex
python3 -m venv .venv && source .venv/bin/activate   # Python 3.10 이상
pip install -r requirements-dev.txt
```

필요한 것은 Python 3.10 이상뿐이다. (`requirements-dev.txt`는 분석용 6개
패키지 + pytest + Jupyter. Jupyter가 필요 없으면 `requirements.txt`만 설치하고
pytest를 따로 설치해도 된다.) MadGraph5는 5~6단계에만 필요하고 없어도
나머지는 전부 돌아간다.

**어디서 실행하나.** 모든 명령은 저장소 최상위(`electron_col_ex/`)에서
실행한다. 스크립트는 `data/raw/`를 상대 경로가 아니라 자기 위치 기준으로 찾기
때문에 다른 곳에서 실행해도 되지만, 출력 경로를 헷갈리지 않으려면 최상위가
편하다.

**어디에 결과가 생기나.** 모든 표와 그림은 `data/processed/`에 생긴다. 이
폴더는 git이 추적하지 않으므로 마음껏 지우고 다시 만들어도 된다. README가
보여 주는 그림은 `figures/`에 따로 복사해 둔 것이다.

---

## 데이터가 어떻게 흐르는가

```
data/raw/*.csv  (OPAL 논문 표를 손으로 옮긴 것)
   │  N (사건 수), L (광도), f (보정계수)
   ▼
σ = f·N / L                         ← 1단계: compute_cross_section.py
   │  7개 에너지 점의 단면적 + 오차
   ▼
fit_breit_wigner()       (ISR 없음)  ┐
fit_breit_wigner_isr()   (ISR 있음)  ┘ → M_Z, Γ_Z
   │
   ▼
error_budget()                        → 오차가 어디서 오는지

같은 피팅 함수를 가짜 데이터에도 적용한다:
  simulate_mumu.py      정답을 아는 토이 MC         ← 2단계
  pull_study.py         그 토이를 500번            ← 3단계
  simulate_mumu_mg5.py  MadGraph5가 만든 사건      ← 5, 6단계
그리고 실제 데이터로 더 나아간다:
  extended_fits.py      점 추가, 다른 모형, 다채널 ← 4단계
```

핵심은 **피팅 코드가 하나**라는 점이다. 실제 데이터와 가짜 데이터가 똑같은
`fit_breit_wigner_isr()`를 지나가므로, 가짜 데이터에서 정답을 복원하면 실제
데이터의 결과도 믿을 근거가 생긴다.

---

## 단계 0. 테스트 (5초)

```bash
python -m pytest
```

**무엇을 하나.** 코드가 바뀌어도 물리가 깨지지 않았는지 29가지를 확인한다.
예를 들어 ISR 복사 함수가 정확히 1로 적분되는지, 전방-후방 비대칭이 피크
아래에서 음수·위에서 양수인지, 노이즈 없는 데이터에서 피팅이 입력값을 그대로
돌려주는지, 손으로 옮긴 데이터가 OPAL 발표값과 2% 이내로 맞는지.

**무엇이 나오나.** `29 passed in 4.13s`. 실패가 있으면 그 테스트 이름이
무엇이 깨졌는지 알려 준다.

**왜 먼저 하나.** 이후 단계의 숫자를 믿기 전에 환경(패키지 버전 등)이
정상인지 확인하는 가장 빠른 방법이다.

---

## 단계 1. 실제 OPAL 데이터 (10초)

```bash
python src/compute_cross_section.py
```

**무엇을 하나.**
1. `data/raw/opal_zedometry_table1_event_counts.csv`에서 N과 L을, 보정계수
   표에서 f를 읽어 σ = fN/L을 계산한다. 보정계수가 있는 7개 점만 쓴다.
2. 계산값을 OPAL이 발표한 단면적(`*_xsec_published.csv`)과 비교한다. 발표값은
   **검증에만** 쓰고 계산에는 넣지 않는다.
3. Breit-Wigner를 세 번 피팅한다: 7개 점 전부, 이상한 1개를 뺀 6개, 그리고
   6개에 ISR(초기상태 복사) 보정을 넣어서.
4. 오차 예산을 만든다: 통계, 광도 반올림, 보정계수, 복사 모형, 빔 에너지
   퍼짐, LEP 에너지 보정이 각각 M_Z와 Γ_Z에 몇 MeV씩 기여하는지.
5. 4개 채널(qq, ee, μμ, ττ)을 교차 검사한다.

**화면에서 볼 것.**

```
=== mu+mu- cross section: computed vs published ===
 year   sample  ...  sigma_computed_nb  ...  xsec_corrected_nb  diff_pct
 1993   peak-2  ...             0.4991  ...             0.4959    0.6395
 1993     peak  ...             1.3490  ...             1.4589   -7.5336   ← 이상치
 ...
```
`diff_pct`가 계산값과 발표값의 차이(%)다. 6개는 2% 이내, 1993 peak만 −7.5%.

```
=== Breit-Wigner fit excluding 1 known-anomalous point(s) ===
chi2/ndof = 0.46/3
  mZ         = 91.4594 +/- 0.0105 GeV   (PDG: 91.1876 +/- 0.0021 GeV)
```
ISR 없는 피팅. χ²는 좋은데 PDG보다 272 MeV 크다. **좋은 χ²가 맞는 답을
보장하지 않는다**는 예.

```
=== Same clean points, fit with the ISR-convolved Breit-Wigner (full radiator) ===
chi2/ndof = 1.42/3
  mZ         = 91.1865 +/- 0.0120 GeV   (PDG: 91.1876 +/- 0.0021 GeV)
  gammaZ     = 2.5287 +/- 0.0197 GeV   (PDG: 2.4955 +/- 0.0023 GeV)
  shift vs. non-ISR fit: mZ -272.9 MeV, gammaZ -270.8 MeV
```
**이것이 결과다.** ISR을 넣자 M_Z가 PDG와 0.1σ 이내로 들어왔다.

```
=== Error budget for the ISR-convolved fit (6 clean points) ===
source                                          kind        dM_Z (MeV)  dG_Z (MeV)
statistical (Poisson on N)                      input            11.61       19.24
+ L rounding (0.1 pb^-1 granularity)            input             2.63        0.00
radiator form (full - simplified)               model             7.10       10.12
beam energy spread (smeared - unsmeared)        model            -0.23       -5.85
...
  M_Z     = 91.1864 +/- 0.0119 (fit ...) +/- 0.0073 (model ...) GeV
```
통계가 가장 크고, 모형(복사 함수 형태)이 그다음이다. "이 측정의 정밀도를
올리려면 무엇을 고쳐야 하나"에 대한 답이 이 표다.

```
=== Cross-channel consistency check (% diff from published, by channel) ===
                 qq    ee  mumu  tautau
1993 peak     -0.62 -0.61 -7.53   -4.05
```
1993 peak에서 트랙 기반 채널(μμ, ττ)만 어긋나고 칼로리미터 채널(qq, ee)은
정상이다. 같은 L을 쓰므로 L이 아니라 트랙 검출기 쪽 문제라는 추론의 근거.

**생기는 파일.** `data/processed/mumu_cross_section_computed.csv`,
`error_budget.csv`, `channel_consistency_check.csv`, `mumu_cross_section.png`.

---

## 단계 2. 토이 몬테카를로 (20초)

```bash
python src/simulate_mumu.py
```

**무엇을 하나.** "결과가 PDG 근처에 나왔다"는 것은 약한 증거다. PDG 근처로
밀리도록 잘못 짠 피팅도 같은 결과를 낸다. 그래서 **정답을 미리 아는** 가짜
실험을 만든다:
1. M_Z, Γ_Z를 PDG 값으로 **고정**하고 ISR을 포함한 진짜 lineshape를 만든다.
2. 13개 에너지 점에서 Poisson 난수로 사건 수를 뽑고, 각 사건의 각도도 뽑아
   |cosθ| < 0.95 컷을 적용한다(가짜 검출기).
3. 50만 사건짜리 별도 표본으로 검출 효율을 측정한다. OPAL의 f와 같은 역할.
4. 1단계와 **똑같은** 피팅 함수로 M_Z, Γ_Z를 구해 입력값과 비교한다.
5. 전방/후방 사건 수로 A_FB를 만들고 sin²θ_W를 피팅한다.

**화면에서 볼 것.**

```
=== Naive Breit-Wigner fit to simulated points ===
chi2/ndof = 75.56/10
  pull vs. TRUE input: mZ +17.87 sigma, gammaZ +9.56 sigma
=== ISR-convolved Breit-Wigner fit to simulated points ===
chi2/ndof = 5.39/10
  pull vs. TRUE input: mZ +0.29 sigma, gammaZ -0.83 sigma
```
`pull`은 (피팅값 − 정답)/피팅오차. ISR 없는 피팅은 18σ 틀리고 ISR 피팅은
1σ 이내. 13개 점에서는 ISR 누락이 χ²에서도 드러난다(75/10).

```
=== A_FB(s) fit ===
  sin2(theta_W_eff) = 0.23021 +/- 0.00408   (true input: 0.23155)
  pull vs. TRUE input: -0.33 sigma
```

**주의.** 난수 시드가 고정되어 있어 누가 돌려도 같은 숫자가 나온다. 시드를
바꾸면 숫자는 달라지지만 결론(ISR 피팅만 정답 복원)은 같아야 한다. 그것을
확인하는 것이 3단계다.

**생기는 파일.** `simulated_mumu_scan.csv`, `simulated_mumu_closure.png`,
`simulated_mumu_afb.png`.

---

## 단계 3. 토이 500번 반복 (8분)

```bash
python src/pull_study.py                # 500회
python src/pull_study.py --n-toys 50    # 급하면 50회 (1분)
```

**무엇을 하나.** 2단계를 시드만 바꿔 500번 돌리고 pull의 분포를 본다.
- 평균이 0이면 **편향이 없다**.
- 폭이 1이면 **피팅이 보고하는 오차가 실제 흩어짐과 같다**(오차가 정직하다).
- 폭이 1보다 작으면 오차를 과대평가, 크면 과소평가한 것이다.

**화면에서 볼 것.**

```
              fit        parameter   n   mean  mean_err  width  width_err  frac_within_1sigma
        naive fit            $M_Z$ 500 17.437     0.043  0.957      0.030               0.000
ISR-convolved fit            $M_Z$ 500  0.038     0.044  0.988      0.031               0.678
ISR-convolved fit       $\Gamma_Z$ 500 -0.107     0.043  0.965      0.031               0.678
     $A_{FB}$ fit  sin2(theta_W)   500 -0.082     0.040  0.903      0.029               0.760
```
ISR 피팅: 평균 ≈ 0, 폭 ≈ 1, 68%가 1σ 안 → 합격. A_FB 피팅은 폭 0.90으로
오차를 10% 과대평가한다. 이유는 README 2절에 있다(sin²θ_W = 0.25에서 감도가
사라지는 구조). **한 번만 돌려서는 절대 볼 수 없는 종류의 사실**이다.

**생기는 파일.** `pull_study_toys.csv`(500행), `pull_study_summary.csv`,
`pull_study.png`.

---

## 단계 4. 확장 (1분)

```bash
python src/extended_fits.py
```

**무엇을 하나.** 세 가지 "더 해 보면 어떻게 되나"를 실제 데이터로 시도한다.

**4.1 점 추가.** 보정계수 f가 없는 6개 점에 같은 연도의 f를 빌려 쓴다.
```
  6 own-f points (baseline)        chi2/ndof   1.42/3   mZ 91.1865 +/- 0.0120
  12 points incl. borrowed f       chi2/ndof   4.52/9   mZ 91.1858 +/- 0.0119
```
거의 안 변한다. 추가된 점이 전부 피크 위에 있어서 M_Z·Γ_Z 정보가 늘지 않기
때문이다. "점을 늘리면 좋아진다"가 항상 참은 아니라는 예.

**4.2 모형 비교.** 비상대론적 BW(기본), running-width BW(LEP 표준),
광자+Z+간섭(IBA) 세 가지로 같은 6개 점을 피팅한다.
```
                             model  mZ_GeV  gammaZ_GeV   norm
    non-relativistic BW (baseline) 91.1864      2.5284 2.0105
     relativistic running-width BW 91.1865      2.5349 2.0095
IBA gamma+Z+interf., s'/s > 0.7225 91.1536      2.5107 1.0277
```
세 번째 줄의 M_Z가 33 MeV 낮은 것은 질량 **정의**가 다르기 때문이다(README
0.2절). `norm` 1.028은 "계산한 단면적이 표준모형 절대 예측의 1.028배"라는
뜻으로, PDG 입력 없이 절대 정규화가 3% 이내로 맞는다는 독립 검증이다.

**4.3 결합 피팅.** qq, μμ, ττ를 공통 M_Z, Γ_Z로 한꺼번에 피팅한다.
```
  mZ                 = 91.1974 +/- 0.0035 GeV
  gammaZ             = 2.5144 +/- 0.0058 GeV
  pull vs. PDG: mZ +2.77 sigma, gammaZ +3.25 sigma
  sigma_mumu / sigma_tautau at peak = 1.0001 +/- 0.0060   (lepton universality: 1)
```
오차가 3.5 MeV까지 줄었지만 PDG와 2.8σ 차이가 난다. 이것은 데이터가 틀린 게
아니라 **피팅 오차가 모형 오차(1단계 오차 예산의 7~10 MeV)보다 작아졌다**는
뜻이다. 여기서부터는 데이터가 아니라 모형(O(α²) 복사 보정)을 고쳐야 한다.

**생기는 파일.** `mumu_cross_section_extended.csv`,
`lineshape_model_comparison.csv`, `joint_fit.csv`, 그림 2개.

---

## 단계 5, 6. MadGraph5 (선택, 각 1~2분)

```bash
export MG5_PATH=/path/to/MG5_aMC_v3_5_0/bin/mg5_aMC
python src/simulate_mumu_mg5.py          # ISR 끔
python src/simulate_mumu_mg5.py --isr    # ISR 켬 (MG5 3.2 이상, 아직 아무도 안 돌려 봄)
```

**무엇을 하나.** 2단계의 토이는 "정답 모형"과 "피팅 모형"을 같은 사람이
짰다는 약점이 있다. 둘 다 같은 실수를 하면 검증이 통과해 버린다. 그래서
정답 모형을 독립적으로 구현된 MadGraph5의 행렬 요소로 바꾼다. 스크립트가
MG5를 13번 호출해 각 에너지에서 사건을 만들고, 나온 μ⁻의 각도를 읽어 2단계와
같은 처리를 한다.

**ISR 끔(5단계)에서 기대하는 것.** 생성기에 ISR이 없으므로 **ISR 없는 피팅이
맞고 ISR 피팅은 −8.5σ로 틀려야 한다**. 실제 데이터와 정반대. ISR 보정이
"항상 PDG 쪽으로 미는" 게 아니라 물리적으로 맞는 방향으로 움직인다는 증거.

**ISR 켬(6단계)에서 기대하는 것.** 이번엔 실제 데이터처럼 ISR 피팅이 정답을
복원하고 ISR 없는 피팅이 위로 치우쳐야 한다. 이 코드는 MG5가 없는 환경에서
작성되어 **아직 실행된 적이 없다**. 처음 돌리는 사람이 결과를 README 3절에
추가해 주면 된다. MG5가 run_card 파라미터 이름을 모르겠다고 하면 MG5 버전이
3.2 미만이거나 이름이 바뀐 것이다.

---

## 문제가 생기면

| 증상 | 원인과 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'compute_cross_section'` | 스크립트를 `python src/xxx.py`로 실행해야 한다. `src/` 안에 들어가서 실행하거나 `python -m src.xxx`로 실행하면 import 경로가 달라진다. |
| `pytest`에서 import 오류 | `tests/conftest.py`가 `src/`를 경로에 넣는다. 저장소 최상위에서 `python -m pytest`로 실행. |
| 3단계가 너무 느리다 | `--n-toys 50`. 결론을 보는 데는 충분하고 통계적 정밀도만 떨어진다. |
| `MadGraph5 executable not found` | 5~6단계만 해당. `MG5_PATH` 설정. 없으면 건너뛰어도 1~4단계 결과는 완전하다. |
| 숫자가 README와 소수점 넷째 자리에서 다르다 | 패키지 버전 차이(특히 scipy의 적분, iminuit). `requirements.txt` 주석의 버전으로 맞추면 같아진다. 결론이 달라지면 문제지만 마지막 자리는 아니다. |
| 그림이 안 열린다 / `DISPLAY` 오류 | 스크립트는 화면에 띄우지 않고 파일로만 저장한다. `data/processed/*.png`를 열면 된다. |

---

## 직접 바꿔 보기

코드를 이해했는지 확인하는 가장 좋은 방법은 한 줄을 바꾸고 결과가 **예상한
방향으로** 움직이는지 보는 것이다.

1. `src/compute_cross_section.py`에서 `ISR_RADIATOR = "simplified"`로 바꾸고
   1단계를 다시 돌린다. M_Z가 7 MeV 내려가야 한다(오차 예산의 radiator 줄).
2. `ANOMALY_THRESHOLD_PCT = 10.0`으로 바꾸면 1993 peak가 피팅에 들어간다.
   χ²/ndof가 69/4처럼 나빠지고 M_Z가 움직이는 것을 본다.
3. `src/simulate_mumu.py`에서 `LUMI_PBINV = 20.0`(10배)으로 바꾸고 2단계를
   돌린다. 오차가 약 √10 ≈ 3.2배 줄어야 한다. 3단계의 pull 폭은 그대로 1
   근처여야 한다(오차가 줄어도 정직함은 유지).
4. `SCAN_POINTS_GEV`를 피크 ±1 GeV로 좁히면 Γ_Z 오차가 크게 나빠진다. 폭은
   날개에서 결정된다는 것을 숫자로 본다.
5. `src/extended_fits.py`에서 `JOINT_CHANNELS = ["mumu", "tautau"]`로 qq를
   빼면 결합 피팅의 오차가 1단계 수준으로 돌아간다. qq가 통계를 지배한다.

각 실험 뒤에는 `git checkout -- src/`로 되돌리고 `python -m pytest`로
원상복구를 확인한다.
