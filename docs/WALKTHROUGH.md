# 따라하기: 처음부터 끝까지 직접 돌려 보기

이 문서는 저장소를 처음 받은 사람이 **명령을 하나씩 실행하면서 각 단계가
무엇을 하고, 화면에 무엇이 나오고, 그것을 어떻게 읽어야 하는지**를 설명한다.
물리 배경은 `README.md`의 0절에, 결과 해석은 1~4절에 있으므로 여기서는
"실행"에 집중한다.

전체를 한 번에 돌리려면 `./run_all.sh` 또는 `make all` 하나면 된다 (약 10분).
아래는 그 명령이 하는 일을 단계별로 풀어 쓴 것이다.

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
보여 주는 그림은 `figures/`에 있고, `make figures`가 필요한 스크립트만 다시
돌린 뒤 복사해 준다.

---

## 데이터가 어떻게 흐르는가

```
data/raw/*.csv  (OPAL 논문 표를 손으로 옮긴 것)
   │  N (사건 수), L (광도), f (보정계수), 채널별 평균 √s와 빔 에너지 퍼짐
   ▼
σ = f·N / L                         ← 1단계: compute_cross_section.py
   │  7개 에너지 점의 단면적 + 오차 (통계 / f / L 반올림을 따로)
   ▼
fit_breit_wigner()       (ISR 없음)  ┐
fit_breit_wigner_isr()   (ISR 있음)  ┘ → M_Z, Γ_Z
   │
   ▼
error_budget()                        → 오차가 어디서 오는지, 그리고
                                        Breit-Wigner가 빼먹은 물리(광자 교환)의 크기

같은 피팅 함수를 가짜 데이터에도 적용한다:
  simulate_mumu.py      정답을 아는 토이 MC         ← 2단계
  pull_study.py         그 토이를 500번            ← 3단계
  simulate_mumu_mg5.py  MadGraph5가 만든 사건      ← 5, 6단계
그리고 실제 데이터로 더 나아간다:
  extended_fits.py      점 추가, 다른 모형, 다채널 ← 4단계

공통 부품:
  compute_cross_section.py   ISR 복사 함수(radiator)와 lineshape 모형
  electroweak.py             광자 + Z + 간섭 단면적 (README 0.3절)
```

핵심은 **피팅 코드가 하나**라는 점이다. 실제 데이터와 가짜 데이터가 똑같은
`fit_breit_wigner_isr()`를 지나가므로, 가짜 데이터에서 정답을 복원하면 실제
데이터의 결과도 믿을 근거가 생긴다.

---

## 단계 0. 테스트 (10초)

```bash
python -m pytest
```

**무엇을 하나.** 코드가 바뀌어도 물리가 깨지지 않았는지 60가지를 확인한다.
예를 들어 ISR 복사 함수의 leading-log 부분이 정확히 1로 적분되는지, 그것이
전자 하나짜리 구조함수 두 개를 합성곱한 것과 같은지, 전방-후방 비대칭이 피크
아래에서 음수·위에서 양수인지, 노이즈 없는 데이터에서 피팅이 입력값을 그대로
돌려주는지, 손으로 옮긴 데이터가 OPAL 발표값과 2% 이내로 맞는지, 보정계수 표의
합계 행이 그 구성 행들과 맞는지.

**무엇이 나오나.** `60 passed in 15s` 정도. `pytest.ini`가 조용한 출력을 쓰므로
점만 한 줄 찍힌다. 실패가 있으면 그 테스트 이름이 무엇이 깨졌는지 알려 준다.

**왜 먼저 하나.** 이후 단계의 숫자를 믿기 전에 환경(패키지 버전 등)이
정상인지 확인하는 가장 빠른 방법이다.

---

## 단계 1. 실제 OPAL 데이터 (20초)

```bash
python src/compute_cross_section.py
```

**무엇을 하나.**
1. `data/raw/opal_zedometry_table1_event_counts.csv`에서 N과 L을, 보정계수
   표에서 f를 읽어 σ = fN/L을 계산한다. 보정계수가 있는 7개 점만 쓴다.
2. 계산값을 OPAL이 발표한 단면적(`*_xsec_published.csv`의 "measured" 열)과
   비교한다. 발표 단면적은 **검증에만** 쓴다. 같은 표의 평균 √s와 빔 에너지
   퍼짐(rms)은 빔의 성질이라 피팅 입력으로 쓴다.
3. Breit-Wigner를 피팅한다: 7개 점 전부, 이상한 1개를 뺀 6개, 그리고 6개에
   ISR(초기상태 복사) 보정을 넣어서. 실데이터 피팅은 모두 빔 에너지 퍼짐을
   모형에 넣는다.
4. 같은 피팅을 radiator 네 가지로 반복해 차수에 따른 변화를 본다.
5. 오차 예산을 만든다: 통계, 광도 반올림, 보정계수, radiator 절단, 채널별
   광도, LEP 에너지 보정이 각각 M_Z와 Γ_Z에 몇 MeV씩 기여하는지. 끝에
   광자 교환과 질량 정의까지 넣은 "완전한 모형" 결과가 나온다.
6. 4개 채널(qq, ee, μμ, ττ)을 교차 검사한다.

**화면에서 볼 것.**

```
=== mu+mu- cross section: computed vs published (OPAL 'measured' column) ===
 year   sample ... sqrt_s_GeV  sqrt_s_table1_GeV ... sigma_computed_nb ... xsec_measured_nb ... diff_pct
 1993   peak-2 ...    89.4505            89.4500 ...            0.4991 ...           0.4964 ...   0.5381
 1993     peak ...    91.2072            91.2100 ...            1.3490 ...           1.4563 ...  -7.3685   ← 이상치
 ...
```
`diff_pct`가 계산값과 발표값의 차이(%)다. 6개는 2% 이내, 1993 peak만 −7.4%.
`sqrt_s_GeV`는 채널별 평균 에너지(0.1 MeV 정밀도), `sqrt_s_table1_GeV`는 Table 1의
10 MeV 반올림 값이다.

```
=== Breit-Wigner fit, no ISR, excluding 1 flagged point(s) ===
 year sample  L_pbinv  diff_pct  L_mumu_implied_pbinv
 1993   peak   8.8000   -7.3685                8.1516
chi2/ndof = 0.37/3   (Migrad valid: True)
  mZ         = 91.4586 +/- 0.0105 GeV   (PDG: 91.1876 +/- 0.0021 GeV)
```
ISR 없는 피팅. χ²는 좋은데 PDG보다 271 MeV 크다. **좋은 χ²가 맞는 답을
보장하지 않는다**는 예. 위의 `L_mumu_implied_pbinv`는 1993 peak가 왜 어긋나는지
보여 준다: Table 1의 L(8.8)은 qq 분석의 광도이고, μμ 분석이 실제로 쓴 광도는
8.15 pb⁻¹이다(논문 8절). 측정이 틀린 게 아니라 우리가 넣은 L이 이 채널의 것이
아니다.

```
=== Same clean points, ISR-convolved Breit-Wigner (O(alpha) + leading-log O(alpha^2) radiator) ===
chi2/ndof = 1.63/3   (Migrad valid: True)
  mZ         = 91.1824 +/- 0.0120 GeV   (PDG: 91.1876 +/- 0.0021 GeV)
  gammaZ     = 2.5164 +/- 0.0194 GeV   (PDG: 2.4955 +/- 0.0023 GeV)
  shift vs. non-ISR fit: mZ -276.2 MeV, gammaZ -275.9 MeV
```
ISR을 넣자 M_Z가 276 MeV 내려와 PDG와 0.4σ 이내로 들어왔다.

```
=== Radiator order check: same fit with each radiator ===
  O(alpha) + leading-log O(alpha^2)    chi2  1.63   mZ 91.1824   gammaZ 2.5164   sigma_peak 2.0092
  O(alpha)                             chi2  1.59   mZ 91.1859   gammaZ 2.5210   sigma_peak 2.0022
  leading-log only, to O(alpha^2)      chi2  1.62   mZ 91.1824   gammaZ 2.5165   sigma_peak 2.0214
  simplified (pre-2026-11-05)          chi2  1.67   mZ 91.1789   gammaZ 2.5111   sigma_peak 2.1687
  o2 - o1: mZ -3.5 MeV, gammaZ -4.6 MeV
```
첫 줄이 기본값이다. 마지막 줄(차수를 하나 올렸을 때의 변화)이 오차 예산의
radiator 절단 항목이 된다. 셋째 줄과 첫 줄을 비교하면 비로그 O(α) 항이 M_Z, Γ_Z는
그대로 두고 σ_peak만 0.6% 낮춘다는 것이 보인다.

```
=== Error budget (6 clean points) ===
source                                                             kind          dM_Z (MeV)  dG_Z (MeV)
statistical (Poisson on N)                                         input              11.61       18.76
+ L rounding (0.1 pb^-1 granularity)                               input               2.66        4.15
radiator truncation (o2 - o1)                                      model              -3.50       -4.56
channel luminosity (L_mumu from OPAL's measured sigma vs L_qq)     data               -6.39      -12.66
LEP energy calibration (OPAL sec. 4; dM_Z/dE=1.00)                 external            1.80        1.30
propagator convention (running width - baseline)                   convention         -0.01        6.24
photon exchange + gamma/Z interference (improved Born - running BW) omission            2.96      -36.93
...
  M_Z     = 91.1824 +/- 0.0119 (fit ...) +/- 0.0075 (syst ...) GeV
  Gamma_Z = 2.5163 +/- 0.0195 (fit ...) +/- 0.0135 (syst ...) GeV
  complete Born model (+ photon exchange, interference):
    M_Z     = 91.1853 GeV, -0.16 sigma from PDG (fit + syst + PDG errors)
    Gamma_Z = 2.4856 GeV, -0.42 sigma from PDG (fit + syst + PDG errors)
```
`kind` 열을 보고 읽는다. `input`은 피팅을 통해 전파된 오차, `model`·`data`·
`external`은 계통오차로 합산되는 항목이다. `convention`·`omission`·`applied`는
오차가 **아니라** 보정이라 어느 합계에도 들어가지 않는다. 통계가 가장 크고,
그다음이 Table 1에 없는 채널별 광도다. 마지막 두 줄이 PDG와 비교할 숫자다:
Breit-Wigner가 빼먹은 광자 교환은 Γ_Z를 37 MeV나 움직인다.

```
=== Cross-channel consistency check (% diff from OPAL 'measured', by channel) ===
channel          qq    ee  mumu  tautau
1993 peak     -0.44 -0.46 -7.37   -3.89
  computed / measured = L_channel / L_qq, since N and f are the same on both sides.
```
1993 peak에서 트랙 기반 채널(μμ, ττ)만 어긋난다. N과 f가 양쪽에 똑같이 들어가므로
이 표는 사실 "채널별 광도 / qq 광도"의 표다.

**생기는 파일.** `data/processed/mumu_cross_section_computed.csv`,
`error_budget.csv`, `channel_consistency_check.csv`, `mumu_cross_section.png`.

---

## 단계 2. 토이 몬테카를로 (5초)

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
chi2/ndof = 100.06/10   (Migrad valid: True)
  pull vs. TRUE input: mZ +18.06 sigma, gammaZ +10.42 sigma
=== ISR-convolved Breit-Wigner fit to simulated points ===
chi2/ndof = 17.74/10   (Migrad valid: True)
  pull vs. TRUE input: mZ +0.72 sigma, gammaZ +0.33 sigma
```
`pull`은 (피팅값 − 정답)/피팅오차. ISR 없는 피팅은 18σ 틀리고 ISR 피팅은
1σ 이내. 13개 점에서는 ISR 누락이 χ²에서도 드러난다(100/10). ISR 피팅의
17.7/10은 이 시드에서 조금 높은 편(p = 0.06)이고, 3단계의 500회 평균은 10.05다.

```
=== A_FB(s) fit (mZ, gammaZ fixed from the ISR-convolved lineshape fit above) ===
chi2/ndof = 7.25/12
  sin2(theta_W_eff) = 0.22887 +0.00428 -0.00354 (MINOS), +/- 0.00384 (Hesse)   (true input: 0.23155)
  pull vs. TRUE input: -0.63 sigma (MINOS), -0.70 sigma (Hesse)
```
오차가 위아래로 다르다(+0.0043 / −0.0035). A_FB가 sin²θ_W에 v²를 통해서만
의존해 χ²가 포물선이 아니기 때문이며, MINOS가 그 비대칭을 그대로 따라간다.

**주의.** 난수 시드가 고정되어 있어 누가 돌려도 같은 숫자가 나온다. 단,
radiator가 바뀌면 같은 시드로도 다른 사건이 뽑히므로 이 숫자들은 코드 버전에
묶여 있다. 시드를 바꾸면 숫자는 달라지지만 결론(ISR 피팅만 정답 복원)은 같아야
한다. 그것을 확인하는 것이 3단계다.

**생기는 파일.** `simulated_mumu_scan.csv`, `simulated_mumu_closure.png`,
`simulated_mumu_afb.png`.

---

## 단계 3. 토이 500번 반복 (5분)

```bash
python src/pull_study.py                # 500회
python src/pull_study.py --n-toys 50    # 급하면 50회 (30초)
python src/pull_study.py --replot       # 저장된 토이로 표와 그림만 다시
```

**무엇을 하나.** 2단계를 시드만 바꿔 500번 돌리고 pull의 분포를 본다.
- 평균이 0이면 **편향이 없다**.
- 폭이 1이면 **피팅이 보고하는 오차가 실제 흩어짐과 같다**(오차가 정직하다).
- 폭이 1보다 작으면 오차를 과대평가, 크면 과소평가한 것이다.

**화면에서 볼 것.**

```
                      fit        parameter   n   mean  mean_err  width  width_err  frac_within_1sigma ...
                naive fit            $M_Z$ 500 17.586     0.042  0.937      0.030               0.000
        ISR-convolved fit            $M_Z$ 500 -0.034     0.043  0.969      0.031               0.690
        ISR-convolved fit       $\Gamma_Z$ 500 -0.117     0.043  0.964      0.031               0.680
$A_{FB}$ fit, Hesse error    sin2(theta_W) 500  0.002     0.039  0.881      0.028               0.746
$A_{FB}$ fit, MINOS error    sin2(theta_W) 500  0.124     0.042  0.941      0.030               0.678
  ISR-convolved fit, Gamma_Z: mean (fit - truth) = -3.8 +/- 1.7 MeV, mean fit error 39.3 MeV, mean chi2 10.05
```
ISR 피팅: 평균 ≈ 0, 폭 ≈ 1, 68~69%가 1σ 안 → 합격. Γ_Z는 3.8 MeV(오차의
1/10) 낮게 나오는데, 그 절반은 각 점의 오차를 그 점의 관측 사건 수로 잡는 χ²
피팅의 알려진 성질이다(`--expected-errors`로 확인할 수 있다).

A_FB 피팅은 대칭(Hesse) 오차로는 74.6%가 1σ 안에 들어온다. 오차 막대가 너무
길다는 뜻이다. MINOS 비대칭 오차로는 67.8%로 맞는다. 그래도 분포가 완전한
가우시안은 아니어서(+1σ 위 20%, −1σ 아래 12%) 평균이 +0.12로 움직인다.
**한 번만 돌려서는 절대 볼 수 없는 종류의 사실**이다.

**생기는 파일.** `pull_study_toys.csv`(500행), `pull_study_summary.csv`,
`pull_study.png`.

---

## 단계 4. 확장 (4분)

```bash
python src/extended_fits.py
```

**무엇을 하나.** 세 가지 "더 해 보면 어떻게 되나"를 실제 데이터로 시도한다.

**4.1 점 추가.** 보정계수 f가 없는 6개 점에 같은 연도의 f를 빌려 쓴다.
```
  6 own-f points (baseline)                        chi2/ndof   1.63/3   mZ 91.1824 +/- 0.0120   gammaZ 2.5164 +/- 0.0194
  12 points incl. borrowed f                       chi2/ndof   4.69/9   mZ 91.1818 +/- 0.0119   gammaZ 2.5193 +/- 0.0190
```
거의 안 변한다. 추가된 점이 전부 피크 위에 있어서 M_Z·Γ_Z 정보가 늘지 않기
때문이다. "점을 늘리면 좋아진다"가 항상 참은 아니라는 예.

**4.2 모형 비교.** 비상대론적 BW(기본), running-width BW(LEP 표준),
광자+Z+간섭(improved Born) 세 가지로 같은 6개 점을 피팅한다.
```
                             model       convention   chi2 ...  mZ_GeV ... gammaZ_GeV ...   norm ... mZ_running_GeV  gammaZ_running_GeV
    non-relativistic BW (baseline) non-relativistic 1.6287 ... 91.1824 ...     2.5162 ... 2.0093 ...            NaN                 NaN
     relativistic running-width BW          running 1.8258 ... 91.1824 ...     2.5225 ... 2.0084 ...        91.1824              2.5225
improved Born gamma+Z+interference            fixed 1.8235 ... 91.1515 ...     2.4846 ... 1.0028 ...        91.1853              2.4856
```
세 번째 줄의 `mZ_GeV`가 31 MeV 낮은 것은 질량 **정의**(fixed width)가 다르기
때문이다(README 0.2절). 오른쪽의 `mZ_running_GeV` 열이 같은 정의로 환산한 값이고,
PDG와 비교할 때는 이 열을 본다. `norm` 1.003은 "계산한 단면적이 표준모형
tree-level 절대 예측의 1.003배"라는 뜻으로, PDG 단면적을 넣지 않고도 절대
정규화가 1% 수준에서 맞는다는 독립 검증이다(표준모형 기대값은 1.013).

**4.3 결합 피팅.** qq, μμ, ττ를 공통 M_Z, Γ_Z로 한꺼번에 피팅한다. 채널마다
Breit-Wigner를 쓰는 피팅 세 가지 뒤에, 채널마다 광자+Z+간섭을 쓰는 피팅이
나온다. 마지막 것이 결과다.
```
-- improved Born: photon + Z + interference per channel, with beam spread --
chi2/ndof = 9.96/14   (Migrad valid: True)
  running-width convention: mZ = 91.1908 GeV, gammaZ = 2.5065 GeV
  sigma0_had = 41.35 +/- 0.05 nb   (PDG 41.481)
  R_l = Gamma_had / Gamma_ll = 20.80 +/- 0.06   (PDG R_l = 20.767)
  with systematics (radiator truncation, LEP energy calibration, lepton luminosity), running-width convention:
    mZ      = 91.1908 +/- 0.0035 (fit) +/- 0.0040 (syst) GeV, +0.56 sigma from PDG
    gammaZ  = 2.5065 +/- 0.0058 (fit) +/- 0.0050 (syst) GeV, +1.38 sigma from PDG
```
qq 채널 덕분에 피팅 오차가 3.5 MeV까지 줄어 계통오차(4.0 MeV)와 같은 크기가
됐다. 여기서부터는 데이터를 더 넣는 것보다 모형(더 높은 차수의 복사 보정,
에너지 보정의 상관)을 고치는 쪽이 이득이다. 위쪽 Breit-Wigner 피팅들의 Γ_Z가
PDG보다 2~3σ 높게 나오는 것은 대부분 광자 교환을 빼먹어서다.

**생기는 파일.** `mumu_cross_section_extended.csv`,
`lineshape_model_comparison.csv`, `joint_fit.csv`, 그림 2개.

---

## 단계 5, 6. MadGraph5 (선택, ISR 끔 1~2분, ISR 켬 약 10분)

```bash
export MG5_PATH=/path/to/MG5_aMC_v3_5_0/bin/mg5_aMC
python src/simulate_mumu_mg5.py          # ISR 끔
python src/simulate_mumu_mg5.py --isr    # ISR 켬 (MG5 3.2 이상, 약 10분)
python src/simulate_mumu_mg5.py --refit  # 사건 생성 없이, 저장된 scan을 지금 코드로 다시 피팅
```

**무엇을 하나.** 2단계의 토이는 "정답 모형"과 "피팅 모형"을 같은 사람이
짰다는 약점이 있다. 둘 다 같은 실수를 하면 검증이 통과해 버린다. 그래서
정답 모형을 독립적으로 구현된 MadGraph5의 행렬 요소로 바꾼다. 스크립트가
MG5를 13번 호출해 각 에너지에서 사건을 만들고, 나온 μ⁻의 각도를 읽어 2단계와
같은 처리를 한다.

**ISR 끔(5단계)에서 기대하는 것.** 생성기에 ISR이 없으므로 **ISR 없는 피팅이
맞고 ISR 피팅은 약 −8σ로 틀려야 한다**. 실제 데이터와 정반대. ISR 보정이
"항상 PDG 쪽으로 미는" 게 아니라 물리적으로 맞는 방향으로 움직인다는 증거.
이 역전은 MG5 없이도 테스트 하나(`test_isr_fit_of_radiation_free_data_is_biased_low`)가
확인한다.

**ISR 켬(6단계)에서 볼 것.** 이번엔 실제 데이터처럼 ISR 피팅이 정답을 복원하고
ISR 없는 피팅이 위로 치우쳐야 한다. 2026-11-11의 첫 완주(MG5 3.5.0, 10분 이내)
결과:

```
=== Naive Breit-Wigner fit to MG5-generated points ===
  pull vs. MG5 param_card truth: mZ +8.16 sigma, gammaZ +2.93 sigma
=== ISR-convolved Breit-Wigner fit to MG5-generated points (leading-log only, to O(alpha^2) radiator) ===
chi2/ndof = 25.83/10   (Migrad valid: True)
  pull vs. MG5 param_card truth: mZ +0.85 sigma, gammaZ -1.22 sigma
=== Noise-free shape check: the fits on MG5's integrated cross sections ===
  naive Breit-Wigner           mZ  +349.3 MeV, gammaZ  +469.5 MeV from the param_card truth
  ISR-convolved Breit-Wigner   mZ   +36.0 MeV, gammaZ   +27.4 MeV from the param_card truth
  improved Born                mZ    +4.6 MeV, gammaZ    -5.2 MeV from the param_card truth
```
ISR 피팅이 독립된 생성기의 정답을 1σ 근처로 복원했다. χ² 26은 모형 문제가 아니라
스크립트가 점마다 사건 수를 Poisson으로 뽑을 때의 흔들림이다(93.19 GeV 점 하나가
−3.4σ). "Noise-free" 블록은 MG5가 직접 적분한 단면적으로 같은 피팅을 한 것이라
통계 잡음이 없다. 여기서 Breit-Wigner는 36 MeV 어긋나고 광자+Z+간섭 모형은
5 MeV 이내로 맞는다. Breit-Wigner의 36 MeV는 질량 정의 차이(README 4.2절에서
실데이터로 잰 31 MeV)와 빠진 광자 항이다.

피팅은 `--isr`일 때 MG5의 전자 PDF와 같은 radiator, 같은 α(1/132.23)를 쓴다.
α(0)을 쓰면 M_Z가 9 MeV 높게 나온다.

**README 3절 숫자 갱신.** 3절의 표는 2026-09-27 실행 결과다. 그때 저장된
`data/processed/simulated_mumu_mg5_scan.csv`가 남아 있으면 `--refit`으로 지금의
radiator와 MINOS 오차를 적용한 숫자를 MG5 실행 없이 얻을 수 있다.

---

## 문제가 생기면

| 증상 | 원인과 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'compute_cross_section'` | 스크립트를 `python src/xxx.py`로 실행해야 한다. `python -m src.xxx`로 실행하면 import 경로가 달라진다. |
| `pytest`에서 import 오류 | `tests/conftest.py`가 `src/`를 경로에 넣는다. 저장소 최상위에서 `python -m pytest`로 실행. |
| 3단계가 너무 느리다 | `--n-toys 50`. 결론을 보는 데는 충분하고 통계적 정밀도만 떨어진다. |
| `make figures`가 전부 다시 돌린다 | `src/compute_cross_section.py`나 `src/electroweak.py`는 모든 스크립트가 쓰므로, 이 둘을 고치면 네 스크립트가 다 다시 돈다(약 10분). 의도된 동작이다. |
| `MadGraph5 executable not found` | 5~6단계만 해당. `MG5_PATH` 설정. 없으면 건너뛰어도 1~4단계 결과는 완전하다. |
| 숫자가 README와 소수점 넷째 자리에서 다르다 | 패키지 버전 차이(특히 scipy의 적분, iminuit). `requirements.txt` 주석의 버전으로 맞추면 같아진다. 결론이 달라지면 문제지만 마지막 자리는 아니다. |
| 그림이 안 열린다 / `DISPLAY` 오류 | 스크립트는 화면에 띄우지 않고 파일로만 저장한다. `data/processed/*.png`를 열면 된다. |

---

## 직접 바꿔 보기

코드를 이해했는지 확인하는 가장 좋은 방법은 한 줄을 바꾸고 결과가 **예상한
방향으로** 움직이는지 보는 것이다.

1. `src/compute_cross_section.py`에서 `ISR_RADIATOR = "o1"`로 바꾸고 1단계를
   다시 돌린다. M_Z가 3.5 MeV, Γ_Z가 4.6 MeV 올라가야 한다(오차 예산의 radiator
   truncation 줄과 같은 크기, 반대 부호).
2. `ANOMALY_THRESHOLD_PCT = 10.0`으로 바꾸면 1993 peak가 피팅에 들어간다.
   ISR 피팅의 χ²/ndof가 1.6/3에서 74.0/4로 나빠지고 Γ_Z가 29 MeV 커진다.
   넣은 L이 틀린 점 하나가 무엇을 하는지 본다.
3. `SPRIME_OVER_S_MIN = 0.7225`로 바꾼다(복사 꼬리를 더 많이 잘라내는 컷).
   오차 예산의 photon exchange 줄이 −36.9에서 −27.2 MeV로 줄어든다. 광자
   교환의 영향이 acceptance 정의에 달려 있다는 것, 그래서 이 값을 논문에서
   읽어야 한다는 것을 숫자로 본다.
4. `src/simulate_mumu.py`에서 `LUMI_PBINV = 20.0`(10배)으로 바꾸고 2단계를
   돌린다. 오차가 약 √10 ≈ 3.2배 줄어야 한다. 3단계의 pull 폭은 그대로 1
   근처여야 한다(오차가 줄어도 정직함은 유지).
5. `SCAN_POINTS_GEV`를 피크 ±1 GeV로 좁히면 Γ_Z 오차가 크게 나빠진다. 폭은
   날개에서 결정된다는 것을 숫자로 본다.

각 실험 뒤에는 `git checkout -- src/`로 되돌리고 `python -m pytest`로
원상복구를 확인한다.
