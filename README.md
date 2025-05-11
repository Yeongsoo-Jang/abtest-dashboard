# A/B 테스트 결과 통합 대시보드

CSV 파일을 업로드하면 A/B 테스트 결과를 자동으로 분석하고 시각화하는 대시보드입니다.

## 주요 기능

- CSV 파일 업로드 및 데이터 검증
- 그룹 열과 종속변수 열 선택 필터
- 정규성 검정(Q-Q 플롯, Shapiro-Wilk 검정)
- 등분산성 검정(Bartlett, Levene 검정)
- 그룹 수에 따른 적절한 통계 검정 수행
  - 2그룹: t-검정, Welch's t-검정, Mann-Whitney U 검정 등
  - 3그룹 이상: ANOVA, Kruskal-Wallis 검정, 사후 검정(Tukey HSD, Dunn's test) 등
- 효과 크기 측정 및 시각화(Cohen's d, Eta-squared)
- 그룹별 분포 및 평균 비교 시각화
- 부트스트랩 신뢰구간 분석
- 제1종/제2종 오류 및 검정력 분석
- 추가 분석(피어슨 상관계수, 카이제곱 검정, 오즈비 등)
- HTML 보고서 생성 및 이메일 전송

## 설치 방법

1. 저장소 클론 또는 다운로드
   ```bash
   git clone https://github.com/yourusername/ab-test-dashboard.git
   cd ab-test-dashboard
   ```

2. 가상환경 생성 및 활성화
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. 필요한 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

## 실행 방법

Streamlit 앱 실행:
```bash
streamlit run app.py
```

실행 후 웹 브라우저에서 `http://localhost:8501`로 접속하면 대시보드를 이용할 수 있습니다.

## 사용 방법

1. CSV 파일 업로드 (사이드바 이용)
2. 그룹 열과 종속변수 열 선택
3. 분석 옵션 설정 (유의수준, 추가 분석 등)
4. '분석 실행' 버튼 클릭
5. 탭을 이용해 다양한 분석 결과 확인
6. '결과 요약' 탭에서 보고서 생성 및 다운로드

## 파일 구조

```
ab_test_dashboard/
├── app.py                   # 메인 Streamlit 앱
├── utils/
│   ├── __init__.py
│   ├── data_processor.py    # 데이터 로딩 및 전처리
│   ├── statistical_tester.py # 통계 검정 관련 기능
│   ├── visualizer.py        # 데이터 시각화
│   └── reporter.py          # 보고서 생성 및 이메일 전송
├── assets/
│   └── style.css            # 스타일시트
└── requirements.txt         # 필요한 패키지 목록
```

## 데이터 형식

분석을 위한 CSV 파일은 다음 형식을 따라야 합니다:

1. **필수 열**:
   - 그룹 열: 실험 그룹을 나타내는 열 (예: 'group' 열에 'A', 'B', 'C' 등의 값)
   - 종속변수 열: 측정하려는 지표 (수치형 데이터여야 함)

2. **예시 데이터**:

| group | conversion_rate | time_spent | revenue |
|-------|-----------------|------------|---------|
| A     | 0.12            | 120        | 45.5    |
| A     | 0.08            | 105        | 38.2    |
| B     | 0.15            | 95         | 52.1    |
| B     | 0.11            | 85         | 49.3    |
| ...   | ...             | ...        | ...     |

## 주의사항

- 그룹당 최소 30개 이상의 샘플이 권장됩니다.
- 결측치가 없는 데이터를 사용하는 것이 좋습니다.
- 이메일 전송 기능을 사용하려면 SMTP 서버 정보가 필요합니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.
