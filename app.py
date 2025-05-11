import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import os
import plotly.io as pio
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 민감한 정보 가져오기
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
DEFAULT_RECIPIENT = os.getenv("DEFAULT_RECIPIENT", "")


# 커스텀 모듈 임포트
from utils.data_processor import DataProcessor
from utils.statistical_tester import StatisticalTester
from utils.visualizer import Visualizer
from utils.reporter import Reporter

# 페이지 설정
st.set_page_config(
    page_title="A/B 테스트 통합 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        background-color: #f9f9f9;
        border-left: 5px solid #1E88E5;
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #f0f8ff;
        border: 1px solid #e0e0e0;
    }
    .insight-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #e3f2fd;
        border-left: 5px solid #2196F3;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #fff3e0;
        border-left: 5px solid #FF9800;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #e8f5e9;
        border-left: 5px solid #4CAF50;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .metric-card {
        padding: 1rem;
        text-align: center;
        border-radius: 0.5rem;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .info-text {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# 상태 초기화
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = None
if 'statistical_tester' not in st.session_state:
    st.session_state.statistical_tester = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None
if 'reporter' not in st.session_state:
    st.session_state.reporter = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'columns_set' not in st.session_state:
    st.session_state.columns_set = False
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

# 유틸리티 함수
def display_success(message):
    """성공 메시지 표시"""
    st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)

def display_warning(message):
    """경고 메시지 표시"""
    st.markdown(f'<div class="warning-box">{message}</div>', unsafe_allow_html=True)

def display_error(message):
    """오류 메시지 표시"""
    st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)

def display_insight(message):
    """인사이트 메시지 표시"""
    st.markdown(f'<div class="insight-box">{message}</div>', unsafe_allow_html=True)

def reset_state():
    """앱 상태 초기화"""
    st.session_state.data_processor = None
    st.session_state.statistical_tester = None
    st.session_state.visualizer = None
    st.session_state.reporter = None
    st.session_state.data_loaded = False
    st.session_state.columns_set = False
    st.session_state.analysis_run = False
    st.experimental_rerun()

#----- 메인 애플리케이션 -----#

# 헤더 및 소개
st.markdown('<div class="main-header">A/B 테스트 결과 통합 대시보드</div>', unsafe_allow_html=True)

st.markdown("""
이 대시보드는 A/B 테스트 데이터를 분석하여 통계적 인사이트를 제공합니다. 
CSV 파일을 업로드하고 그룹 열과 종속변수 열을 선택하면, 자동으로 정규성 검정, 
등분산성 검정, 가설 검정 등을 수행하고 결과를 시각화합니다.
""")

# 사이드바
with st.sidebar:
    st.header("설정")
    
    # 파일 업로드 섹션
    st.subheader("1. 데이터 업로드")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    
    if uploaded_file is not None:
        # 데이터 로드
        try:
            # DataProcessor 인스턴스 생성 및 데이터 로드
            if not st.session_state.data_loaded:
                st.session_state.data_processor = DataProcessor()
                data = st.session_state.data_processor.load_data(uploaded_file)
                valid, message = st.session_state.data_processor.validate_data()
                
                if valid:
                    st.session_state.data_loaded = True
                    st.success("데이터가 성공적으로 로드되었습니다.")
                else:
                    st.error(message)
            
            # 열 선택
            if st.session_state.data_loaded:
                st.subheader("2. 열 선택")
                
                # 그룹 열 선택
                all_columns = st.session_state.data_processor.data.columns.tolist()
                categorical_cols = st.session_state.data_processor.get_column_types()["categorical"]
                numeric_cols = st.session_state.data_processor.get_column_types()["numeric"]
                
                # 범주형 열만 있으면 그룹 열로 추천
                group_col_suggestion = categorical_cols[0] if categorical_cols else all_columns[0]
                group_col = st.selectbox(
                    "그룹 열 선택 (실험 그룹을 구분하는 열)",
                    all_columns,
                    index=all_columns.index(group_col_suggestion) if group_col_suggestion in all_columns else 0
                )
                
                # 종속변수 열 선택 (수치형 열만 표시)
                target_col_suggestion = numeric_cols[0] if numeric_cols else all_columns[0]
                target_col = st.selectbox(
                    "종속변수 열 선택 (측정하려는 지표)",
                    all_columns,
                    index=all_columns.index(target_col_suggestion) if target_col_suggestion in all_columns else 0
                )
                
                # 열 설정 적용
                if st.button("열 설정 적용"):
                    try:
                        st.session_state.data_processor.set_group_and_target(group_col, target_col)
                        
                        # StatisticalTester 및 Visualizer 인스턴스 생성
                        st.session_state.statistical_tester = StatisticalTester(st.session_state.data_processor)
                        st.session_state.visualizer = Visualizer(st.session_state.data_processor, st.session_state.statistical_tester)
                        st.session_state.reporter = Reporter(st.session_state.data_processor, st.session_state.statistical_tester, st.session_state.visualizer)
                        
                        st.session_state.columns_set = True
                        st.success(f"그룹 열: '{group_col}', 종속변수 열: '{target_col}'이(가) 설정되었습니다.")
                    except Exception as e:
                        st.error(f"열 설정 중 오류 발생: {str(e)}")
                
                # 열이 설정되었다면 분석 옵션 표시
                if st.session_state.columns_set:
                    st.subheader("3. 분석 옵션")
                    
                    # 유의수준 설정
                    alpha = st.slider("유의수준 (α)", min_value=0.01, max_value=0.1, value=0.05, step=0.01)
                    st.session_state.statistical_tester.set_alpha(alpha)
                    
                    # 부트스트랩 리샘플링 수
                    bootstrap_samples = st.slider("부트스트랩 리샘플링 수", min_value=100, max_value=5000, value=1000, step=100)
                    
                    # 추가 분석 옵션
                    with st.expander("추가 분석 옵션"):
                        chi_square = st.checkbox("카이제곱 검정 수행 (이진화 분석)", value=False)
                        include_pearson = st.checkbox("피어슨 상관계수 계산 (두 그룹인 경우)", value=False)
                    
                    # 분석 실행 버튼
                    if st.button("분석 실행"):
                        with st.spinner("분석 중..."):
                            # 모든 테스트 실행
                            st.session_state.statistical_tester.run_all_tests()
                            
                            # 추가 옵션에 따른 분석
                            if chi_square:
                                try:
                                    st.session_state.chi_square_results = st.session_state.statistical_tester.chi_square_test()
                                except Exception as e:
                                    st.warning(f"카이제곱 검정 중 오류 발생: {str(e)}")
                                    st.session_state.chi_square_results = None
                            
                            if include_pearson and len(st.session_state.data_processor.groups) == 2:
                                try:
                                    st.session_state.pearson_results = st.session_state.statistical_tester.pearson_correlation()
                                except Exception as e:
                                    st.warning(f"피어슨 상관계수 계산 중 오류 발생: {str(e)}")
                                    st.session_state.pearson_results = None
                            
                            st.session_state.analysis_run = True
                            time.sleep(0.5)  # 약간의 지연으로 스피너 표시
                        
                        st.success("분석이 완료되었습니다! 결과를 확인하세요.")
                    
                    # 상태 초기화 버튼
                    if st.button("처음부터 다시 시작"):
                        reset_state()
        
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {str(e)}")
    
    else:
        # 가이드 표시
        if not st.session_state.data_loaded:
            st.markdown("""
            ### 데이터 업로드 가이드
            - CSV 형식의 파일을 준비해주세요.
            - 파일에는 그룹 열(예: 'group')과 측정하려는 종속변수 열이 포함되어야 합니다.
            - 그룹 열은 실험 그룹(예: 'A', 'B', 'C')을 구분합니다.
            - 종속변수 열은 수치형 데이터여야 합니다.
            """)

# 메인 콘텐츠
if st.session_state.data_loaded:
    # 데이터 탭과 결과 탭
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["결과 요약", "데이터 개요", "기본 가정 검정", "가설 검정", "심화 분석"])
    
    # 결과 요약 탭
    with tab1:
        if st.session_state.analysis_run:
            st.markdown('<div class="sub-header">A/B 테스트 결과 요약</div>', unsafe_allow_html=True)
            
            # 필요한 데이터 가져오기
            hypothesis_test = st.session_state.statistical_tester.hypothesis_test_results
            effect_size = st.session_state.statistical_tester.effect_size_results
            hypothesis = st.session_state.statistical_tester.get_null_alternative_hypothesis()
            
            # 주요 정보 추출
            test_name = hypothesis_test.get("test_name", "알 수 없음")
            p_value = hypothesis_test.get("p_value", 0)
            significant = hypothesis_test.get("significant", False)
            effect_measure = effect_size.get("measure", "알 수 없음")
            effect_value = effect_size.get("value", 0)
            effect_interpretation = effect_size.get("interpretation", "알 수 없음")
            alpha = st.session_state.statistical_tester.alpha
            
            # 결과에 따른 색상 및 아이콘 설정
            result_color = "#4CAF50" if significant else "#9E9E9E"
            if significant and abs(effect_value) < 0.2:
                result_color = "#FFC107"  # 유의하지만 효과 크기가 작은 경우
            
            # 3개 열로 주요 지표 표시
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # p-value 메트릭
                st.metric(
                    label="p-value",
                    value=f"{p_value:.4f}",
                    delta=f"유의수준(α): {alpha}",
                    delta_color="inverse" if p_value < alpha else "off"
                )
                
            with col2:
                # 효과 크기 메트릭
                st.metric(
                    label=f"효과 크기 ({effect_measure})",
                    value=f"{effect_value:.3f}",
                    delta=effect_interpretation
                )
                
            with col3:
                # 검정 방법
                st.metric(
                    label="검정 방법",
                    value=test_name
                )
            
            # 구분선
            st.markdown("---")
            
            # 통계적 유의성 결과 표시
            if significant:
                st.success(f"""
                ### 통계적으로 유의미한 차이가 있습니다.
                p-value({p_value:.4f})가 유의수준({alpha})보다 작으므로, 귀무가설을 기각합니다.
                측정된 효과 크기({effect_value:.3f})는 '{effect_interpretation}' 수준입니다.
                """)
            else:
                st.info(f"""
                ### 통계적으로 유의미한 차이가 없습니다.
                p-value({p_value:.4f})가 유의수준({alpha})보다 크므로, 귀무가설을 채택합니다.
                측정된 효과 크기({effect_value:.3f})는 '{effect_interpretation}' 수준이지만, 통계적으로 유의하지 않습니다.
                """)
            
            # 귀무가설과 대립가설 표시
            st.markdown("### 가설 정보")
            
            hypothesis_col1, hypothesis_col2 = st.columns(2)
            with hypothesis_col1:
                st.info(f"**귀무가설 (H₀)**: {hypothesis['null']}")
            
            with hypothesis_col2:
                st.info(f"**대립가설 (H₁)**: {hypothesis['alternative']}")
            
            # 효과 크기 시각화 (작은 게이지 차트)
            st.markdown("### 효과 크기 시각화")
            
            # 효과 크기 게이지 차트는 시각적 정보로서 가치가 있으므로 유지
            try:
                effect_fig = st.session_state.visualizer.plot_effect_size()
                st.plotly_chart(effect_fig, use_container_width=True, key="summary_effect_size_chart")
            except Exception as e:
                st.error(f"효과 크기 시각화 중 오류 발생: {str(e)}")
            
            # 추가 분석 결과
            st.markdown("### 추가 분석 정보")
            
            if len(st.session_state.data_processor.groups) == 2:
                bootstrap = st.session_state.statistical_tester.bootstrap_results
                if "difference" in bootstrap:
                    diff_result = bootstrap["difference"]
                    ci_low = diff_result["ci_lower"]
                    ci_up = diff_result["ci_upper"]
                    
                    st.info(f"""
                    **부트스트랩 95% 신뢰구간**: [{ci_low:.3f}, {ci_up:.3f}]
                    
                    신뢰구간이 0을 포함{'하지 않으므로' if diff_result['significant'] else '하므로'} 
                    부트스트랩 방법으로도 결과가 {'유의합니다.' if diff_result['significant'] else '유의하지 않습니다.'}
                    """)
            
            # 검정력 정보
            error_analysis = st.session_state.statistical_tester.error_analysis
            power = error_analysis['power']
            
            power_status = "높음" if power > 0.8 else "중간" if power > 0.5 else "낮음"
            power_color = "green" if power > 0.8 else "orange" if power > 0.5 else "red"
            
            st.markdown(f"""
            **검정력(1-β)**: <span style='color:{power_color};'>{power:.3f} ({power_status})</span>
            """, unsafe_allow_html=True)
            
            if power < 0.8:
                st.warning(f"""
                **참고**: 현재 검정력({power:.2f})이 권장 수준(0.8) 미만입니다. 
                샘플 크기를 늘리거나 효과 크기가 더 클 경우 검정력이 향상될 수 있습니다.
                """)
            
            # 보고서 생성 및 다운로드 버튼은 기존과 동일하게 유지
            st.markdown("---")
            st.markdown('<div class="sub-header">보고서 생성</div>', unsafe_allow_html=True)

            if st.button("보고서 생성"):
                with st.spinner("HTML 보고서 생성 중..."):
                    try:
                        report_html = st.session_state.reporter.generate_report()
                        download_link = st.session_state.reporter.download_report()
                        st.markdown(download_link, unsafe_allow_html=True)
                        st.success("HTML 보고서가 성공적으로 생성되었습니다. 위 링크를 클릭하여 다운로드하세요.")
                    except Exception as e:
                        st.error(f"보고서 생성 중 오류 발생: {str(e)}")
            
            # 이메일 전송 섹션
            with st.expander("이메일로 보고서 전송"):
                st.markdown("보고서를 이메일로 전송하려면 아래 정보를 입력하세요.")
                
                recipient_email = st.text_input("수신자 이메일", value=DEFAULT_RECIPIENT) # 기본 수신자 이메일 추가
                subject = st.text_input("이메일 제목", "A/B 테스트 결과 보고서")
                
                col1, col2 = st.columns(2)
                with col1:
                    smtp_server = st.text_input("SMTP 서버", SMTP_SERVER)
                with col2:
                    smtp_port = st.number_input("SMTP 포트", value=SMTP_PORT)
                
                col3, col4 = st.columns(2)
                with col3:
                    sender_email = st.text_input("발신자 이메일", EMAIL_SENDER)
                with col4:
                    # 보안을 위해 기본값은 표시하지 않음
                    sender_password_display = "●●●●●●●●" if EMAIL_PASSWORD else ""
                    sender_password_input = st.text_input("발신자 비밀번호", 
                                                value=sender_password_display, 
                                                type="password")
                    
                    # 사용자가 "●●●●●●●●"를 그대로 두면 환경 변수 값 사용, 아니면 입력값 사용
                    if sender_password_input == "●●●●●●●●" and EMAIL_PASSWORD:
                        sender_password = EMAIL_PASSWORD
                    else:
                        sender_password = sender_password_input
                
                # 추가 메시지 입력 옵션 (선택사항)
                additional_message = st.text_area("추가 메시지 (선택사항)", 
                                                "안녕하세요,\n\n첨부된 A/B 테스트 결과 보고서를 확인해주세요.\n\n감사합니다.", 
                                                height=100)
                
                # 전송 버튼
                if st.button("이메일 전송"):
                    if not recipient_email:
                        st.error("수신자 이메일을 입력해주세요.")
                    else:
                        with st.spinner("이메일 전송 중..."):
                            try:
                                success = st.session_state.reporter.send_email(
                                    recipient_email=recipient_email,
                                    subject=subject,
                                    message=additional_message,
                                    smtp_server=smtp_server,
                                    smtp_port=smtp_port,
                                    sender_email=sender_email,
                                    sender_password=sender_password
                                )
                                
                                if success:
                                    st.success("HTML 보고서가 이메일로 성공적으로 전송되었습니다.")
                                else:
                                    st.error("이메일 전송에 실패했습니다.")
                            except Exception as e:
                                st.error(f"이메일 전송 중 오류 발생: {str(e)}")
    
    # 데이터 개요 탭
    with tab2:
        st.markdown('<div class="sub-header">데이터 미리보기</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.data_processor.data.head(10))
        
        st.markdown('<div class="sub-header">데이터 정보</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 행 수", len(st.session_state.data_processor.data))
        with col2:
            st.metric("총 열 수", len(st.session_state.data_processor.data.columns))
        with col3:
            if st.session_state.columns_set:
                st.metric("그룹 수", len(st.session_state.data_processor.groups))
        
        if st.session_state.columns_set:
            st.markdown('<div class="sub-header">그룹별 기초 통계</div>', unsafe_allow_html=True)
            st.dataframe(st.session_state.data_processor.get_group_summary())
            
            # 그룹별 샘플 수 확인 및 경고
            group_counts = st.session_state.data_processor.get_group_summary()['개수'].to_dict()
            small_groups = {group: count for group, count in group_counts.items() if count < 30}
            
            if small_groups:
                warning_msg = "다음 그룹은 샘플 수가 30개 미만으로, 통계적 검정력이 낮을 수 있습니다: "
                warning_msg += ", ".join([f"'{group}' ({count}개)" for group, count in small_groups.items()])
                display_warning(warning_msg)
            
            # 그룹별 분포 시각화
            if st.session_state.visualizer is not None:
                st.markdown('<div class="sub-header">그룹별 분포 비교</div>', unsafe_allow_html=True)
                
                # 그래프 유형 선택 드롭다운 추가
                plot_type = st.selectbox(
                    "분포 시각화 유형 선택",
                    ["Violin Plot", "Histogram", "Ridgeline Plot", "Box Plot"],
                    index=0
                )
                
                try:
                    # 선택된 그래프 유형에 따라 다른 시각화 함수 호출
                    if plot_type == "Violin Plot":
                        dist_fig = st.session_state.visualizer.plot_distribution_comparison()
                    elif plot_type == "Histogram":
                        dist_fig = st.session_state.visualizer.plot_distribution_comparison_histogram()
                    elif plot_type == "Ridgeline Plot":
                        dist_fig = st.session_state.visualizer.plot_distribution_comparison_ridgeline()
                    elif plot_type == "Box Plot":
                        dist_fig = st.session_state.visualizer.plot_distribution_comparison_boxplot()
                    
                    st.plotly_chart(dist_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"분포 시각화 중 오류 발생: {str(e)}")
    
    # 기본 가정 검정 탭
    with tab3:
        if st.session_state.columns_set:
            st.markdown('<div class="sub-header">정규성 검정</div>', unsafe_allow_html=True)
            st.markdown("""
            정규성 검정은 데이터가 정규 분포를 따르는지 확인합니다. 
            p-value가 유의수준(α)보다 크면 정규성을 만족합니다.
            """)
            
            # 정규성 검정 실행
            if not st.session_state.analysis_run:
                normality_results = st.session_state.statistical_tester.test_normality()
            else:
                normality_results = st.session_state.statistical_tester.normality_results
            
            # 정규성 결과 표시
            normality_df = pd.DataFrame({
                '그룹': [],
                'Shapiro-Wilk 통계량': [],
                'p-value': [],
                '정규성': []
            })

            for group, result in normality_results.items():
                if result["shapiro"]["statistic"] is not None:
                    normality_df = pd.concat([
                        normality_df,
                        pd.DataFrame({
                            '그룹': [group],
                            'Shapiro-Wilk 통계량': [f"{result['shapiro']['statistic']:.4f}"],  # 소수점 4자리 제한
                            'p-value': [f"{result['shapiro']['p_value']:.4f}"],  # 소수점 4자리 제한
                            '정규성': ["만족" if result["shapiro"]["normal"] else "불만족"]
                        })
                    ])

            st.dataframe(normality_df)
            
            # Q-Q 플롯
            st.markdown('<div class="sub-header">Q-Q 플롯 (정규성 검정)</div>', unsafe_allow_html=True)
            st.markdown("""
            Q-Q 플롯은 데이터의 분포가 정규 분포와 얼마나 일치하는지 시각적으로 보여줍니다.
            점들이 직선에 가까울수록 정규성을 만족합니다.
            """)
            
            try:
                qq_fig = st.session_state.visualizer.plot_qq_plots()
                st.plotly_chart(qq_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Q-Q 플롯 생성 중 오류 발생: {str(e)}")
            
            # 등분산성 검정
            st.markdown('<div class="sub-header">등분산성 검정</div>', unsafe_allow_html=True)
            st.markdown("""
            등분산성 검정은 그룹 간 분산이 동일한지 확인합니다.
            p-value가 유의수준(α)보다 크면 등분산성을 만족합니다.
            """)
            
            # 등분산성 검정 실행
            if not st.session_state.analysis_run:
                homogeneity_results = st.session_state.statistical_tester.test_homogeneity()
            else:
                homogeneity_results = st.session_state.statistical_tester.homogeneity_results
            
            # 등분산성 결과 표시
            homogeneity_df = pd.DataFrame({
                '검정': ['Bartlett', 'Levene'],
                '통계량': [f"{homogeneity_results['bartlett']['statistic']:.4f}", 
                        f"{homogeneity_results['levene']['statistic']:.4f}"],  # 소수점 4자리 제한
                'p-value': [f"{homogeneity_results['bartlett']['p_value']:.4f}", 
                            f"{homogeneity_results['levene']['p_value']:.4f}"],  # 소수점 4자리 제한
                '등분산성': ["만족" if homogeneity_results["bartlett"]["equal_variances"] else "불만족",
                        "만족" if homogeneity_results["levene"]["equal_variances"] else "불만족"]
            })

            st.dataframe(homogeneity_df)
            
            # 정규성 및 등분산성 요약
            all_normal = all(result["shapiro"]["normal"] for result in normality_results.values() if result["shapiro"]["normal"] is not None)
            levene_equal_var = homogeneity_results["levene"]["equal_variances"]
            
            if all_normal:
                if levene_equal_var:
                    display_success("모든 그룹이 정규성과 등분산성을 만족합니다. 모수적 검정(t-검정, ANOVA 등)을 사용할 수 있습니다.")
                else:
                    display_warning("모든 그룹이 정규성을 만족하지만 등분산성을 만족하지 않습니다. Welch's t-검정이나 Welch's ANOVA를 사용하는 것이 좋습니다.")
            else:
                display_warning("일부 또는 모든 그룹이 정규성을 만족하지 않습니다. 비모수적 검정(Mann-Whitney U, Kruskal-Wallis 등)을 사용하는 것이 좋습니다.")
    
    # 가설 검정 탭
    with tab4:
        if st.session_state.analysis_run:
            # 가설 설정
            st.markdown('<div class="sub-header">가설 설정</div>', unsafe_allow_html=True)
            hypothesis = st.session_state.statistical_tester.get_null_alternative_hypothesis()
            
            st.markdown(f"""
            **귀무가설 (H₀)**: {hypothesis['null']}  
            **대립가설 (H₁)**: {hypothesis['alternative']}
            """)
            
            # 그룹별 평균 비교
            st.markdown('<div class="sub-header">그룹별 평균 비교</div>', unsafe_allow_html=True)
            try:
                mean_fig = st.session_state.visualizer.plot_mean_comparison()
                st.plotly_chart(mean_fig, use_container_width=True)
            except Exception as e:
                st.error(f"평균 비교 시각화 중 오류 발생: {str(e)}")
            
            # 가설 검정 결과
            st.markdown('<div class="sub-header">가설 검정 결과</div>', unsafe_allow_html=True)
            test_results = st.session_state.statistical_tester.hypothesis_test_results
            
            st.markdown(f"""
            **검정 방법**: {test_results['test_name']}  
            **p-value**: {test_results['p_value']:.4f}  
            **유의수준 (α)**: {st.session_state.statistical_tester.alpha:.2f}  
            **결과**: {"귀무가설 기각 (통계적으로 유의함)" if test_results['significant'] else "귀무가설 채택 (통계적으로 유의하지 않음)"}
            """)
            
            # 사후 검정 결과 (3개 이상 그룹일 경우)
            if 'post_hoc' in test_results:
                st.markdown('<div class="sub-header">사후 검정 결과</div>', unsafe_allow_html=True)
                st.markdown(f"**방법**: {test_results['post_hoc']['method']}")
                
                post_hoc_df = pd.DataFrame(test_results['post_hoc']['results'])
                
                # 컬럼 이름 재정의
                if 'meandiff' in post_hoc_df.columns:
                    post_hoc_df = post_hoc_df.rename(columns={
                        'group1': '그룹1',
                        'group2': '그룹2',
                        'meandiff': '평균 차이',
                        'p-adj': 'p-value',
                        'lower': '신뢰구간 하한',
                        'upper': '신뢰구간 상한',
                        'reject': '유의성'
                    })
                    post_hoc_df['유의성'] = post_hoc_df['유의성'].map({True: '유의함', False: '유의하지 않음'})
                elif 'p_value' in post_hoc_df.columns:
                    post_hoc_df = post_hoc_df.rename(columns={
                        'group1': '그룹1',
                        'group2': '그룹2',
                        'p_value': 'p-value',
                        'significant': '유의성'
                    })
                    post_hoc_df['유의성'] = post_hoc_df['유의성'].map({True: '유의함', False: '유의하지 않음'})
                
                st.dataframe(post_hoc_df)
            
            # 효과 크기
            st.markdown('<div class="sub-header">효과 크기 분석</div>', unsafe_allow_html=True)
            effect_size = st.session_state.statistical_tester.effect_size_results
            
            st.markdown(f"""
            **효과 크기 측정**: {effect_size['measure']}  
            **값**: {effect_size['value']:.4f}  
            **해석**: {effect_size['interpretation']}  
            **비교**: {effect_size['comparison']}
            """)
            
            try:
                effect_fig = st.session_state.visualizer.plot_effect_size()
                st.plotly_chart(effect_fig, use_container_width=True)
            except Exception as e:
                st.error(f"효과 크기 시각화 중 오류 발생: {str(e)}")
            
            # 결과 해석
            st.markdown('<div class="sub-header">결과 해석</div>', unsafe_allow_html=True)
            
            if test_results['significant']:
                if abs(effect_size['value']) < 0.2 and effect_size['measure'] == "Cohen's d":
                    display_insight("""
                    **통계적으로 유의하지만, 효과 크기가 작습니다.**  
                    - 통계적 유의성이 항상 실질적인 중요성을 의미하는 것은 아닙니다.
                    - 대규모 샘플에서는 작은 차이도 통계적으로 유의할 수 있습니다.
                    - 비즈니스 맥락과 비용-편익 분석을 고려하여 결과를 해석해야 합니다.
                    """)
                else:
                    display_success("""
                    **통계적으로 유의하며, 효과 크기도 충분합니다.**  
                    - 실험 처치가 효과적이라는 강력한 증거입니다.
                    - 결과를 확신할 수 있으며, 실제 환경에 적용할 수 있습니다.
                    """)
            else:
                # 검정력 정보 활용
                error_analysis = st.session_state.statistical_tester.error_analysis
                power = error_analysis['power']
                
                if power < 0.8:
                    display_warning(f"""
                    **통계적으로 유의하지 않으며, 검정력({power:.2f})이 낮습니다.**  
                    - 제2종 오류(실제 효과가 있는데 감지하지 못함)의 가능성이 있습니다.
                    - 더 큰 샘플 크기로 실험을 반복하는 것이 좋습니다.
                    - 최소 검정력 0.8을 달성하려면 더 많은 샘플이 필요합니다.
                    """)
                else:
                    display_insight("""
                    **통계적으로 유의하지 않지만, 검정력은 충분합니다.**  
                    - 실험 처치에 실질적인 효과가 없다고 판단할 수 있습니다.
                    - 다른 대안을 고려하거나 실험 설계를 재검토하는 것이 좋습니다.
                    """)
        else:
            display_warning("분석을 실행해주세요. 사이드바에서 '분석 실행' 버튼을 클릭하세요.")
    
    # 심화 분석 탭
    with tab5:
        if st.session_state.analysis_run:
            # 부트스트랩 분석
            st.markdown('<div class="sub-header">부트스트랩 신뢰구간 분석</div>', unsafe_allow_html=True)
            st.markdown("""
            부트스트랩은 반복적인 리샘플링을 통해 신뢰구간을 추정하는 방법으로, 
            정규성 가정에 의존하지 않고 강건한 결과를 제공합니다.
            """)
            
            bootstrap_results = st.session_state.statistical_tester.bootstrap_results
            
            try:
                bootstrap_fig = st.session_state.visualizer.plot_bootstrap_ci()
                st.plotly_chart(bootstrap_fig, use_container_width=True)
            except Exception as e:
                st.error(f"부트스트랩 시각화 중 오류 발생: {str(e)}")
            
            # 제1종, 제2종 오류 분석
            st.markdown('<div class="sub-header">통계적 오류 및 검정력 분석</div>', unsafe_allow_html=True)
            st.markdown("""
            - **제1종 오류(α)**: 귀무가설이 참일 때 이를 기각할 확률 (거짓양성)
            - **제2종 오류(β)**: 귀무가설이 거짓일 때 이를 채택할 확률 (거짓음성)
            - **검정력(1-β)**: 대립가설이 참일 때 이를 올바르게 채택할 확률
            """)
            
            try:
                error_fig = st.session_state.visualizer.plot_error_matrix()
                st.plotly_chart(error_fig, use_container_width=True)
            except Exception as e:
                st.error(f"오류 매트릭스 시각화 중 오류 발생: {str(e)}")
            
            # 추가 분석 (피어슨 상관계수, 카이제곱)
            if hasattr(st.session_state, 'pearson_results') and st.session_state.pearson_results is not None:
                st.markdown('<div class="sub-header">피어슨 상관 분석</div>', unsafe_allow_html=True)
                pearson = st.session_state.pearson_results
                
                st.markdown(f"""
                **피어슨 상관계수(r)**: {pearson['pearson_r']:.4f}  
                **p-value**: {pearson['p_value']:.4f}  
                **유의성**: {"유의함" if pearson['significant'] else "유의하지 않음"}  
                **해석**: {pearson['interpretation']}
                """)
            
            if hasattr(st.session_state, 'chi_square_results') and st.session_state.chi_square_results is not None:
                st.markdown('<div class="sub-header">카이제곱 검정 (이진화 분석)</div>', unsafe_allow_html=True)
                chi2 = st.session_state.chi_square_results
                
                st.markdown(f"""
                **카이제곱 통계량**: {chi2['chi2']:.4f}  
                **p-value**: {chi2['p_value']:.4f}  
                **자유도**: {chi2['dof']}  
                **유의성**: {"유의함" if chi2['significant'] else "유의하지 않음"}
                """)
                
                if chi2['odds_ratio'] is not None:
                    st.markdown(f"**오즈비**: {chi2['odds_ratio']:.4f}")
    
        else:
            display_warning("분석을 실행해주세요. 사이드바에서 '분석 실행' 버튼을 클릭하세요.")
else:
    # 데이터가 로드되지 않은 경우 가이드 표시
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">A/B 테스트 데이터 업로드 가이드</div>', unsafe_allow_html=True)
    
    st.markdown("""
    올바른 A/B 테스트 분석을 위해 다음 형식의 CSV 파일을 준비해주세요:
    
    1. **필수 열**:
       - 그룹 열: 실험 그룹을 나타내는 열 (예: 'group' 열에 'A', 'B', 'C' 등의 값)
       - 종속변수 열: 측정하려는 지표 (수치형 데이터여야 함)
       
    2. **권장 사항**:
       - 모든 열에 적절한 헤더(열 이름)가 있어야 합니다.
       - 결측치가 없어야 합니다.
       - 충분한 샘플 수를 확보해야 합니다(그룹당 최소 30개 이상 권장).
       
    3. **예시 데이터**:
    """)
    
    # 예시 데이터 표시
    example_data = pd.DataFrame({
        'group': ['A', 'A', 'A', 'B', 'B', 'B'],
        'conversion_rate': [0.12, 0.08, 0.15, 0.18, 0.21, 0.19],
        'time_spent': [120, 105, 115, 95, 85, 90],
        'revenue': [45.5, 38.2, 42.0, 52.1, 49.3, 55.8]
    })
    
    st.dataframe(example_data)
    
    st.markdown("""
    4. **참고사항**:
       - 그룹 열과 종속변수 열은 업로드 후 선택할 수 있습니다.
       - 여러 종속변수에 대해 분석하려면 각 변수를 개별 열로 추가하세요.
       - 사이드바에서 CSV 파일을 업로드하고 필요한 열을 선택하세요.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 샘플 데이터 생성 옵션
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">샘플 데이터 생성</div>', unsafe_allow_html=True)
    
    st.markdown("""
    테스트를 위한 샘플 데이터를 생성하여 다운로드할 수 있습니다.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_groups = st.selectbox("그룹 수", [2, 3, 4], 0)
    with col2:
        samples_per_group = st.number_input("그룹당 샘플 수", min_value=10, max_value=1000, value=50)
    with col3:
        effect_size = st.slider("효과 크기", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    
    if st.button("샘플 데이터 생성"):
        try:
            # 그룹 생성
            groups = [chr(65 + i) for i in range(num_groups)]  # A, B, C, ...
            
            # 데이터 생성
            np.random.seed(42)  # 재현성을 위한 시드 설정
            
            data = []
            
            for i, group in enumerate(groups):
                # 기본 전환율
                base_conv = 0.1
                # 그룹별로 다른 효과 크기 적용
                if i > 0:
                    group_effect = effect_size * (i / (num_groups - 1)) if num_groups > 1 else effect_size
                else:
                    group_effect = 0
                
                # 각 그룹별 샘플 생성
                for _ in range(samples_per_group):
                    # 전환율
                    conv_rate = np.random.normal(base_conv + group_effect, 0.05)
                    conv_rate = max(0, min(1, conv_rate))  # 0~1 범위로 제한
                    
                    # 체류 시간
                    time_spent = np.random.normal(100 - i*10, 20)
                    time_spent = max(10, time_spent)  # 최소 10초
                    
                    # 매출
                    revenue = np.random.normal(40 + i*5, 10)
                    revenue = max(0, revenue)  # 최소 0
                    
                    data.append({
                        'group': group,
                        'conversion_rate': round(conv_rate, 3),
                        'time_spent': round(time_spent, 1),
                        'revenue': round(revenue, 2)
                    })
            
            # 데이터프레임 생성
            sample_df = pd.DataFrame(data)
            
            # CSV 다운로드 링크 생성
            csv = sample_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="ab_test_sample_data.csv">샘플 데이터 다운로드</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # 데이터 미리보기
            st.markdown("#### 샘플 데이터 미리보기")
            st.dataframe(sample_df.head(10))
            
        except Exception as e:
            st.error(f"샘플 데이터 생성 중 오류 발생: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# 푸터
st.markdown("""
---
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    © A/B 테스트 결과 통합 대시보드 | 데이터 기반 의사결정을 위한 통계 분석 도구
</div>
""", unsafe_allow_html=True)
