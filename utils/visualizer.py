import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Union, Any
import base64


class Visualizer:
    """A/B 테스트 결과를 시각화하는 클래스"""
    
    def __init__(self, data_processor, statistical_tester, theme="streamlit"):
        """
        Args:
            data_processor: DataProcessor 인스턴스
            statistical_tester: StatisticalTester 인스턴스
            theme: 시각화에 사용할 테마 ('streamlit', 'plotly', 'seaborn')
        """
        self.data_processor = data_processor
        self.statistical_tester = statistical_tester
        self.theme = theme
        
        # 기본 테마 설정
        if theme == "seaborn":
            sns.set_theme(style="whitegrid")
        
        # 색상 팔레트 설정
        self.color_palette = px.colors.qualitative.Plotly
        
    def plot_distribution_comparison(self) -> go.Figure:
        """각 그룹의 데이터 분포 비교 시각화 (Plotly)"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다.")
        
        group_data = self.data_processor.get_group_data()
        
        # 플롯을 위한 데이터 준비
        fig = go.Figure()
        
        for i, (group, data) in enumerate(group_data.items()):
            # 커널 밀도 추정
            kde_data = data.to_list()
            
            fig.add_trace(go.Violin(
                x=[group] * len(kde_data),
                y=kde_data,
                name=group,
                box_visible=True,
                meanline_visible=True,
                line_color=self.color_palette[i % len(self.color_palette)],
                fillcolor=self.color_palette[i % len(self.color_palette)],
                opacity=0.7,
                side='both',  # 양쪽에 표시
                points='all',  # 모든 데이터 포인트 표시 (선택 사항)
                jitter=0.05,   # 데이터 포인트 지터링 (선택 사항)
                pointpos=-0.1  # 데이터 포인트 위치 조정 (선택 사항)
            ))

        # 대시보드 레이아웃 설정
        fig.update_layout(
            title=f"{self.data_processor.target_col} 그룹별 분포 비교",
            xaxis_title="그룹",
            yaxis_title=self.data_processor.target_col,
            violingap=0.3,           # 바이올린 플롯 간 간격 추가
            violinmode='group',      # 겹치지 않게 그룹화
            template="plotly_dark",  # 다크 테마 적용 (가독성 향상)
            legend_title_text="그룹"
        )
        
        # 각 그룹의 요약 통계를 주석으로 추가
        summary = self.data_processor.get_group_summary()
        annotations = []
        
        for i, group in enumerate(self.data_processor.groups):
            group_stats = summary.loc[group]
            stats_text = (
                f"<b>{group}</b><br>"
                f"개수: {group_stats['개수']:.0f}<br>"
                f"평균: {group_stats['평균']:.3f}<br>"
                f"표준편차: {group_stats['표준편차']:.3f}<br>"
                f"중앙값: {group_stats['중앙값']:.3f}"
            )
            
            annotations.append(dict(
                x=group,
                y=group_stats['최대값'] * 1.1,  # 위치 약간 조정
                text=stats_text,
                showarrow=False,
                font=dict(
                    size=11,
                    color="white"  # 다크 테마에 맞춰 흰색 텍스트
                ),
                bgcolor="rgba(0,0,0,0.6)",  # 반투명 검은 배경으로 가독성 향상
                bordercolor="white",
                borderwidth=1,
                borderpad=4
            ))
        
        fig.update_layout(annotations=annotations)
        
        return fig
    
    def plot_distribution_comparison_histogram(self) -> go.Figure:
        """각 그룹의 데이터 분포를 히스토그램과 KDE로 비교 시각화"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다.")
        
        group_data = self.data_processor.get_group_data()
        
        # 플롯을 위한 데이터 준비
        fig = go.Figure()
        
        # 히스토그램 빈(bin) 수 계산 - 모든 데이터에 공통으로 적용
        all_data = []
        for data in group_data.values():
            all_data.extend(data.tolist())
        
        # Sturges 공식 사용하여 빈 수 계산
        import numpy as np
        bins = int(np.ceil(np.log2(len(all_data)) + 1))
        
        # 각 그룹별로 히스토그램과 KDE 그리기
        for i, (group, data) in enumerate(group_data.items()):
            color = self.color_palette[i % len(self.color_palette)]
            
            # 히스토그램
            fig.add_trace(go.Histogram(
                x=data,
                name=f"{group} (히스토그램)",
                opacity=0.5,
                marker_color=color,
                nbinsx=bins,
                histnorm='probability density',  # 확률 밀도로 정규화
                showlegend=True
            ))
            
            # KDE (커널 밀도 추정)
            import numpy as np
            from scipy import stats
            
            kde_x = np.linspace(min(all_data), max(all_data), 500)
            kde = stats.gaussian_kde(data)
            kde_y = kde(kde_x)
            
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name=f"{group} (KDE)",
                line=dict(color=color, width=2),
                showlegend=True
            ))
        
        # 레이아웃 설정
        fig.update_layout(
            title=f"{self.data_processor.target_col} 그룹별 분포 비교",
            xaxis_title=self.data_processor.target_col,
            yaxis_title="확률 밀도",
            template="plotly_dark",
            legend_title_text="그룹 및 그래프 유형",
            bargap=0.1,       # 히스토그램 바 간 간격
            barmode='overlay', # 히스토그램 겹침
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # 각 그룹의 요약 통계를 주석으로 추가
        summary = self.data_processor.get_group_summary()
        annotations = []
        
        max_y = fig.data[0].y.max()  # 첫 번째 히스토그램의 최대 높이
        for i, (trace1, trace2) in enumerate(zip(fig.data[::2], fig.data[1::2])):  # 히스토그램과 KDE 쌍으로 순회
            group = self.data_processor.groups[i]
            group_stats = summary.loc[group]
            
            # 평균선 추가
            fig.add_shape(
                type="line",
                x0=group_stats['평균'],
                y0=0,
                x1=group_stats['평균'],
                y1=max_y * 0.9,
                line=dict(
                    color=self.color_palette[i % len(self.color_palette)],
                    width=2,
                    dash="dash",
                ),
            )
            
            # 통계 주석 추가
            annotations.append(dict(
                x=group_stats['평균'],
                y=max_y * (0.95 - i * 0.15),  # 겹치지 않도록 조정
                text=(
                    f"<b>{group}</b><br>"
                    f"평균: {group_stats['평균']:.3f}<br>"
                    f"표준편차: {group_stats['표준편차']:.3f}"
                ),
                showarrow=True,
                arrowhead=1,
                arrowcolor=self.color_palette[i % len(self.color_palette)],
                font=dict(
                    size=11,
                    color="white"
                ),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor=self.color_palette[i % len(self.color_palette)],
                borderwidth=1,
                borderpad=4
            ))
        
        fig.update_layout(annotations=annotations)
        
        return fig

    def plot_distribution_comparison_ridgeline(self) -> go.Figure:
        """각 그룹의 데이터 분포를 Ridgeline Plot으로 비교 시각화"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다.")
        
        group_data = self.data_processor.get_group_data()
        
        # Ridgeline Plot을 위한 데이터 준비
        import numpy as np
        from scipy import stats
        
        # 모든 데이터 범위 계산
        all_data = []
        for data in group_data.values():
            all_data.extend(data.tolist())
        
        x_min, x_max = min(all_data), max(all_data)
        x_range = np.linspace(x_min, x_max, 500)
        
        # 각 그룹의 KDE 계산
        kde_data = {}
        for group, data in group_data.items():
            if len(data) > 1:  # 충분한 데이터가 있는 경우에만 KDE 계산
                kde = stats.gaussian_kde(data)
                kde_data[group] = kde(x_range)
            else:
                # 데이터가 부족한 경우 더미 데이터 생성
                kde_data[group] = np.zeros_like(x_range)
        
        # KDE 최대값 찾기 (정규화를 위해)
        max_density = max([max(kde) for kde in kde_data.values()]) if kde_data else 1
        
        # 각 그룹별 통계 가져오기
        summary = self.data_processor.get_group_summary()
        
        # 플롯 생성
        fig = go.Figure()
        
        # 각 그룹별 Ridgeline 추가
        y_offset = 0
        y_step = 1.0  # 각 분포 간 간격
        
        for i, (group, kde_y) in enumerate(kde_data.items()):
            # 색상 설정
            color = self.color_palette[i % len(self.color_palette)]
            
            # KDE 곡선 추가
            fig.add_trace(go.Scatter(
                x=x_range,
                y=kde_y / max_density * y_step * 0.9 + y_offset,  # 정규화 및 오프셋 적용
                mode='lines',
                fill='tozeroy',
                name=group,
                line=dict(color=color, width=2),
                fillcolor=f'rgba{tuple(int(c) for c in color[4:-1].split(",")) + (0.5,)}'  # 색상 투명도 조정
            ))
            
            # 평균선 추가
            group_mean = summary.loc[group]['평균']
            fig.add_shape(
                type="line",
                x0=group_mean,
                y0=y_offset,
                x1=group_mean,
                y1=y_offset + (max(kde_y) / max_density * y_step * 0.9),
                line=dict(color=color, width=2, dash="dot"),
            )
            
            # 통계 정보 주석 추가
            fig.add_annotation(
                x=group_mean,
                y=y_offset + (max(kde_y) / max_density * y_step * 0.9) * 0.7,
                text=(
                    f"<b>{group}</b><br>"
                    f"평균: {group_mean:.3f}<br>"
                    f"표준편차: {summary.loc[group]['표준편차']:.3f}<br>"
                    f"N={summary.loc[group]['개수']:.0f}"
                ),
                showarrow=True,
                arrowhead=1,
                arrowcolor=color,
                font=dict(size=11, color="white"),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor=color,
                borderwidth=1,
                borderpad=3,
                align="left"
            )
            
            # 다음 그룹을 위한 오프셋 증가
            y_offset += y_step
        
        # 레이아웃 설정
        fig.update_layout(
            title=f"{self.data_processor.target_col} 그룹별 분포 비교 (Ridgeline Plot)",
            xaxis_title=self.data_processor.target_col,
            yaxis_title="그룹",
            template="plotly_dark",
            showlegend=True,
            legend_title_text="그룹",
            # y축 눈금 설정
            yaxis=dict(
                tickvals=[i * y_step + y_step * 0.3 for i in range(len(group_data))],
                ticktext=list(group_data.keys()),
                zeroline=False,
                showgrid=False,
            ),
            height=100 + 150 * len(group_data),  # 그룹 수에 따라 높이 조정
            margin=dict(l=50, r=50, t=80, b=50),
        )
        
        return fig

    def plot_distribution_comparison_boxplot(self) -> go.Figure:
        """각 그룹의 데이터 분포를 Box Plot으로 비교 시각화"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다.")
        
        group_data = self.data_processor.get_group_data()
        
        # 플롯을 위한 데이터 준비
        fig = go.Figure()
        
        for i, (group, data) in enumerate(group_data.items()):
            # Box Plot 추가
            fig.add_trace(go.Box(
                y=data,
                name=group,
                boxmean=True,  # 평균 표시
                boxpoints='all',  # 모든 데이터 포인트 표시
                jitter=0.3,      # 데이터 포인트 지터링
                pointpos=-1.8,    # 데이터 포인트 위치 조정
                marker=dict(
                    color=self.color_palette[i % len(self.color_palette)],
                    size=4        # 데이터 포인트 크기
                ),
                line=dict(color=self.color_palette[i % len(self.color_palette)])
            ))
        
        # 대시보드 레이아웃 설정
        fig.update_layout(
            title=f"{self.data_processor.target_col} 그룹별 분포 비교 (Box Plot)",
            yaxis_title=self.data_processor.target_col,
            xaxis_title="그룹",
            template="plotly_dark",  # 다크 테마 적용
            boxmode='group',         # 그룹화된 박스 표시
            boxgap=0.1,              # 박스 간 간격
            showlegend=True,
            legend_title_text="그룹",
        )
        
        # 각 그룹의 요약 통계를 주석으로 추가
        summary = self.data_processor.get_group_summary()
        annotations = []
        
        y_max = max([max(data) for data in group_data.values()])
        
        for i, group in enumerate(self.data_processor.groups):
            group_stats = summary.loc[group]
            stats_text = (
                f"<b>{group}</b><br>"
                f"개수: {group_stats['개수']:.0f}<br>"
                f"평균: {group_stats['평균']:.3f}<br>"
                f"중앙값: {group_stats['중앙값']:.3f}<br>"
                f"IQR: {group_stats['3사분위수'] - group_stats['1사분위수']:.3f}"
            )
            
            annotations.append(dict(
                x=i,
                y=y_max * 1.1,
                text=stats_text,
                showarrow=False,
                font=dict(
                    size=11,
                    color="white"
                ),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor=self.color_palette[i % len(self.color_palette)],
                borderwidth=1,
                borderpad=4
            ))
        
        fig.update_layout(annotations=annotations)
        
        return fig

    def plot_qq_plots(self) -> go.Figure:
        """각 그룹의 Q-Q 플롯 (정규성 검정)"""
        if not self.statistical_tester.normality_results:
            self.statistical_tester.test_normality()
        
        # 서브플롯 생성
        cols = min(3, len(self.data_processor.groups))
        rows = (len(self.data_processor.groups) + cols - 1) // cols  # 올림 나눗셈
        
        fig = go.Figure()
        
        for i, group in enumerate(self.data_processor.groups):
            qq_data = self.statistical_tester.normality_results[group]["qq_plot"]
            theo_q = qq_data["theoretical_quantiles"]
            sample_q = qq_data["sample_quantiles"]
            
            # 정규성 검정 결과
            shapiro_result = self.statistical_tester.normality_results[group]["shapiro"]
            p_value = shapiro_result["p_value"]
            is_normal = shapiro_result["normal"]
            
            # QQ 플롯 생성
            fig.add_trace(go.Scatter(
                x=theo_q,
                y=sample_q,
                mode='markers',
                name=f"{group}",
                marker=dict(
                    color=self.color_palette[i % len(self.color_palette)],
                    size=6
                )
            ))
            
            # 참조선 (y=x)
            min_val = min(theo_q.min(), sample_q.min())
            max_val = max(theo_q.max(), sample_q.max())
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name=f"{group} - 참조선",
                line=dict(
                    color=self.color_palette[i % len(self.color_palette)],
                    width=1,
                    dash='dot'
                ),
                showlegend=False
            ))
            
            # 결과 주석 추가
            fig.add_annotation(
                x=theo_q.min() + (theo_q.max() - theo_q.min()) * 0.1,
                y=sample_q.max() - (sample_q.max() - sample_q.min()) * 0.1,
                text=(f"<b>{group}</b><br>"
                      f"Shapiro-Wilk p-value: {p_value:.4f}<br>"
                      f"정규성: {'만족' if is_normal else '불만족'}"),
                showarrow=False,
                font=dict(size=10, color=self.color_palette[i % len(self.color_palette)])
            )
        
        fig.update_layout(
            title="그룹별 Q-Q 플롯 (정규성 검정)",
            xaxis_title="이론적 분위수",
            yaxis_title="표본 분위수",
            template="plotly_white",
            legend_title="그룹",
            height=500,
            width=800
        )
        
        return fig
    
    def plot_mean_comparison(self) -> go.Figure:
        """그룹별 평균 비교 (오차 막대 포함)"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다.")
        
        group_data = self.data_processor.get_group_data()
        
        # 평균 및 표준오차 계산
        means = [data.mean() for data in group_data.values()]
        stderrs = [data.sem() for data in group_data.values()]
        
        # 그래프 데이터 생성
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(self.data_processor.groups),
            y=means,
            error_y=dict(
                type='data',
                array=stderrs,
                visible=True
            ),
            marker_color=[self.color_palette[i % len(self.color_palette)] 
                         for i in range(len(self.data_processor.groups))]
        ))
        
        # 평균값 주석 추가
        annotations = []
        for i, (group, mean, stderr) in enumerate(zip(self.data_processor.groups, means, stderrs)):
            annotations.append(dict(
                x=group,
                y=mean + stderr + (max(means) * 0.05),
                text=f"{mean:.3f} ± {stderr:.3f}",
                showarrow=False,
                font=dict(size=10)
            ))
        
        # 가설 검정 결과 표시
        if hasattr(self.statistical_tester, 'hypothesis_test_results') and self.statistical_tester.hypothesis_test_results:
            test_result = self.statistical_tester.hypothesis_test_results
            result_text = (f"검정: {test_result.get('test_name', '분석 중...')}<br>"
                          f"p-value: {test_result.get('p_value', 'N/A'):.4f}<br>"
                          f"유의성: {'있음' if test_result.get('significant', False) else '없음'}")
            
            fig.add_annotation(
                x=0.5,
                y=1.15,
                xref="paper",
                yref="paper",
                text=result_text,
                showarrow=False,
                font=dict(size=12),
                align="center",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white"
            )
        
        # 레이아웃 설정
        fig.update_layout(
            title=f"{self.data_processor.target_col} 그룹별 평균 비교",
            xaxis_title="그룹",
            yaxis_title=f"{self.data_processor.target_col} 평균",
            annotations=annotations,
            template="plotly_white",
            showlegend=False,
            height=500,
            width=800
        )
        
        return fig
    
    def plot_effect_size(self) -> go.Figure:
        """효과 크기 시각화"""
        if not hasattr(self.statistical_tester, 'effect_size_results') or not self.statistical_tester.effect_size_results:
            self.statistical_tester.calculate_effect_size()
        
        effect_size = self.statistical_tester.effect_size_results
        
        # Cohen's d 또는 Eta-squared에 따라 다른 시각화
        if effect_size["measure"] == "Cohen's d":
            # Cohen's d 시각화 (두 그룹 비교)
            d_value = effect_size["value"]
            interpretation = effect_size["interpretation"]
            comparison = effect_size["comparison"]
            
            # 효과 크기의 임계값
            thresholds = [-1.2, -0.8, -0.5, -0.2, 0.2, 0.5, 0.8, 1.2]
            labels = ["매우 큰<br>음의 효과", "큰<br>음의 효과", "중간<br>음의 효과", "작은<br>음의 효과", 
                     "작은<br>양의 효과", "중간<br>양의 효과", "큰<br>양의 효과", "매우 큰<br>양의 효과"]
            
            # 게이지 차트 생성
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=d_value,
                title={"text": f"효과 크기 (Cohen's d)<br><sub>{comparison}</sub>"},
                gauge={
                    "axis": {"range": [-1.5, 1.5], "tickvals": thresholds},
                    "bar": {"color": "#1E88E5"},
                    "steps": [
                        {"range": [-1.5, -0.8], "color": "#EF5350"},
                        {"range": [-0.8, -0.5], "color": "#FFA726"},
                        {"range": [-0.5, -0.2], "color": "#FFEE58"},
                        {"range": [-0.2, 0.2], "color": "#E0E0E0"},
                        {"range": [0.2, 0.5], "color": "#FFEE58"},
                        {"range": [0.5, 0.8], "color": "#FFA726"},
                        {"range": [0.8, 1.5], "color": "#66BB6A"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": d_value
                    }
                },
                domain={"x": [0, 1], "y": [0, 1]}
            ))
            
            # 해석 추가
            fig.add_annotation(
                x=0.5, y=0.3,
                xref="paper", yref="paper",
                text=f"해석: <b>{interpretation}</b>",
                showarrow=False,
                font=dict(size=14)
            )
            
        else:
            # Eta-squared 시각화 (3개 이상 그룹 비교)
            eta_value = effect_size["value"]
            interpretation = effect_size["interpretation"]
            
            # 효과 크기 임계값
            thresholds = [0, 0.01, 0.06, 0.14, 0.25]
            labels = ["효과 없음", "작은 효과", "중간 효과", "큰 효과", "매우 큰 효과"]
            
            # 게이지 차트 생성
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=eta_value,
                number={"valueformat": ".3f"},
                title={"text": "효과 크기 (Eta-squared)"},
                gauge={
                    "axis": {"range": [0, 0.3], "tickvals": thresholds},
                    "bar": {"color": "#1E88E5"},
                    "steps": [
                        {"range": [0, 0.01], "color": "#E0E0E0"},
                        {"range": [0.01, 0.06], "color": "#FFEE58"},
                        {"range": [0.06, 0.14], "color": "#FFA726"},
                        {"range": [0.14, 0.3], "color": "#66BB6A"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": eta_value
                    }
                },
                domain={"x": [0, 1], "y": [0, 1]}
            ))
            
            # 해석 추가
            fig.add_annotation(
                x=0.5, y=0.3,
                xref="paper", yref="paper",
                text=f"해석: <b>{interpretation}</b>",
                showarrow=False,
                font=dict(size=14)
            )
        
        fig.update_layout(
            height=400,
            width=600,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def plot_bootstrap_ci(self) -> go.Figure:
        """부트스트랩 신뢰구간 시각화"""
        if not hasattr(self.statistical_tester, 'bootstrap_results') or not self.statistical_tester.bootstrap_results:
            self.statistical_tester.perform_bootstrap()
        
        bootstrap_results = self.statistical_tester.bootstrap_results
        
        # 각 그룹별 평균 및 신뢰구간
        groups = []
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for group, result in bootstrap_results.items():
            if group != "difference":  # difference는 별도로 처리
                groups.append(group)
                means.append(result["mean"])
                ci_lowers.append(result["ci_lower"])
                ci_uppers.append(result["ci_upper"])
        
        # 오차 범위 계산
        error_minus = [mean - ci_low for mean, ci_low in zip(means, ci_lowers)]
        error_plus = [ci_up - mean for mean, ci_up in zip(means, ci_uppers)]
        
        # 그래프 생성
        fig = go.Figure()
        
        # 각 그룹별 평균 및 신뢰구간 플롯
        fig.add_trace(go.Scatter(
            x=groups,
            y=means,
            mode='markers',
            error_y=dict(
                type='data',
                symmetric=False,
                array=error_plus,
                arrayminus=error_minus,
                visible=True
            ),
            marker=dict(
                color=[self.color_palette[i % len(self.color_palette)] for i in range(len(groups))],
                size=10
            ),
            name="그룹별 평균 및 95% 신뢰구간"
        ))
        
        # 그룹별 평균값 표시
        for i, (group, mean, ci_low, ci_up) in enumerate(zip(groups, means, ci_lowers, ci_uppers)):
            fig.add_annotation(
                x=group,
                y=mean,
                text=f"평균: {mean:.3f}<br>95% CI: [{ci_low:.3f}, {ci_up:.3f}]",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=self.color_palette[i % len(self.color_palette)],
                ax=0,
                ay=-40
            )
        
        # 두 그룹 간 차이가 있는 경우 추가 표시
        if "difference" in bootstrap_results:
            diff_result = bootstrap_results["difference"]
            groups_comp = diff_result["groups"]
            mean_diff = diff_result["mean_diff"]
            ci_low = diff_result["ci_lower"]
            ci_up = diff_result["ci_upper"]
            significant = diff_result["significant"]
            
            fig.add_annotation(
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                text=(f"<b>그룹 간 차이 ({groups_comp})</b><br>"
                      f"평균 차이: {mean_diff:.3f}<br>"
                      f"95% 신뢰구간: [{ci_low:.3f}, {ci_up:.3f}]<br>"
                      f"통계적 유의성: {'있음' if significant else '없음'}"),
                showarrow=False,
                font=dict(size=12),
                align="center",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white"
            )
        
        # 레이아웃 설정
        fig.update_layout(
            title="그룹별 부트스트랩 평균 추정 및 95% 신뢰구간",
            xaxis_title="그룹",
            yaxis_title=self.data_processor.target_col,
            template="plotly_white",
            height=500,
            width=800
        )
        
        return fig
    
    def plot_error_matrix(self) -> go.Figure:
        """제1종, 제2종 오류 매트릭스 시각화"""
        if not hasattr(self.statistical_tester, 'error_analysis') or not self.statistical_tester.error_analysis:
            self.statistical_tester.analyze_errors()
        
        error_analysis = self.statistical_tester.error_analysis
        error_matrix = error_analysis["error_matrix"]
        
        # 매트릭스 데이터 준비
        matrix_data = np.array([
            [error_matrix["not_reject_null_no_diff"], error_matrix["not_reject_null_true_diff"]],
            [error_matrix["reject_null_no_diff"], error_matrix["reject_null_true_diff"]]
        ])
        
        # 그룹 이름
        axis_labels = ["귀무가설이 참", "대립가설이 참"]
        decision_labels = ["귀무가설 채택", "귀무가설 기각"]
        
        # 주석 텍스트
        annotations = [
            ["올바른 결정<br>(1-α)", "제2종 오류<br>(β)"],
            ["제1종 오류<br>(α)", "올바른 결정<br>(검정력, 1-β)"]
        ]
        
        # 히트맵 생성
        fig = ff.create_annotated_heatmap(
            z=matrix_data,
            x=axis_labels,
            y=decision_labels,
            annotation_text=annotations,
            colorscale='Blues',
            showscale=True
        )
        
        # 매트릭스 값 추가
        for i in range(2):
            for j in range(2):
                fig.data[0].customdata = np.zeros((2, 2), dtype=object)
                value_text = f"<b>{matrix_data[i, j]:.3f}</b>"
                fig.data[0].customdata[i, j] = value_text
                
                fig.data[0].hovertemplate = (
                    "<b>결정:</b> %{y}<br>"
                    "<b>실제:</b> %{x}<br>"
                    "<b>확률:</b> %{customdata}<br>"
                    "<extra></extra>"
                )
        
        # 추가 정보 표시
        statistical_info = (
            f"<b>통계 정보:</b><br>"
            f"- 유의수준(α): {error_analysis['type_1_error']:.3f}<br>"
            f"- 제2종 오류(β): {error_analysis['type_2_error']:.3f}<br>"
            f"- 검정력(1-β): {error_analysis['power']:.3f}<br>"
            f"- 효과 크기: {error_analysis['effect_size']:.3f}"
        )
        
        fig.add_annotation(
            x=1.25,
            y=0.5,
            xref="paper",
            yref="paper",
            text=statistical_info,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white"
        )
        
        # 레이아웃 설정
        fig.update_layout(
            title="통계적 의사 결정 매트릭스",
            xaxis_title="실제 상태",
            yaxis_title="의사 결정",
            width=800,
            height=500,
            margin=dict(l=60, r=140, t=60, b=60)
        )
        
        return fig
    
    def create_effect_size_gauge(self) -> go.Figure:
        """효과 크기만 시각화하는 간소화된 게이지 차트"""
        if not hasattr(self.statistical_tester, 'effect_size_results') or not self.statistical_tester.effect_size_results:
            self.statistical_tester.calculate_effect_size()
        
        effect_size = self.statistical_tester.effect_size_results
        
        # Cohen's d 또는 Eta-squared에 따라 다른 시각화
        if effect_size["measure"] == "Cohen's d":
            # Cohen's d 시각화 (두 그룹 비교)
            d_value = effect_size["value"]
            interpretation = effect_size["interpretation"]
            
            # 효과 크기의 임계값
            thresholds = [-1.2, -0.8, -0.5, -0.2, 0.2, 0.5, 0.8, 1.2]
            
            # 게이지 차트 생성
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=d_value,
                number={"font": {"size": 28}},
                gauge={
                    "axis": {"range": [-1.5, 1.5], "tickvals": thresholds},
                    "bar": {"color": "#1E88E5"},
                    "steps": [
                        {"range": [-1.5, -0.8], "color": "#EF5350"},
                        {"range": [-0.8, -0.5], "color": "#FFA726"},
                        {"range": [-0.5, -0.2], "color": "#FFEE58"},
                        {"range": [-0.2, 0.2], "color": "#E0E0E0"},
                        {"range": [0.2, 0.5], "color": "#FFEE58"},
                        {"range": [0.5, 0.8], "color": "#FFA726"},
                        {"range": [0.8, 1.5], "color": "#66BB6A"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": d_value
                    }
                },
                domain={"x": [0, 1], "y": [0, 1]}
            ))
                
        else:
            # Eta-squared 시각화 (3개 이상 그룹 비교)
            eta_value = effect_size["value"]
            interpretation = effect_size["interpretation"]
            
            # 효과 크기 임계값
            thresholds = [0, 0.01, 0.06, 0.14, 0.25]
            
            # 게이지 차트 생성
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=eta_value,
                number={"valueformat": ".3f", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 0.3], "tickvals": thresholds},
                    "bar": {"color": "#1E88E5"},
                    "steps": [
                        {"range": [0, 0.01], "color": "#E0E0E0"},
                        {"range": [0.01, 0.06], "color": "#FFEE58"},
                        {"range": [0.06, 0.14], "color": "#FFA726"},
                        {"range": [0.14, 0.3], "color": "#66BB6A"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": eta_value
                    }
                },
                domain={"x": [0, 1], "y": [0, 1]}
            ))
        
        # 간소화된 레이아웃
        fig.update_layout(
            height=300,
            width=500,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        
        # 해석 주석 추가
        fig.add_annotation(
            x=0.5, y=0.2,
            xref="paper", yref="paper",
            text=f"<b>해석: {interpretation}</b>",
            showarrow=False,
            font=dict(size=14)
        )
        
        return fig
