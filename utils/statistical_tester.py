import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import statsmodels.api as sm
from statsmodels.formula.api import ols
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from enum import Enum


class TestType(Enum):
    """테스트 유형 분류"""
    PARAMETRIC = "parametric"
    NON_PARAMETRIC = "non_parametric"


class StatisticalTester:
    """통계 검정을 수행하는 클래스"""
    
    def __init__(self, data_processor):
        """
        Args:
            data_processor: DataProcessor 클래스의 인스턴스
        """
        self.data_processor = data_processor
        self.normality_results = {}
        self.homogeneity_results = {}
        self.hypothesis_test_results = {}
        self.effect_size_results = {}
        self.bootstrap_results = {}
        self.error_analysis = {}
        self.test_type = None
        self.alpha = 0.05  # 기본 유의수준
        
    def set_alpha(self, alpha: float) -> None:
        """유의수준 설정"""
        if 0 < alpha < 1:
            self.alpha = alpha
        else:
            raise ValueError("유의수준은 0과 1 사이의 값이어야 합니다.")
    
    def test_normality(self) -> Dict[str, Dict[str, Any]]:
        """각 그룹의 정규성 검정 (Shapiro-Wilk, QQ Plot 데이터)"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다. 그룹 열과 타겟 열을 먼저 설정해 주세요.")
        
        self.normality_results = {}
        
        # 각 그룹별로 정규성 검정
        for group in self.data_processor.groups:
            group_data = self.data_processor.get_group_data(group)
            
            # 샘플 크기가 3보다 작으면 정규성 검정을 수행할 수 없음
            if len(group_data) < 3:
                self.normality_results[group] = {
                    "shapiro": {"statistic": None, "p_value": None, "normal": None},
                    "qq_plot": {"theoretical_quantiles": None, "sample_quantiles": None}
                }
                continue
            
            # Shapiro-Wilk 검정
            shapiro_test = stats.shapiro(group_data)
            is_normal = shapiro_test.pvalue > self.alpha
            
            # QQ Plot 데이터 생성
            sample_quantiles = np.sort(group_data)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sample_quantiles)))
            
            self.normality_results[group] = {
                "shapiro": {
                    "statistic": shapiro_test.statistic,
                    "p_value": shapiro_test.pvalue,
                    "normal": is_normal
                },
                "qq_plot": {
                    "theoretical_quantiles": theoretical_quantiles,
                    "sample_quantiles": sample_quantiles
                }
            }
        
        return self.normality_results
    
    def test_homogeneity(self) -> Dict[str, Dict[str, float]]:
        """등분산성 검정 (Bartlett, Levene)"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다. 그룹 열과 타겟 열을 먼저 설정해 주세요.")
        
        # 각 그룹의 데이터 수집
        group_data = []
        for group in self.data_processor.groups:
            group_data.append(self.data_processor.get_group_data(group))
        
        # Bartlett 검정 - 정규성 가정이 충족될 때 사용
        bartlett_test = stats.bartlett(*group_data)
        
        # Levene 검정 - 정규성 가정이 충족되지 않아도 사용 가능
        levene_test = stats.levene(*group_data, center='median')
        
        self.homogeneity_results = {
            "bartlett": {
                "statistic": bartlett_test.statistic,
                "p_value": bartlett_test.pvalue,
                "equal_variances": bartlett_test.pvalue > self.alpha
            },
            "levene": {
                "statistic": levene_test.statistic,
                "p_value": levene_test.pvalue,
                "equal_variances": levene_test.pvalue > self.alpha
            }
        }
        
        return self.homogeneity_results
    
    def determine_test_type(self) -> TestType:
        """정규성 및 등분산성 검정 결과를 기반으로 적절한 검정 유형 결정"""
        if not self.normality_results or not self.homogeneity_results:
            self.test_normality()
            self.test_homogeneity()
            
        # 모든 그룹이 정규성을 만족하는지 확인
        all_normal = all(result["shapiro"]["normal"] for result in self.normality_results.values())
        
        # 등분산성을 만족하는지 확인 (Levene 검정 사용)
        equal_variances = self.homogeneity_results["levene"]["equal_variances"]
        
        # 그룹 수 확인
        num_groups = len(self.data_processor.groups)
        
        # 결정 로직
        self.test_type = TestType.PARAMETRIC if all_normal else TestType.NON_PARAMETRIC
        
        return self.test_type
    
    def run_hypothesis_test(self) -> Dict[str, Any]:
        """그룹 수와 데이터 특성에 따라 적절한 가설 검정 실행"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다. 그룹 열과 타겟 열을 먼저 설정해 주세요.")
        
        # 테스트 유형 결정 (아직 결정되지 않은 경우)
        if self.test_type is None:
            self.determine_test_type()
            
        num_groups = len(self.data_processor.groups)
        group_data = self.data_processor.get_group_data()
        
        # 데이터 준비
        all_data = []
        all_groups = []
        for group, data in group_data.items():
            all_data.extend(data.tolist())
            all_groups.extend([group] * len(data))
        
        # 2개 그룹 비교
        if num_groups == 2:
            if self.test_type == TestType.PARAMETRIC:
                # 독립 t-검정
                group1_data = group_data[self.data_processor.groups[0]]
                group2_data = group_data[self.data_processor.groups[1]]
                
                # 등분산성 여부에 따라 다른 t-검정 적용
                equal_var = self.homogeneity_results["levene"]["equal_variances"]
                t_test = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
                
                self.hypothesis_test_results = {
                    "test_name": "독립표본 t-검정" if equal_var else "Welch's t-검정",
                    "statistic": t_test.statistic,
                    "p_value": t_test.pvalue,
                    "significant": t_test.pvalue < self.alpha,
                    "equal_variances": equal_var,
                    "groups_compared": self.data_processor.groups
                }
            else:
                # 비모수 검정 (Mann-Whitney U)
                group1_data = group_data[self.data_processor.groups[0]]
                group2_data = group_data[self.data_processor.groups[1]]
                mw_test = stats.mannwhitneyu(group1_data, group2_data)
                
                self.hypothesis_test_results = {
                    "test_name": "Mann-Whitney U 검정",
                    "statistic": mw_test.statistic,
                    "p_value": mw_test.pvalue,
                    "significant": mw_test.pvalue < self.alpha,
                    "groups_compared": self.data_processor.groups
                }
        
        # 3개 이상 그룹 비교
        else:
            if self.test_type == TestType.PARAMETRIC:
                # 분산분석(ANOVA)
                df = pd.DataFrame({
                    'value': all_data,
                    'group': all_groups
                })
                
                # ANOVA 모델 피팅
                model = ols('value ~ C(group)', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # 사후 검정 (Tukey HSD)
                tukey = pairwise_tukeyhsd(endog=df['value'], groups=df['group'], alpha=self.alpha)
                tukey_result = pd.DataFrame(data=tukey._results_table.data[1:], 
                                           columns=tukey._results_table.data[0])
                
                self.hypothesis_test_results = {
                    "test_name": "일원배치 분산분석(ANOVA)",
                    "f_statistic": anova_table.loc['C(group)', 'F'],
                    "p_value": anova_table.loc['C(group)', 'PR(>F)'],
                    "significant": anova_table.loc['C(group)', 'PR(>F)'] < self.alpha,
                    "post_hoc": {
                        "method": "Tukey HSD",
                        "results": tukey_result.to_dict('records')
                    }
                }
            else:
                # 비모수 검정 (Kruskal-Wallis)
                kruskal = stats.kruskal(*[data for data in group_data.values()])
                
                # 사후 검정 (Dunn's test)
                # SciPy에는 없어서 직접 구현하거나 statsmodels의 MultiComparison 사용
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = pd.DataFrame({
                        'value': all_data,
                        'group': all_groups
                    })
                    
                    # 사후 검정을 위한 pairwise 비교
                    mc = MultiComparison(df['value'], df['group'])
                    dunn_result = mc.allpairtest(stats.mannwhitneyu, method='bonf')
                    
                    # 결과 변환
                    pairwise_results = []
                    for i, row in enumerate(dunn_result[0].data):
                        if i > 0:  # 헤더 행 제외
                            pairwise_results.append({
                                'group1': row[0],
                                'group2': row[1],
                                'statistic': row[2],
                                'p_value': row[3],
                                'significant': row[3] < self.alpha
                            })
                
                self.hypothesis_test_results = {
                    "test_name": "Kruskal-Wallis 검정",
                    "statistic": kruskal.statistic,
                    "p_value": kruskal.pvalue,
                    "significant": kruskal.pvalue < self.alpha,
                    "post_hoc": {
                        "method": "Dunn's test (Bonferroni 보정)",
                        "results": pairwise_results
                    }
                }
        
        return self.hypothesis_test_results
    
    def calculate_effect_size(self) -> Dict[str, Any]:
        """효과 크기 계산 (Cohen's d, 오즈비 등)"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다. 그룹 열과 타겟 열을 먼저 설정해 주세요.")
        
        # 효과 크기 계산 결과 초기화
        self.effect_size_results = {}
        group_data = self.data_processor.get_group_data()
        num_groups = len(self.data_processor.groups)
        
        # 두 그룹 비교: Cohen's d 계산
        if num_groups == 2:
            group1_data = group_data[self.data_processor.groups[0]]
            group2_data = group_data[self.data_processor.groups[1]]
            
            # Cohen's d 계산
            mean1, mean2 = group1_data.mean(), group2_data.mean()
            n1, n2 = len(group1_data), len(group2_data)
            var1, var2 = group1_data.var(), group2_data.var()
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            # Cohen's d
            cohen_d = (mean1 - mean2) / pooled_std
            
            # Cohen's d 해석
            if abs(cohen_d) < 0.2:
                effect_interpretation = "매우 작음"
            elif abs(cohen_d) < 0.5:
                effect_interpretation = "작음"
            elif abs(cohen_d) < 0.8:
                effect_interpretation = "중간"
            else:
                effect_interpretation = "큼"
                
            self.effect_size_results = {
                "measure": "Cohen's d",
                "value": cohen_d,
                "interpretation": effect_interpretation,
                "comparison": f"{self.data_processor.groups[0]} vs {self.data_processor.groups[1]}"
            }
        
        # 세 그룹 이상: Eta-squared 계산
        else:
            # 전체 그룹 데이터 준비
            all_data = []
            all_groups = []
            for group, data in group_data.items():
                all_data.extend(data)
                all_groups.extend([group] * len(data))
            
            df = pd.DataFrame({"value": all_data, "group": all_groups})
            
            # ANOVA 모델 피팅
            model = ols('value ~ C(group)', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Eta-squared 계산
            ss_group = anova_table.loc['C(group)', 'sum_sq']
            ss_total = ss_group + anova_table.loc['Residual', 'sum_sq']
            eta_squared = ss_group / ss_total
            
            # Eta-squared 해석
            if eta_squared < 0.01:
                effect_interpretation = "매우 작음"
            elif eta_squared < 0.06:
                effect_interpretation = "작음"
            elif eta_squared < 0.14:
                effect_interpretation = "중간"
            else:
                effect_interpretation = "큼"
                
            self.effect_size_results = {
                "measure": "Eta-squared",
                "value": eta_squared,
                "interpretation": effect_interpretation,
                "comparison": "전체 그룹 간 비교"
            }
        
        return self.effect_size_results
    
    def perform_bootstrap(self, n_resamples: int = 1000) -> Dict[str, Any]:
        """부트스트랩을 통한 신뢰구간 계산"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다. 그룹 열과 타겟 열을 먼저 설정해 주세요.")
        
        self.bootstrap_results = {}
        group_data = self.data_processor.get_group_data()
        
        # 각 그룹별 부트스트랩 샘플링
        for group in self.data_processor.groups:
            data = group_data[group]
            bootstrap_means = []
            
            # 부트스트랩 리샘플링 수행
            for _ in range(n_resamples):
                resample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(resample))
            
            # 신뢰구간 계산 (95%)
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            
            self.bootstrap_results[group] = {
                "mean": np.mean(data),
                "bootstrap_mean": np.mean(bootstrap_means),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "bootstrap_samples": bootstrap_means
            }
        
        # 두 그룹 간 차이에 대한 부트스트랩 (그룹이 2개일 경우)
        if len(self.data_processor.groups) == 2:
            group1, group2 = self.data_processor.groups
            data1, data2 = group_data[group1], group_data[group2]
            diff_means = []
            
            for _ in range(n_resamples):
                resample1 = np.random.choice(data1, size=len(data1), replace=True)
                resample2 = np.random.choice(data2, size=len(data2), replace=True)
                diff_means.append(np.mean(resample1) - np.mean(resample2))
            
            ci_diff_lower = np.percentile(diff_means, 2.5)
            ci_diff_upper = np.percentile(diff_means, 97.5)
            
            self.bootstrap_results["difference"] = {
                "groups": f"{group1} - {group2}",
                "mean_diff": np.mean(data1) - np.mean(data2),
                "bootstrap_mean_diff": np.mean(diff_means),
                "ci_lower": ci_diff_lower,
                "ci_upper": ci_diff_upper,
                "significant": (ci_diff_lower > 0 and ci_diff_upper > 0) or (ci_diff_lower < 0 and ci_diff_upper < 0)
            }
        
        return self.bootstrap_results
    
    def analyze_errors(self) -> Dict[str, Any]:
        """제1종, 제2종 오류 분석 및 검정력 계산"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다. 그룹 열과 타겟 열을 먼저 설정해 주세요.")
        
        group_data = self.data_processor.get_group_data()
        num_groups = len(self.data_processor.groups)
        
        # 제1종 오류 (알파): 이미 설정된 alpha 값 사용
        type_1_error = self.alpha
        
        # 효과 크기가 아직 계산되지 않았다면 계산
        if not self.effect_size_results:
            self.calculate_effect_size()
        
        # 검정력 및 제2종 오류 계산 (두 그룹 비교의 경우)
        if num_groups == 2:
            group1, group2 = self.data_processor.groups
            data1, data2 = group_data[group1], group_data[group2]
            n1, n2 = len(data1), len(data2)
            
            # 효과 크기
            effect_size = abs(self.effect_size_results["value"])
            
            # 제2종 오류(베타) 계산을 위한 검정력 분석
            if self.test_type == TestType.PARAMETRIC:
                # t-검정의 검정력
                from statsmodels.stats.power import TTestIndPower
                power_analysis = TTestIndPower()
                power = power_analysis.power(effect_size, nobs1=n1, ratio=n2/n1, alpha=type_1_error)
            else:
                # 비모수 검정의 경우 근사값 사용 (효율 ~0.95)
                from statsmodels.stats.power import TTestIndPower
                power_analysis = TTestIndPower()
                # 비모수 검정은 모수적 방법보다 약간 검정력이 낮음 (효율 계수 적용)
                power = power_analysis.power(effect_size * 0.95, nobs1=n1, ratio=n2/n1, alpha=type_1_error)
            
            type_2_error = 1 - power
            
            self.error_analysis = {
                "type_1_error": type_1_error,
                "type_2_error": type_2_error,
                "power": power,
                "effect_size": effect_size,
                "sample_sizes": {group1: n1, group2: n2},
                "error_matrix": {
                    "reject_null_true_diff": 1 - type_2_error,  # 옳은 결정 (1)
                    "not_reject_null_true_diff": type_2_error,  # 제2종 오류 (2)
                    "reject_null_no_diff": type_1_error,  # 제1종 오류 (3)
                    "not_reject_null_no_diff": 1 - type_1_error  # 옳은 결정 (4)
                }
            }
        
        else:
            # 3개 이상 그룹에 대한 검정력 분석은 복잡함
            # ANOVA의 검정력에 대한 근사값 제공
            from statsmodels.stats.power import FTestAnovaPower
            power_analysis = FTestAnovaPower()
            
            # 효과 크기 (eta-squared에서 f로 변환)
            effect_size = np.sqrt(self.effect_size_results["value"] / (1 - self.effect_size_results["value"]))
            
            # 전체 샘플 수
            total_n = sum(len(data) for data in group_data.values())
            
            # ANOVA 검정력 계산
            power = power_analysis.power(effect_size, num_groups, total_n, alpha=type_1_error)
            type_2_error = 1 - power
            
            self.error_analysis = {
                "type_1_error": type_1_error,
                "type_2_error": type_2_error,
                "power": power,
                "effect_size": effect_size,
                "sample_sizes": {group: len(data) for group, data in group_data.items()},
                "error_matrix": {
                    "reject_null_true_diff": 1 - type_2_error,  # 옳은 결정
                    "not_reject_null_true_diff": type_2_error,  # 제2종 오류
                    "reject_null_no_diff": type_1_error,  # 제1종 오류
                    "not_reject_null_no_diff": 1 - type_1_error  # 옳은 결정
                }
            }
        
        return self.error_analysis

    def pearson_correlation(self) -> Dict[str, Any]:
        """그룹 간 피어슨 상관계수 계산"""
        if not self.data_processor.groups or len(self.data_processor.groups) != 2:
            raise ValueError("피어슨 상관계수는 두 그룹 간에만 계산 가능합니다.")
        
        group_data = self.data_processor.get_group_data()
        group1, group2 = self.data_processor.groups
        data1, data2 = group_data[group1], group_data[group2]
        
        # 길이가 다른 경우 더 짧은 길이로 맞춤
        min_length = min(len(data1), len(data2))
        data1 = data1[:min_length]
        data2 = data2[:min_length]
        
        # 피어슨 상관계수 계산
        pearson_r, p_value = stats.pearsonr(data1, data2)
        
        return {
            "pearson_r": pearson_r,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "interpretation": self._interpret_correlation(pearson_r)
        }
    
    def chi_square_test(self, threshold=None) -> Dict[str, Any]:
        """카이제곱 검정 (범주형 데이터인 경우)"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다.")
        
        group_data = self.data_processor.get_group_data()
        
        # 데이터를 이진화하기 위한 임계값 설정
        if threshold is None:
            # 기본적으로 각 그룹의 중앙값 사용
            thresholds = {group: data.median() for group, data in group_data.items()}
        else:
            thresholds = {group: threshold for group in self.data_processor.groups}
        
        # 이진화된 데이터 준비 (임계값 이상 = 1, 미만 = 0)
        binary_data = {}
        for group, data in group_data.items():
            binary_data[group] = (data >= thresholds[group]).astype(int)
        
        # 카이제곱 검정을 위한 분할표(contingency table) 생성
        contingency_table = np.zeros((len(self.data_processor.groups), 2))
        
        for i, group in enumerate(self.data_processor.groups):
            # 각 범주(0, 1)의 개수 계산
            counts = np.bincount(binary_data[group], minlength=2)
            contingency_table[i] = counts
        
        # 카이제곱 검정 수행
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # 오즈비 계산 (2x2 테이블인 경우만)
        odds_ratio = None
        if len(self.data_processor.groups) == 2:
            # 오즈비 = (a*d) / (b*c) 여기서 a, b, c, d는 2x2 테이블의 셀들
            a, b = contingency_table[0]
            c, d = contingency_table[1]
            odds_ratio = (a * d) / (b * c) if b * c != 0 else None
        
        return {
            "chi2": chi2,
            "p_value": p_value,
            "dof": dof,
            "significant": p_value < self.alpha,
            "contingency_table": contingency_table.tolist(),
            "expected": expected.tolist(),
            "odds_ratio": odds_ratio
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """모든 적절한 테스트를 실행하고 결과 종합"""
        # 정규성 및 등분산성 검정
        self.test_normality()
        self.test_homogeneity()
        
        # 가설 검정 유형 결정
        self.determine_test_type()
        
        # 가설 검정 실행
        self.run_hypothesis_test()
        
        # 효과 크기 계산
        self.calculate_effect_size()
        
        # 부트스트랩 분석
        self.perform_bootstrap()
        
        # 오류 분석
        self.analyze_errors()
        
        # 종합 결과 반환
        return {
            "normality": self.normality_results,
            "homogeneity": self.homogeneity_results,
            "test_type": self.test_type.value,
            "hypothesis_test": self.hypothesis_test_results,
            "effect_size": self.effect_size_results,
            "bootstrap": self.bootstrap_results,
            "error_analysis": self.error_analysis
        }
    
    def _interpret_correlation(self, r: float) -> str:
        """상관계수 해석"""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "매우 약한 상관관계"
        elif abs_r < 0.3:
            return "약한 상관관계"
        elif abs_r < 0.5:
            return "중간 정도의 상관관계"
        elif abs_r < 0.7:
            return "강한 상관관계"
        else:
            return "매우 강한 상관관계"
    
    def get_null_alternative_hypothesis(self) -> Dict[str, str]:
        """귀무가설과 대립가설 생성"""
        if not self.data_processor.group_col or not self.data_processor.target_col:
            raise ValueError("그룹 열과 타겟 열이 설정되지 않았습니다.")
        
        num_groups = len(self.data_processor.groups)
        target_col = self.data_processor.target_col
        
        if num_groups == 2:
            group1, group2 = self.data_processor.groups
            null_hypothesis = f"귀무가설(H₀): {group1}과 {group2} 그룹 간 {target_col}의 평균에 차이가 없다."
            alternative_hypothesis = f"대립가설(H₁): {group1}과 {group2} 그룹 간 {target_col}의 평균에 차이가 있다."
        else:
            groups_str = ", ".join(self.data_processor.groups)
            null_hypothesis = f"귀무가설(H₀): 모든 그룹({groups_str}) 간 {target_col}의 평균에 차이가 없다."
            alternative_hypothesis = f"대립가설(H₁): 적어도 한 그룹의 {target_col} 평균이 다른 그룹과 다르다."
        
        return {
            "null": null_hypothesis,
            "alternative": alternative_hypothesis
        }
