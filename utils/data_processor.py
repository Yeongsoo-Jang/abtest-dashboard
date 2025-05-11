import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import io
from scipy import stats

class DataProcessor:
    """데이터 로딩 및 전처리를 위한 클래스"""
    
    def __init__(self):
        self.data = None
        self.group_col = None
        self.target_col = None
        self.groups = None
        
    def load_data(self, file_obj) -> pd.DataFrame:
        """CSV 파일을 로드하고 기본 검증 수행"""
        try:
            # 파일 객체가 StringIO이거나 BytesIO일 경우 처리
            if isinstance(file_obj, (io.StringIO, io.BytesIO)):
                self.data = pd.read_csv(file_obj)
            else:
                self.data = pd.read_csv(file_obj)
                
            # 기본 검증: 비어있는 데이터프레임인지 확인
            if self.data.empty:
                raise ValueError("업로드된 CSV 파일이 비어 있습니다.")
                
            return self.data
        except Exception as e:
            raise ValueError(f"CSV 파일 로딩 중 오류 발생: {str(e)}")
    
    def validate_data(self) -> Tuple[bool, str]:
        """데이터의 유효성 검증"""
        if self.data is None:
            return False, "데이터가 로드되지 않았습니다."
        
        # 최소 행 수 체크
        if len(self.data) < 10:
            return False, "데이터 행 수가 너무 적습니다. 최소 10개 이상의 데이터가 필요합니다."
        
        # 결측치 확인
        missing_values = self.data.isnull().sum().sum()
        if missing_values > 0:
            return False, f"데이터에 결측치가 {missing_values}개 있습니다. 전처리 후 다시 시도해주세요."
        
        return True, "데이터 검증 완료"
    
    def get_column_types(self) -> Dict[str, List[str]]:
        """데이터프레임의 각 열 타입 반환"""
        if self.data is None:
            return {"numeric": [], "categorical": []}
        
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols
        }
    
    def set_group_and_target(self, group_col: str, target_col: str) -> None:
        """그룹 열과 타겟 열 설정"""
        if self.data is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
            
        if group_col not in self.data.columns:
            raise ValueError(f"'{group_col}' 열이 데이터에 존재하지 않습니다.")
            
        if target_col not in self.data.columns:
            raise ValueError(f"'{target_col}' 열이 데이터에 존재하지 않습니다.")
        
        self.group_col = group_col
        self.target_col = target_col
        self.groups = sorted(self.data[group_col].unique())
        
        # 데이터 유형 확인
        if self.data[target_col].dtype not in [np.float64, np.int64, np.float32, np.int32]:
            raise ValueError(f"'{target_col}' 열은 수치형 데이터여야 합니다.")
    
    def get_group_data(self, group_name: str = None) -> Union[pd.Series, Dict[str, pd.Series]]:
        """특정 그룹 또는 모든 그룹의 타겟 데이터 반환"""
        if self.data is None or self.group_col is None or self.target_col is None:
            raise ValueError("데이터, 그룹 열, 타겟 열이 모두 설정되어야 합니다.")
        
        if group_name:
            return self.data[self.data[self.group_col] == group_name][self.target_col]
        else:
            return {group: self.data[self.data[self.group_col] == group][self.target_col] 
                    for group in self.groups}
    
    def get_group_summary(self) -> pd.DataFrame:
        """각 그룹별 기본 통계 요약"""
        if self.data is None or self.group_col is None or self.target_col is None:
            raise ValueError("데이터, 그룹 열, 타겟 열이 모두 설정되어야 합니다.")
        
        # scipy.stats를 사용하여 첨도(kurtosis) 계산
        summary = self.data.groupby(self.group_col)[self.target_col].agg([
            'count',            # 샘플 수
            'mean',             # 평균
            'std',              # 표준편차 
            'min',              # 최소값
            'max',              # 최대값
            lambda x: x.quantile(0.25),  # 1사분위수
            lambda x: x.quantile(0.5),   # 중앙값
            lambda x: x.quantile(0.75),  # 3사분위수
            'sem',              # 표준오차
            'var',              # 분산
            lambda x: stats.skew(x),     # 왜도 (scipy 사용)
            lambda x: stats.kurtosis(x, fisher=True)  # 첨도 (scipy 사용)
        ])
        
        # 열 이름 변경
        summary.columns = [
            '개수', '평균', '표준편차', '최소값', '최대값', 
            '1사분위수', '중앙값', '3사분위수', '표준오차', '분산', '왜도', '첨도'
        ]
        
        return summary
    
    def prepare_data_guide(self) -> str:
        """A/B 테스트 데이터 업로드 가이드 제공"""
        guide = """
        ## A/B 테스트 데이터 업로드 가이드
        
        올바른 A/B 테스트 분석을 위해 다음 형식의 CSV 파일을 준비해주세요:
        
        1. **필수 열**:
           - `group`: 실험 그룹을 나타내는 열 (예: 'A', 'B', 'C' 등)
           - 종속변수: 측정하려는 지표 (수치형 데이터여야 함)
           
        2. **권장 사항**:
           - 헤더(열 이름)가 있어야 합니다.
           - 결측치가 없어야 합니다.
           - 충분한 샘플 수를 확보해야 합니다(그룹당 최소 30개 이상 권장).
           
        3. **예시 데이터**:
        
        | group | conversion_rate | time_spent | revenue |
        |-------|-----------------|------------|---------|
        | A     | 0.12            | 120        | 45.5    |
        | A     | 0.08            | 105        | 38.2    |
        | B     | 0.15            | 95         | 52.1    |
        | B     | 0.11            | 85         | 49.3    |
        | ...   | ...             | ...        | ...     |
        
        4. **참고사항**:
           - 그룹 열과 종속변수 열은 업로드 후 선택할 수 있습니다.
           - 여러 종속변수에 대해 분석하려면 각 변수를 개별 열로 추가하세요.
        """
        return guide
