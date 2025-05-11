import os
import base64
import smtplib
import pandas as pd
import plotly.graph_objects as go
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import Dict, List, Optional, Any
import jinja2
import tempfile
from datetime import datetime
import pdfkit

class Reporter:
    """A/B 테스트 결과 보고서 생성 및 이메일 전송 클래스"""
    
    def __init__(self, data_processor, statistical_tester, visualizer):
        """
        Args:
            data_processor: DataProcessor 인스턴스
            statistical_tester: StatisticalTester 인스턴스
            visualizer: Visualizer 인스턴스
        """
        self.data_processor = data_processor
        self.statistical_tester = statistical_tester
        self.visualizer = visualizer
        self.report_html = None
    
    def generate_simple_html_report(self) -> str:
        """간소화된 HTML 보고서 생성 (PDF 변환 목적)"""
        if not self.data_processor.groups:
            raise ValueError("그룹 데이터가 설정되지 않았습니다.")
        
        # 필요한 경우 모든 테스트 실행
        if not hasattr(self.statistical_tester, 'hypothesis_test_results') or not self.statistical_tester.hypothesis_test_results:
            self.statistical_tester.run_all_tests()
        
        # 데이터 요약 및 기본 정보
        summary = self.data_processor.get_group_summary()
        hypothesis = self.statistical_tester.get_null_alternative_hypothesis()
        test_results = self.statistical_tester.hypothesis_test_results
        effect_size = self.statistical_tester.effect_size_results
        
        # 현재 날짜
        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        
        # 간소화된 HTML 보고서 템플릿
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>A/B 테스트 결과 보고서</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                h1 {
                    text-align: center;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #eee;
                    margin-bottom: 30px;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    border-left: 5px solid #2196F3;
                }
                .summary {
                    background-color: #e3f2fd;
                    border-left: 5px solid #2196F3;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                table, th, td {
                    border: 1px solid #ddd;
                }
                th, td {
                    padding: 10px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .conclusion {
                    font-weight: bold;
                    margin-top: 20px;
                    padding: 15px;
                    background-color: {% if test_results.significant %}#e8f5e9{% else %}#fff3e0{% endif %};
                    border-radius: 5px;
                    border-left: 5px solid {% if test_results.significant %}#4CAF50{% else %}#FF9800{% endif %};
                }
                .footer {
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #eee;
                    text-align: center;
                    font-size: 0.9em;
                    color: #777;
                }
                .header-info {
                    text-align: right;
                    margin-bottom: 20px;
                    font-size: 0.9em;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="header-info">
                생성일: {{ current_date }}
            </div>
            
            <h1>A/B 테스트 결과 보고서</h1>
            
            <!-- 요약 섹션 -->
            <div class="section summary">
                <h2>결과 요약</h2>
                <p><strong>검정 방법:</strong> {{ test_results.test_name }}</p>
                <p><strong>p-value:</strong> {{ "%.4f"|format(test_results.p_value) }}</p>
                <p><strong>통계적 유의성:</strong> {% if test_results.significant %}있음 (귀무가설 기각){% else %}없음 (귀무가설 채택){% endif %}</p>
                <p><strong>효과 크기 ({{ effect_size.measure }}):</strong> {{ "%.4f"|format(effect_size.value) }}</p>
                <p><strong>효과 크기 해석:</strong> {{ effect_size.interpretation }}</p>
                
                <div class="conclusion">
                    {% if test_results.significant %}
                    <p>귀무가설을 기각하고 대립가설을 지지하는 통계적으로 유의한 증거가 있습니다.</p>
                    <p>측정된 효과 크기는 {{ effect_size.interpretation }} 수준입니다.</p>
                    {% else %}
                    <p>귀무가설을 기각할 만한 통계적으로 유의한 증거가 없습니다.</p>
                    {% endif %}
                </div>
            </div>
            
            <!-- 데이터 개요 섹션 -->
            <div class="section">
                <h2>데이터 개요</h2>
                <p><strong>그룹 컬럼:</strong> {{ group_col }}</p>
                <p><strong>종속 변수:</strong> {{ target_col }}</p>
                <p><strong>그룹 수:</strong> {{ group_count }}</p>
                
                <h3>기초 통계</h3>
                <table>
                    <tr>
                        <th>그룹</th>
                        <th>샘플 수</th>
                        <th>평균</th>
                        <th>표준편차</th>
                        <th>최소값</th>
                        <th>중앙값</th>
                        <th>최대값</th>
                    </tr>
                    {% for group, stats in summary.iterrows() %}
                    <tr>
                        <td>{{ group }}</td>
                        <td>{{ stats['개수']|int }}</td>
                        <td>{{ "%.3f"|format(stats['평균']) }}</td>
                        <td>{{ "%.3f"|format(stats['표준편차']) }}</td>
                        <td>{{ "%.3f"|format(stats['최소값']) }}</td>
                        <td>{{ "%.3f"|format(stats['중앙값']) }}</td>
                        <td>{{ "%.3f"|format(stats['최대값']) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <!-- 가설 설정 섹션 -->
            <div class="section">
                <h2>가설 설정</h2>
                <p><strong>귀무가설 (H₀):</strong> {{ hypothesis.null }}</p>
                <p><strong>대립가설 (H₁):</strong> {{ hypothesis.alternative }}</p>
            </div>
            
            <!-- 검정 결과 섹션 -->
            <div class="section">
                <h2>통계 검정 결과</h2>
                
                <h3>검정 결과</h3>
                <table>
                    <tr>
                        <th>검정 방법</th>
                        <th>통계량</th>
                        <th>p-value</th>
                        <th>유의수준 (α)</th>
                        <th>결과</th>
                    </tr>
                    <tr>
                        <td>{{ test_results.test_name }}</td>
                        <td>
                            {% if "f_statistic" in test_results %}
                                {{ "%.4f"|format(test_results.f_statistic) }}
                            {% else %}
                                {{ "%.4f"|format(test_results.statistic) }}
                            {% endif %}
                        </td>
                        <td>{{ "%.4f"|format(test_results.p_value) }}</td>
                        <td>{{ "%.2f"|format(alpha) }}</td>
                        <td>{{ "귀무가설 기각 (유의함)" if test_results.significant else "귀무가설 채택 (유의하지 않음)" }}</td>
                    </tr>
                </table>
                
                {% if 'post_hoc' in test_results and test_results.post_hoc.results %}
                <h3>사후 검정 결과</h3>
                <table>
                    <tr>
                        <th>그룹1</th>
                        <th>그룹2</th>
                        <th>평균 차이</th>
                        <th>p-value</th>
                        <th>유의성</th>
                    </tr>
                    {% for result in test_results.post_hoc.results %}
                    <tr>
                        <td>{{ result.group1 }}</td>
                        <td>{{ result.group2 }}</td>
                        <td>{% if 'meandiff' in result %}{{ "%.4f"|format(result.meandiff) }}{% else %}{{ "" }}{% endif %}</td>
                        <td>{% if 'p_value' in result %}{{ "%.4f"|format(result.p_value) }}{% else %}{{ "%.4f"|format(result.pvalue) }}{% endif %}</td>
                        <td>{% if 'reject' in result %}{{ "유의함" if result.reject else "유의하지 않음" }}{% else %}{{ "유의함" if result.significant else "유의하지 않음" }}{% endif %}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            
            <!-- 효과 크기 섹션 -->
            <div class="section">
                <h2>효과 크기 분석</h2>
                
                <table>
                    <tr>
                        <th>측정</th>
                        <th>값</th>
                        <th>해석</th>
                        <th>비교</th>
                    </tr>
                    <tr>
                        <td>{{ effect_size.measure }}</td>
                        <td>{{ "%.4f"|format(effect_size.value) }}</td>
                        <td>{{ effect_size.interpretation }}</td>
                        <td>{{ effect_size.comparison }}</td>
                    </tr>
                </table>
            </div>
            
            <!-- 결론 섹션 -->
            <div class="section">
                <h2>결론 및 권장사항</h2>
                <p>유의수준 {{ "%.2f"|format(alpha) }}에서 {{ test_results.test_name }}을(를) 수행한 결과,
                p-value는 {{ "%.4f"|format(test_results.p_value) }}였습니다.</p>
                
                <div class="conclusion">
                    {% if test_results.significant %}
                    <p>귀무가설을 기각하고 대립가설을 지지하는 통계적으로 유의한 증거가 있습니다.</p>
                    <p>측정된 효과 크기({{ effect_size.measure }})는 {{ "%.4f"|format(effect_size.value) }}이며, 이는 {{ effect_size.interpretation }} 수준의 효과입니다.</p>
                    <p><strong>권장사항:</strong> 실험 처치를 확대 적용하는 것이 권장됩니다.</p>
                    {% else %}
                    <p>귀무가설을 기각할 만한 통계적으로 유의한 증거가 없습니다.</p>
                    <p>현재 효과 크기({{ "%.4f"|format(effect_size.value) }})를 고려할 때, 
                    다음과 같은 조치를 고려해볼 수 있습니다:</p>
                    <ul>
                        <li>샘플 크기를 증가시켜 검정력을 높인다</li>
                        <li>다른 실험 설계를 고려한다</li>
                        <li>변수를 더 세밀하게 측정하는 방법을 모색한다</li>
                    </ul>
                    {% endif %}
                </div>
            </div>
            
            <div class="footer">
                <p>이 보고서는 A/B 테스트 결과 통합 대시보드를 통해 {{ current_date }}에 자동 생성되었습니다.</p>
            </div>
        </body>
        </html>
        """
        
        # 템플릿 렌더링
        template = jinja2.Template(template_str)
        try:
            report_html = template.render(
                group_col=self.data_processor.group_col,
                target_col=self.data_processor.target_col,
                group_count=len(self.data_processor.groups),
                summary=summary,
                hypothesis=hypothesis,
                test_results=test_results,
                effect_size=effect_size,
                alpha=self.statistical_tester.alpha,
                current_date=current_date
            )
            self.report_html = report_html
            return report_html
        except Exception as e:
            raise ValueError(f"HTML 생성 중 오류 발생: {str(e)}")
    
    def generate_pdf_report(self, filepath: str = None) -> str:
        """PDF 형식의 보고서 생성 및 저장"""
        # HTML 보고서 생성
        if self.report_html is None:
            self.generate_simple_html_report()
        
        # 파일 경로 지정
        if filepath is None:
            # 임시 파일 생성 (현재 날짜 포함)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ab_test_report_{timestamp}.pdf"
        
        try:
            # HTML을 PDF로 변환 (pdfkit 사용)
            # 옵션 설정
            options = {
                'page-size': 'A4',
                'margin-top': '1cm',
                'margin-right': '1cm',
                'margin-bottom': '1cm',
                'margin-left': '1cm',
                'encoding': 'UTF-8',
                'no-outline': None
            }
            
            # wkhtmltopdf 경로 자동 감지 (기본값)
            pdfkit.from_string(self.report_html, filepath, options=options)
            return filepath
        except Exception as e:
            raise ValueError(f"PDF 생성 중 오류 발생: {str(e)}")
    
    def download_pdf_report(self) -> tuple:
        """PDF 보고서를 다운로드할 수 있는 바이트 데이터와 파일명 반환"""
        # 임시 파일에 PDF 생성
        temp_filepath = self.generate_pdf_report()
        
        # 파일 읽기
        with open(temp_filepath, "rb") as f:
            pdf_bytes = f.read()
        
        # 임시 파일 삭제
        try:
            os.remove(temp_filepath)
        except:
            pass
        
        # 파일명과 바이트 데이터 반환
        filename = os.path.basename(temp_filepath)
        return pdf_bytes, filename
    
    def send_email_with_pdf(self, recipient_email: str, subject: str = None, message: str = None,
                         smtp_server: str = None, smtp_port: int = 587,
                         sender_email: str = None, sender_password: str = None) -> bool:
        """PDF 보고서를 이메일로 전송
        
        Args:
            recipient_email: 수신자 이메일
            subject: 이메일 제목 (기본값: 'A/B 테스트 결과 보고서')
            message: 이메일 본문 (기본값: 간단한 소개 메시지)
            smtp_server: SMTP 서버 주소
            smtp_port: SMTP 포트
            sender_email: 발신자 이메일
            sender_password: 발신자 이메일 비밀번호
            
        Returns:
            bool: 이메일 전송 성공 여부
        """
        if None in [smtp_server, sender_email, sender_password]:
            raise ValueError("SMTP 서버, 발신자 이메일, 비밀번호가 모두 필요합니다.")
        
        # 기본값 설정
        if subject is None:
            subject = 'A/B 테스트 결과 보고서'
        
        if message is None:
            message = '''
            안녕하세요,
            
            첨부된 A/B 테스트 결과 보고서를 확인해주세요.
            
            감사합니다.
            '''
        
        # PDF 보고서 생성
        pdf_filepath = self.generate_pdf_report()
        
        # 이메일 메시지 구성
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # 텍스트 메시지 추가
        msg.attach(MIMEText(message, 'plain'))
        
        # PDF 첨부
        with open(pdf_filepath, 'rb') as f:
            pdf_attachment = MIMEApplication(f.read(), _subtype='pdf')
            pdf_attachment.add_header('Content-Disposition', 'attachment', 
                                     filename=os.path.basename(pdf_filepath))
            msg.attach(pdf_attachment)
        
        try:
            # SMTP 서버 연결 및 로그인
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            
            # 이메일 전송
            server.send_message(msg)
            server.quit()
            
            # 임시 파일 삭제
            try:
                os.remove(pdf_filepath)
            except:
                pass
                
            return True
        except Exception as e:
            print(f"이메일 전송 실패: {str(e)}")
            return False
            
    # 기존 메서드 (하위 호환성 유지)
    def generate_report(self) -> str:
        """기존 HTML 형식의 A/B 테스트 결과 보고서 생성 (하위 호환성 유지)"""
        return self.generate_simple_html_report()
        
    def save_report(self, filepath: str = None) -> str:
        """보고서를 HTML 파일로 저장 (하위 호환성 유지)"""
        if self.report_html is None:
            self.generate_simple_html_report()
        
        if filepath is None:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                filepath = f.name
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.report_html)
        
        return filepath
    
    def download_report(self) -> str:
        """보고서 HTML을 다운로드할 수 있는 링크 생성 (하위 호환성 유지)"""
        if self.report_html is None:
            self.generate_simple_html_report()
        
        # HTML을 base64로 인코딩
        b64 = base64.b64encode(self.report_html.encode()).decode()
        
        # 다운로드 링크 생성
        download_link = f'<a href="data:text/html;base64,{b64}" download="ab_test_report.html">보고서 다운로드</a>'
        return download_link
    
    def send_email(self, recipient_email: str, subject: str = None, message: str = None,
                  smtp_server: str = None, smtp_port: int = 587,
                  sender_email: str = None, sender_password: str = None) -> bool:
        """결과 보고서를 이메일로 전송 (하위 호환성 유지)
        
        Args:
            recipient_email: 수신자 이메일
            subject: 이메일 제목 (기본값: 'A/B 테스트 결과 보고서')
            message: 이메일 본문 (기본값: 간단한 소개 메시지)
            smtp_server: SMTP 서버 주소
            smtp_port: SMTP 포트
            sender_email: 발신자 이메일
            sender_password: 발신자 이메일 비밀번호
            
        Returns:
            bool: 이메일 전송 성공 여부
        """
        if None in [smtp_server, sender_email, sender_password]:
            raise ValueError("SMTP 서버, 발신자 이메일, 비밀번호가 모두 필요합니다.")
        
        if self.report_html is None:
            self.generate_simple_html_report()
        
        # 기본값 설정
        if subject is None:
            subject = 'A/B 테스트 결과 보고서'
        
        if message is None:
            message = '''
            안녕하세요,
            
            첨부된 A/B 테스트 결과 보고서를 확인해주세요.
            
            감사합니다.
            '''
        
        # 이메일 메시지 구성
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # 텍스트 메시지 추가
        msg.attach(MIMEText(message, 'plain'))
        
        # HTML 보고서 첨부
        html_attachment = MIMEText(self.report_html, 'html')
        html_attachment.add_header('Content-Disposition', 'attachment', filename='ab_test_report.html')
        msg.attach(html_attachment)
        
        try:
            # SMTP 서버 연결 및 로그인
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            
            # 이메일 전송
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            print(f"이메일 전송 실패: {str(e)}")
            return False