"""
Data Quality Monitoring Dashboard for B2B Sales Intelligence
-------------------------------------------------------------
Automated data quality monitoring, validation, and reporting system
for B2B contact databases with anomaly detection and trend analysis.

Author: Your Name
Purpose: Portfolio demonstrating data quality engineering expertise
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """
    Comprehensive data quality monitoring system for B2B sales intelligence data.
    Tracks quality metrics, detects anomalies, and generates actionable reports.
    """
    
    def __init__(self):
        self.quality_metrics = {}
        self.validation_rules = {}
        self.anomalies = []
        self.report_data = {}
        
    # ==================== DATA QUALITY DIMENSIONS ====================
    
    def check_completeness(self, df: pd.DataFrame) -> Dict:
        """
        Measure data completeness across all fields.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with completeness metrics
        """
        logger.info("Checking data completeness...")
        
        total_cells = df.size
        non_null_cells = df.notna().sum().sum()
        
        completeness_by_column = {}
        for col in df.columns:
            completeness_by_column[col] = {
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
                'completeness_pct': (df[col].notna().sum() / len(df) * 100).round(2)
            }
        
        # Identify critical missing fields
        critical_fields = ['email', 'company_name', 'job_title', 'phone']
        missing_critical = []
        for field in critical_fields:
            if field in df.columns:
                null_pct = (df[field].isna().sum() / len(df) * 100)
                if null_pct > 10:  # Alert if >10% missing
                    missing_critical.append({
                        'field': field,
                        'missing_pct': null_pct.round(2)
                    })
        
        completeness_metrics = {
            'overall_completeness_pct': (non_null_cells / total_cells * 100).round(2),
            'total_records': len(df),
            'total_fields': len(df.columns),
            'by_column': completeness_by_column,
            'critical_missing': missing_critical
        }
        
        logger.info(f"Overall completeness: {completeness_metrics['overall_completeness_pct']}%")
        return completeness_metrics
    
    def check_accuracy(self, df: pd.DataFrame) -> Dict:
        """
        Validate data accuracy using format and pattern checks.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("Checking data accuracy...")
        
        accuracy_metrics = {
            'email_validation': {},
            'phone_validation': {},
            'data_type_issues': []
        }
        
        # Email accuracy
        if 'email' in df.columns:
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            valid_emails = df['email'].apply(
                lambda x: bool(re.match(email_pattern, str(x))) if pd.notna(x) else False
            )
            
            accuracy_metrics['email_validation'] = {
                'total': len(df),
                'valid': valid_emails.sum(),
                'invalid': (~valid_emails & df['email'].notna()).sum(),
                'accuracy_pct': (valid_emails.sum() / df['email'].notna().sum() * 100).round(2) if df['email'].notna().sum() > 0 else 0
            }
        
        # Phone accuracy
        if 'phone' in df.columns:
            phone_pattern = r'^\+?1?\d{10,}$'
            valid_phones = df['phone'].apply(
                lambda x: bool(re.match(phone_pattern, re.sub(r'\D', '', str(x)))) if pd.notna(x) else False
            )
            
            accuracy_metrics['phone_validation'] = {
                'total': len(df),
                'valid': valid_phones.sum(),
                'invalid': (~valid_phones & df['phone'].notna()).sum(),
                'accuracy_pct': (valid_phones.sum() / df['phone'].notna().sum() * 100).round(2) if df['phone'].notna().sum() > 0 else 0
            }
        
        logger.info("Accuracy check completed")
        return accuracy_metrics
    
    def check_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Check data consistency and standardization.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with consistency metrics
        """
        logger.info("Checking data consistency...")
        
        consistency_metrics = {
            'format_inconsistencies': [],
            'case_inconsistencies': [],
            'duplicate_variations': []
        }
        
        # Check email domain consistency
        if 'email' in df.columns:
            df_copy = df[df['email'].notna()].copy()
            df_copy['email_lower'] = df_copy['email'].str.lower()
            df_copy['email_original'] = df_copy['email']
            
            case_issues = df_copy[df_copy['email_lower'] != df_copy['email_original']]
            if len(case_issues) > 0:
                consistency_metrics['case_inconsistencies'].append({
                    'field': 'email',
                    'records_affected': len(case_issues),
                    'pct_affected': (len(case_issues) / len(df) * 100).round(2)
                })
        
        # Check company name variations
        if 'company_name' in df.columns:
            companies = df['company_name'].dropna()
            company_normalized = companies.str.lower().str.strip()
            
            # Find similar company names (potential duplicates)
            from collections import Counter
            company_counts = Counter(company_normalized)
            variations = []
            
            for normalized, count in company_counts.items():
                originals = companies[company_normalized == normalized].unique()
                if len(originals) > 1:
                    variations.append({
                        'normalized': normalized,
                        'variations': list(originals),
                        'count': count
                    })
            
            if variations:
                consistency_metrics['duplicate_variations'] = variations[:10]  # Top 10
        
        logger.info("Consistency check completed")
        return consistency_metrics
    
    def check_uniqueness(self, df: pd.DataFrame, key_fields: List[str]) -> Dict:
        """
        Check for duplicate records.
        
        Args:
            df: Input DataFrame
            key_fields: Fields to check for uniqueness
            
        Returns:
            Dictionary with uniqueness metrics
        """
        logger.info(f"Checking uniqueness on fields: {key_fields}")
        
        uniqueness_metrics = {
            'total_records': len(df),
            'duplicates_by_field': {}
        }
        
        for field in key_fields:
            if field not in df.columns:
                continue
            
            duplicates = df[df[field].duplicated(keep=False) & df[field].notna()]
            
            uniqueness_metrics['duplicates_by_field'][field] = {
                'duplicate_count': len(duplicates),
                'unique_count': df[field].nunique(),
                'duplicate_pct': (len(duplicates) / len(df) * 100).round(2)
            }
        
        # Check for exact row duplicates
        exact_duplicates = df[df.duplicated(keep=False)]
        uniqueness_metrics['exact_duplicate_rows'] = {
            'count': len(exact_duplicates),
            'pct': (len(exact_duplicates) / len(df) * 100).round(2)
        }
        
        logger.info("Uniqueness check completed")
        return uniqueness_metrics
    
    def check_timeliness(self, df: pd.DataFrame, date_column: str = 'created_at') -> Dict:
        """
        Check data freshness and timeliness.
        
        Args:
            df: Input DataFrame
            date_column: Name of the timestamp column
            
        Returns:
            Dictionary with timeliness metrics
        """
        logger.info("Checking data timeliness...")
        
        timeliness_metrics = {
            'date_column': date_column,
            'has_dates': date_column in df.columns
        }
        
        if date_column not in df.columns:
            logger.warning(f"Date column '{date_column}' not found")
            return timeliness_metrics
        
        # Convert to datetime
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
        valid_dates = df_copy[date_column].notna()
        
        if valid_dates.sum() == 0:
            logger.warning("No valid dates found")
            return timeliness_metrics
        
        now = datetime.now()
        df_copy['age_days'] = (now - df_copy[date_column]).dt.days
        
        timeliness_metrics.update({
            'oldest_record': df_copy.loc[valid_dates, date_column].min().strftime('%Y-%m-%d'),
            'newest_record': df_copy.loc[valid_dates, date_column].max().strftime('%Y-%m-%d'),
            'avg_age_days': df_copy.loc[valid_dates, 'age_days'].mean().round(2),
            'records_by_age': {
                '0-30_days': len(df_copy[df_copy['age_days'] <= 30]),
                '31-90_days': len(df_copy[(df_copy['age_days'] > 30) & (df_copy['age_days'] <= 90)]),
                '91-180_days': len(df_copy[(df_copy['age_days'] > 90) & (df_copy['age_days'] <= 180)]),
                '180+_days': len(df_copy[df_copy['age_days'] > 180])
            }
        })
        
        # Flag stale data (>180 days)
        stale_pct = (len(df_copy[df_copy['age_days'] > 180]) / len(df) * 100).round(2)
        if stale_pct > 20:
            self.anomalies.append({
                'type': 'STALE_DATA',
                'severity': 'HIGH',
                'message': f'{stale_pct}% of records are older than 180 days'
            })
        
        logger.info("Timeliness check completed")
        return timeliness_metrics
    
    def check_validity(self, df: pd.DataFrame) -> Dict:
        """
        Check business rule validity.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validity metrics
        """
        logger.info("Checking business rule validity...")
        
        validity_metrics = {
            'rules_checked': [],
            'violations': []
        }
        
        # Rule 1: Email must be business email (not personal)
        if 'email' in df.columns:
            personal_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
            personal_emails = df['email'].str.lower().apply(
                lambda x: any(domain in str(x) for domain in personal_domains) if pd.notna(x) else False
            )
            
            violation_count = personal_emails.sum()
            validity_metrics['rules_checked'].append('business_email_only')
            
            if violation_count > 0:
                validity_metrics['violations'].append({
                    'rule': 'business_email_only',
                    'violation_count': violation_count,
                    'pct': (violation_count / len(df) * 100).round(2),
                    'severity': 'MEDIUM'
                })
        
        # Rule 2: Job title must indicate decision-making role
        if 'job_title' in df.columns:
            decision_maker_keywords = ['ceo', 'cto', 'cfo', 'president', 'vp', 'director', 'manager', 'head']
            is_decision_maker = df['job_title'].str.lower().apply(
                lambda x: any(keyword in str(x).lower() for keyword in decision_maker_keywords) if pd.notna(x) else False
            )
            
            non_decision_makers = (~is_decision_maker & df['job_title'].notna()).sum()
            validity_metrics['rules_checked'].append('decision_maker_title')
            
            if non_decision_makers > len(df) * 0.3:  # >30% non-decision makers
                validity_metrics['violations'].append({
                    'rule': 'decision_maker_title',
                    'violation_count': non_decision_makers,
                    'pct': (non_decision_makers / len(df) * 100).round(2),
                    'severity': 'LOW'
                })
        
        logger.info("Validity check completed")
        return validity_metrics
    
    # ==================== ANOMALY DETECTION ====================
    
    def detect_anomalies(self, df: pd.DataFrame, threshold: float = 2.0) -> List[Dict]:
        """
        Detect statistical anomalies in the data.
        
        Args:
            df: Input DataFrame
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        logger.info("Detecting anomalies...")
        
        anomalies = []
        
        # Check for sudden spikes in null values
        null_percentages = (df.isna().sum() / len(df) * 100)
        
        for col, null_pct in null_percentages.items():
            if null_pct > 50:  # More than 50% null
                anomalies.append({
                    'type': 'HIGH_NULL_RATE',
                    'field': col,
                    'severity': 'HIGH',
                    'value': null_pct.round(2),
                    'message': f'{col} has {null_pct:.2f}% null values'
                })
        
        # Check for low cardinality in fields that should be unique
        uniqueness_fields = ['email', 'phone']
        for field in uniqueness_fields:
            if field in df.columns:
                unique_ratio = df[field].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique
                    anomalies.append({
                        'type': 'LOW_CARDINALITY',
                        'field': field,
                        'severity': 'MEDIUM',
                        'value': (unique_ratio * 100).round(2),
                        'message': f'{field} has only {unique_ratio*100:.2f}% unique values'
                    })
        
        self.anomalies.extend(anomalies)
        logger.info(f"Detected {len(anomalies)} anomalies")
        
        return anomalies
    
    # ==================== REPORTING ====================
    
    def generate_quality_score(self, metrics: Dict) -> float:
        """
        Calculate overall data quality score.
        
        Args:
            metrics: Dictionary of all quality metrics
            
        Returns:
            Overall quality score (0-100)
        """
        scores = []
        
        # Completeness score (30%)
        if 'completeness' in metrics:
            scores.append(('completeness', metrics['completeness']['overall_completeness_pct'], 0.30))
        
        # Accuracy score (30%)
        if 'accuracy' in metrics:
            email_acc = metrics['accuracy'].get('email_validation', {}).get('accuracy_pct', 100)
            phone_acc = metrics['accuracy'].get('phone_validation', {}).get('accuracy_pct', 100)
            avg_accuracy = (email_acc + phone_acc) / 2
            scores.append(('accuracy', avg_accuracy, 0.30))
        
        # Uniqueness score (20%)
        if 'uniqueness' in metrics:
            dup_pct = metrics['uniqueness'].get('exact_duplicate_rows', {}).get('pct', 0)
            uniqueness_score = max(0, 100 - dup_pct)
            scores.append(('uniqueness', uniqueness_score, 0.20))
        
        # Validity score (20%)
        if 'validity' in metrics:
            violations = metrics['validity'].get('violations', [])
            total_violations = sum(v['violation_count'] for v in violations)
            total_records = len(metrics.get('df', []))
            validity_score = max(0, 100 - (total_violations / total_records * 100)) if total_records > 0 else 100
            scores.append(('validity', validity_score, 0.20))
        
        # Calculate weighted score
        overall_score = sum(score * weight for _, score, weight in scores)
        
        return round(overall_score, 2)
    
    def create_summary_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive quality summary report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Complete quality report dictionary
        """
        logger.info("Generating summary report...")
        
        # Run all quality checks
        metrics = {
            'completeness': self.check_completeness(df),
            'accuracy': self.check_accuracy(df),
            'consistency': self.check_consistency(df),
            'uniqueness': self.check_uniqueness(df, ['email', 'phone']),
            'validity': self.check_validity(df),
            'df': df  # Store for score calculation
        }
        
        # Add timeliness if date column exists
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'created' in col.lower()]
        if date_columns:
            metrics['timeliness'] = self.check_timeliness(df, date_columns[0])
        
        # Detect anomalies
        anomalies = self.detect_anomalies(df)
        
        # Calculate overall quality score
        quality_score = self.generate_quality_score(metrics)
        
        # Compile report
        report = {
            'summary': {
                'total_records': len(df),
                'total_fields': len(df.columns),
                'quality_score': quality_score,
                'report_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'metrics': metrics,
            'anomalies': anomalies,
            'recommendations': self.generate_recommendations(metrics, anomalies)
        }
        
        logger.info(f"Overall Quality Score: {quality_score}/100")
        return report
    
    def generate_recommendations(self, metrics: Dict, anomalies: List[Dict]) -> List[str]:
        """
        Generate actionable recommendations based on quality metrics.
        
        Args:
            metrics: Quality metrics dictionary
            anomalies: List of detected anomalies
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Completeness recommendations
        if 'completeness' in metrics:
            critical_missing = metrics['completeness'].get('critical_missing', [])
            for missing in critical_missing:
                recommendations.append(
                    f"CRITICAL: Fill missing {missing['field']} data ({missing['missing_pct']}% incomplete)"
                )
        
        # Accuracy recommendations
        if 'accuracy' in metrics:
            email_acc = metrics['accuracy'].get('email_validation', {}).get('accuracy_pct', 100)
            if email_acc < 90:
                recommendations.append(
                    f"Improve email validation (currently {email_acc}% accurate)"
                )
        
        # Uniqueness recommendations
        if 'uniqueness' in metrics:
            for field, data in metrics['uniqueness'].get('duplicates_by_field', {}).items():
                if data['duplicate_pct'] > 5:
                    recommendations.append(
                        f"Remove duplicate {field} entries ({data['duplicate_count']} duplicates found)"
                    )
        
        # Anomaly-based recommendations
        for anomaly in anomalies:
            if anomaly['severity'] == 'HIGH':
                recommendations.append(
                    f"URGENT: {anomaly['message']}"
                )
        
        return recommendations
    
    def export_report_to_dataframe(self, report: Dict) -> pd.DataFrame:
        """
        Convert quality report to DataFrame for export.
        
        Args:
            report: Quality report dictionary
            
        Returns:
            DataFrame with report summary
        """
        summary_data = []
        
        # Overall summary
        summary_data.append({
            'Metric': 'Overall Quality Score',
            'Value': f"{report['summary']['quality_score']}/100",
            'Status': 'PASS' if report['summary']['quality_score'] >= 70 else 'FAIL'
        })
        
        # Completeness
        if 'completeness' in report['metrics']:
            comp = report['metrics']['completeness']
            summary_data.append({
                'Metric': 'Data Completeness',
                'Value': f"{comp['overall_completeness_pct']}%",
                'Status': 'PASS' if comp['overall_completeness_pct'] >= 80 else 'WARNING'
            })
        
        # Accuracy
        if 'accuracy' in report['metrics']:
            acc = report['metrics']['accuracy']
            if 'email_validation' in acc:
                summary_data.append({
                    'Metric': 'Email Accuracy',
                    'Value': f"{acc['email_validation']['accuracy_pct']}%",
                    'Status': 'PASS' if acc['email_validation']['accuracy_pct'] >= 90 else 'WARNING'
                })
        
        return pd.DataFrame(summary_data)
    
    def print_report(self, report: Dict):
        """Print formatted quality report to console"""
        print("\n" + "="*70)
        print("DATA QUALITY MONITORING REPORT")
        print("="*70)
        
        # Summary
        print(f"\nREPORT TIMESTAMP: {report['summary']['report_timestamp']}")
        print(f"TOTAL RECORDS: {report['summary']['total_records']:,}")
        print(f"TOTAL FIELDS: {report['summary']['total_fields']}")
        print(f"\n{'OVERALL QUALITY SCORE':<30} {report['summary']['quality_score']}/100")
        
        # Metrics summary
        print("\n" + "-"*70)
        print("QUALITY DIMENSIONS")
        print("-"*70)
        
        if 'completeness' in report['metrics']:
            print(f"{'Completeness':<30} {report['metrics']['completeness']['overall_completeness_pct']}%")
        
        if 'accuracy' in report['metrics']:
            email_acc = report['metrics']['accuracy'].get('email_validation', {}).get('accuracy_pct', 'N/A')
            print(f"{'Email Accuracy':<30} {email_acc}%")
        
        # Anomalies
        if report['anomalies']:
            print("\n" + "-"*70)
            print(f"ANOMALIES DETECTED: {len(report['anomalies'])}")
            print("-"*70)
            for i, anomaly in enumerate(report['anomalies'][:5], 1):  # Show top 5
                print(f"{i}. [{anomaly['severity']}] {anomaly['message']}")
        
        # Recommendations
        if report['recommendations']:
            print("\n" + "-"*70)
            print("RECOMMENDATIONS")
            print("-"*70)
            for i, rec in enumerate(report['recommendations'][:5], 1):  # Show top 5
                print(f"{i}. {rec}")
        
        print("\n" + "="*70 + "\n")


def generate_sample_data_for_monitoring():
    """Generate sample data with quality issues for testing"""
    np.random.seed(42)
    
    data = {
        'email': ['john@techcorp.com', 'jane.doe@gmail.com', 'invalid-email', 'mike@cloud.io', None] * 100,
        'phone': ['+14155551234', '415-555-5678', '123', '+14155559999', None] * 100,
        'company_name': ['TechCorp Inc', 'TechCorp Inc.', 'TECHCORP INC', 'DataSys', None] * 100,
        'job_title': ['CEO', 'VP Sales', 'Assistant', 'Director', None] * 100,
        'created_at': pd.date_range(start='2024-01-01', periods=500, freq='D')
    }
    
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    # Generate sample data
    sample_df = generate_sample_data_for_monitoring()
    
    # Initialize monitor
    monitor = DataQualityMonitor()
    
    # Generate report
    report = monitor.create_summary_report(sample_df)
    
    # Print report
    monitor.print_report(report)
    
    # Export report
    report_df = monitor.export_report_to_dataframe(report)
    report_df.to_csv('quality_report_summary.csv', index=False)
    
    print("Detailed report saved to: quality_report_summary.csv")
