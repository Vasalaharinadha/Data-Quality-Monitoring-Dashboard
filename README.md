# Data Quality Monitoring Dashboard

An automated data quality monitoring system for B2B sales intelligence databases. Tracks quality metrics, detects anomalies, and generates actionable insights.

## 🎯 Overview

This monitoring system evaluates B2B contact data across six critical quality dimensions, providing comprehensive visibility into data health with automated anomaly detection and remediation recommendations.

## ✨ Key Features

### Six Quality Dimensions

#### 1. **Completeness**
- Field-level null analysis
- Critical field identification
- Overall completeness scoring
- Missing data patterns

#### 2. **Accuracy**
- Email format validation with regex
- Phone number validation and standardization
- Data type consistency checks
- Format compliance verification

#### 3. **Consistency**
- Case standardization checks
- Format uniformity validation
- Company name variation detection
- Naming convention compliance

#### 4. **Uniqueness**
- Duplicate record detection
- Field-level uniqueness analysis
- Exact row duplicate identification
- Multi-field deduplication strategies

#### 5. **Timeliness**
- Data freshness analysis
- Age distribution tracking
- Stale data identification (>180 days)
- Temporal trend analysis

#### 6. **Validity**
- Business rule compliance
- Business email verification (vs. personal)
- Decision-maker role validation
- Domain-specific rule enforcement

### Advanced Features

- **Anomaly Detection**: Statistical analysis to identify data quality issues
- **Quality Scoring**: Weighted scoring system (0-100) across all dimensions
- **Automated Recommendations**: Actionable insights for data improvement
- **Comprehensive Reporting**: Executive-level and detailed technical reports
- **Trend Analysis**: Historical quality tracking capabilities

## 🚀 Installation

```bash
git clone https://github.com/yourusername/data-quality-dashboard.git
cd data-quality-dashboard

pip install -r requirements.txt
```

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
```

## 💻 Usage

### Quick Start

```python
from quality_monitor import DataQualityMonitor
import pandas as pd

# Load your data
df = pd.read_csv('contact_data.csv')

# Initialize monitor
monitor = DataQualityMonitor()

# Generate comprehensive report
report = monitor.create_summary_report(df)

# Print formatted report
monitor.print_report(report)

# Export to CSV
report_df = monitor.export_report_to_dataframe(report)
report_df.to_csv('quality_report.csv', index=False)
```

### Individual Quality Checks

```python
# Check specific dimensions
completeness = monitor.check_completeness(df)
accuracy = monitor.check_accuracy(df)
consistency = monitor.check_consistency(df)
uniqueness = monitor.check_uniqueness(df, ['email', 'phone'])
timeliness = monitor.check_timeliness(df, 'created_at')
validity = monitor.check_validity(df)

# Detect anomalies
anomalies = monitor.detect_anomalies(df, threshold=2.0)
```

## 📊 Sample Report Output

```
======================================================================
DATA QUALITY MONITORING REPORT
======================================================================

REPORT TIMESTAMP: 2025-02-15 14:30:00
TOTAL RECORDS: 5,000
TOTAL FIELDS: 8

OVERALL QUALITY SCORE              82.5/100

----------------------------------------------------------------------
QUALITY DIMENSIONS
----------------------------------------------------------------------
Completeness                       87.3%
Email Accuracy                     94.2%

----------------------------------------------------------------------
ANOMALIES DETECTED: 3
----------------------------------------------------------------------
1. [HIGH] company_name has 15.20% null values
2. [MEDIUM] email has only 78.50% unique values
3. [HIGH] 22.00% of records are older than 180 days

----------------------------------------------------------------------
RECOMMENDATIONS
----------------------------------------------------------------------
1. CRITICAL: Fill missing company_name data (15.2% incomplete)
2. Remove duplicate email entries (1,075 duplicates found)
3. URGENT: 22.00% of records are older than 180 days
4. Improve email validation (currently 94.2% accurate)
5. Remove duplicate phone entries (432 duplicates found)

======================================================================
```

## 📈 Quality Scoring System

### Overall Score Calculation

**Weighted Components:**
- Completeness: 30%
- Accuracy: 30%
- Uniqueness: 20%
- Validity: 20%

**Score Interpretation:**
- 90-100: Excellent
- 80-89: Good
- 70-79: Fair
- 60-69: Poor
- <60: Critical

## 🔍 Quality Dimensions Explained

### Completeness Metrics
```python
{
    'overall_completeness_pct': 87.3,
    'total_records': 5000,
    'total_fields': 8,
    'by_column': {
        'email': {
            'non_null_count': 4850,
            'null_count': 150,
            'completeness_pct': 97.0
        }
    },
    'critical_missing': [
        {'field': 'company_name', 'missing_pct': 15.2}
    ]
}
```

### Accuracy Metrics
```python
{
    'email_validation': {
        'total': 5000,
        'valid': 4710,
        'invalid': 140,
        'accuracy_pct': 94.2
    },
    'phone_validation': {
        'total': 5000,
        'valid': 4250,
        'invalid': 320,
        'accuracy_pct': 85.0
    }
}
```

## 🎯 Use Cases

### Sales Operations
- **CRM Health Monitoring**: Track data quality trends over time
- **Data Migration Validation**: Ensure quality during system transitions
- **Lead Quality Assessment**: Score leads based on data completeness

### Data Engineering
- **Pipeline Monitoring**: Automated quality checks in ETL workflows
- **Data Cleansing Prioritization**: Identify high-impact quality issues
- **SLA Compliance**: Verify data quality meets service level agreements

### GTM Operations
- **Campaign Readiness**: Validate contact lists before outreach
- **Segmentation Quality**: Ensure accurate targeting criteria
- **ROI Tracking**: Correlate data quality with campaign performance

## 🔧 Customization

### Custom Business Rules

```python
# Add custom validation rules
def check_custom_rules(df):
    """Add your domain-specific validation"""
    
    # Example: Company must be in specific industries
    valid_industries = ['Technology', 'SaaS', 'FinTech']
    if 'industry' in df.columns:
        invalid = ~df['industry'].isin(valid_industries)
        return {
            'rule': 'valid_industry',
            'violations': invalid.sum()
        }
```

### Anomaly Detection Thresholds

```python
# Adjust sensitivity
anomalies = monitor.detect_anomalies(
    df,
    threshold=2.5  # Higher = less sensitive
)
```

### Quality Score Weights

Modify weights in `generate_quality_score()`:
```python
scores = [
    ('completeness', completeness_pct, 0.30),  # Adjust weight
    ('accuracy', accuracy_pct, 0.30),
    ('uniqueness', uniqueness_score, 0.20),
    ('validity', validity_score, 0.20)
]
```

## 📊 Export Formats

### CSV Export
```python
report_df = monitor.export_report_to_dataframe(report)
report_df.to_csv('quality_summary.csv', index=False)
```

### JSON Export
```python
import json
with open('quality_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
```

## 🔄 Integration Examples

### ETL Pipeline Integration
```python
from quality_monitor import DataQualityMonitor

def etl_with_quality_check(input_file, output_file):
    # Load data
    df = pd.read_csv(input_file)
    
    # Run quality check
    monitor = DataQualityMonitor()
    report = monitor.create_summary_report(df)
    
    # Fail pipeline if quality too low
    if report['summary']['quality_score'] < 70:
        raise ValueError(f"Data quality too low: {report['summary']['quality_score']}")
    
    # Proceed with processing
    df.to_csv(output_file)
```

### Scheduled Monitoring
```python
import schedule
import time

def daily_quality_check():
    df = pd.read_csv('production_data.csv')
    monitor = DataQualityMonitor()
    report = monitor.create_summary_report(df)
    monitor.print_report(report)

# Run every day at 9 AM
schedule.every().day.at("09:00").do(daily_quality_check)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

## 🎓 Technical Details

### Architecture
- **Modular Design**: Each quality dimension is independently testable
- **Pandas-Optimized**: Efficient operations on large datasets
- **Extensible**: Easy to add new quality dimensions or rules
- **Production-Ready**: Comprehensive logging and error handling

### Performance
- Handles datasets up to 10M+ records
- Parallel processing-ready architecture
- Memory-efficient operations
- Incremental checking support

## 📈 Metrics Tracked

- Total records processed
- Field-level completeness percentages
- Validation pass/fail rates
- Duplicate counts and percentages
- Data age distributions
- Anomaly counts by severity
- Quality score trends over time

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional quality dimensions
- More validation patterns
- Visualization dashboards
- Alerting integrations
- ML-based anomaly detection

