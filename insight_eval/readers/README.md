# Problem and Solution Directory Structure Guide

This guide provides the complete specification for creating new problems and solutions for the Insight Discovery Evaluation Framework. Whether you're contributing to the benchmark curriculum or creating custom evaluation datasets, this documentation will help you structure your data correctly.

## Overview

Each problem in the framework consists of two main components:
1. **Problem Definition**: The prediction task with training/test data and metadata
2. **Ground Truth Solution**: Expert-defined features that represent the correct insights for this problem

## Directory Structure Specification

Each problem directory must contain exactly 2 subdirectories following this structure:

### Complete Directory Layout
```shell
problem_name/
├── problem/
│   ├── problem.json                    # Problem metadata and configuration
│   └── data/
│       ├── train.csv                   # Training dataset
│       ├── test.csv                    # Test dataset
│       └── [auxiliary_tables].csv      # Optional: Additional data tables
└── ground_truth/
    ├── solution.json                   # Ground truth feature definitions
    └── data/
        ├── enriched_train.csv          # Training data with ground truth features
        └── enriched_test.csv           # Test data with ground truth features
```

### Example Structure
```shell
customer_churn_prediction/
├── problem/
│   ├── problem.json
│   └── data/
│       ├── train.csv
│       ├── test.csv
│       ├── orders_table.csv
│       ├── product_info_table.csv
│       └── products_in_orders_table.csv
└── ground_truth/
    ├── solution.json
    └── data/
        ├── enriched_train.csv
        └── enriched_test.csv
```

## Data Schema Specifications

### Problem Definition Schema (`problem.json`)

The problem metadata file must contain the following fields:

```json
{
    "target_column": "churn_flag",
    "description": "Identify customers likely to leave service based on usage patterns and engagement metrics",
    "name": "Customer Churn Prediction",
    "problem_domain": "Telecommunications",
    "comments": "target_comments=Binary classification; data_comments=6 months of customer data; schema_comments=Includes auxiliary usage tables"
}
```

**Field Descriptions:**

- **`target_column`**: The column name in your `train.csv` and `test.csv` that contains the desired target
- **`description`**: Clear, detailed explanation of the business problem being solved - this helps frame the evaluation context
- **`name`**: Human-readable name for the problem that will be used in reports and displays
- **`problem_domain`**: Industry or domain context (e.g., "Retail", "Healthcare", "Finance") that helps evaluators understand the business context
- **`comments`**: Optional field for additional context, notes, or data collection details

### Ground Truth Schema (`solution.json`)

The ground truth definition file specifies the expert-defined features that represent the correct insights:

```json
{
    "enriched_column_names": [
        "decrease_in_average_weekly_data_usage_over_the_past_quarter",
        "frequency_of_customer_complaints_within_the_last_year",
        "days_since_last_customer_service_interaction"
    ],
    "features_descriptions": [
        "Decrease in average weekly data usage over the past quarter (normalized)",
        "Frequency of customer complaints within the last year",
        "Number of days since last customer service interaction"
    ]
}
```

**Field Descriptions:**
- **`enriched_column_names`**: List of column names that exist in the `enriched_train.csv` and `enriched_test.csv` files - these represent the ground truth features that domain experts consider important
- **`features_descriptions`**: Human-readable explanations of what each ground truth feature represents and how it should be interpreted - must match the order of `enriched_column_names`

### Agent Solution Forma`

When agents generate solutions, they follow this schema in `solution_attributes.json`:

```json
{
    "solved_by": "agent_name",
    "enriched_column_names": ["total_data_usage", "engagement_frequency"],
    "sorted_feature_functions": {
        "0.85": {
            "name": "total_data_usage", 
            "code": "def total_data_usage(row, aux_dataframes):\n    return aux_dataframes['usage_table.csv'].loc[aux_dataframes['usage_table.csv']['customer_id'] == row['customer_id'], 'data_usage'].sum()"
        }
    },
    "feature_descriptions": ["Total data usage across all periods", "Customer engagement frequency score"]
}
```

## Data File Requirements

### Training and Test Data (`train.csv`, `test.csv`)
- Must contain the target column specified in `problem.json`
- Should have consistent column names and data types
- Both files should have the same schema (columns and types)

### Auxiliary Data Tables
- Optional CSV files that provide additional context or lookup data
- Can be referenced in agent feature engineering functions via the `secondary_data` parameter
- Must be placed in the `problem/data/` directory

### Enriched Data (`enriched_train.csv`, `enriched_test.csv`)
- Must contain all original columns from train.csv/test.csv
- Must contain additional columns matching the `enriched_column_names` in `solution.json`
- **Important**: Do not include auxiliary table data directly - only the computed ground truth features

