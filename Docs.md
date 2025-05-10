# Prompt-to-Pipeline Framework Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Installation & Setup](#installation--setup)
4. [Using the Web Interface](#using-the-web-interface)
5. [API Reference](#api-reference)
6. [Customization & Extension](#customization--extension)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Introduction

Prompt-to-Pipeline is a framework that converts natural language requests into data pipelines. It automates the process of data ingestion, transformation, analysis, visualization, and storytelling. Using AI and advanced analytics, it enables users to generate insights from data with minimal technical knowledge.

Key features include:
- Data ingestion from URLs (CSV, Parquet)
- Automated data cleaning and transformation
- Schema inference and dimension/fact table creation
- AI-generated data visualizations based on natural language requests
- Automated data storytelling and narrative generation
- Interactive web interface for managing the entire pipeline

## Architecture Overview

The framework consists of several integrated components:

### Core Components

1. **Ingestion Engine**: Downloads and parses data files from URLs
2. **Staging Storage**: Loads data into Snowflake tables
3. **Analysis Runner**: Generates and executes visualization code
4. **Analysis Suggester**: Proposes relevant analyses based on data schema
5. **Auto Story Pipeline**: Orchestrates the entire process and generates narratives

### Technology Stack

- **Backend**: FastAPI with WebSockets for real-time updates
- **Frontend**: React with Tailwind CSS
- **Database**: Snowflake for data storage and querying
- **AI**: LLaMa model via Ollama for code generation and narrative creation
- **Visualization**: Plotly for interactive charts

### Data Flow

1. User submits data URL through the web interface
2. Data is downloaded, parsed, and loaded into Snowflake
3. Data is cleaned and transformed (nulls dropped, columns renamed, etc.)
4. Dimension tables are extracted for categorical variables
5. AI suggests relevant analyses based on the schema
6. Visualizations are generated and saved as HTML
7. AI creates a narrative story explaining the insights
8. All results are displayed in the web interface

## Installation & Setup

### Prerequisites

- Docker and Docker Compose
- Snowflake account
- 8GB+ RAM recommended for running LLaMa models

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd prompt-to-pipeline
   ```

2. Configure your Snowflake credentials in the `.env` file:
   ```
   SNOW_ACCOUNT=your_account
   SNOW_USER=your_username
   SNOW_PWD=your_password
   SNOW_ROLE=your_role
   SNOW_WAREHOUSE=your_warehouse
   SNOW_DATABASE=your_database
   SNOW_SCHEMA=your_schema
   ```

3. Run the setup script:
   ```bash
   ./setup.sh
   ```

4. Access the application at `http://localhost:8000`

### Manual Installation

1. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

2. Install Ollama and download the LLaMa model:
   ```bash
   # Follow instructions at https://ollama.ai/
   ollama pull llama3:8b
   ```

3. Start the FastAPI server:
   ```bash
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Using the Web Interface

### Data Ingestion

1. Navigate to the main page at `http://localhost:8000`
2. Enter a URL to a CSV or Parquet file in the "Data URL" field
3. Optionally provide a "Table Hint" to name your tables in Snowflake
4. Click "Start Pipeline" to begin processing
5. Monitor progress in the "Execution Logs" tab

### Viewing Results

Once processing is complete, the application will automatically switch to the "Charts" tab.

1. **Charts Tab**: Browse through generated visualizations
   - Use the chart selector to navigate between different charts
   - Each chart is interactive (hover, zoom, pan)

2. **Data Story Tab**: Read the AI-generated narrative that explains the insights
   - The story connects multiple visualizations into a coherent narrative
   - Key findings are highlighted and explained

### Custom Analysis

To create additional visualizations:

1. Enter a natural language request in the "Analysis Request" field
   - Example: "Show correlation between height and weight"
   - Example: "Create a histogram of ages grouped by gender"
2. Click "Run Analysis" to generate the visualization
3. View the new chart in the "Charts" tab

## API Reference

### REST Endpoints

#### `POST /api/ingest`
Start a data ingestion and analysis pipeline.

**Request Body:**
```json
{
  "url": "https://example.com/data.csv",
  "table_hint": "sales_data"  // optional
}
```

**Response:**
```json
{
  "status": "Pipeline started",
  "message": "Check WebSocket for progress updates"
}
```

#### `GET /api/charts`
Retrieve a list of generated charts.

**Response:**
```json
{
  "charts": [
    {
      "url": "/charts/chart_20250509_123456_correlation_height_weight.html",
      "filename": "chart_20250509_123456_correlation_height_weight.html",
      "created": 1746882169.123
    },
    // ...
  ]
}
```

#### `GET /api/story`
Retrieve the generated data story.

**Response:**
```json
{
  "story": "Our analysis of the height and weight data reveals several interesting patterns..."
}
```

#### `POST /api/analysis`
Run a custom analysis on the current dataset.

**Request Body:**
```json
{
  "request": "Show correlation between height and weight"
}
```

**Response:**
```json
{
  "result": "Chart saved to charts2/chart_20250509_123456_correlation_height_weight.html"
}
```

### WebSocket Events

Connect to `/ws` to receive real-time updates.

**Log Message:**
```json
{
  "type": "log",
  "message": "Running analysis for: Show correlation between height and weight"
}
```

**Pipeline Complete:**
```json
{
  "type": "pipeline_complete",
  "charts": ["/charts/chart_1.html", "/charts/chart_2.html"],
  "story": "Our analysis reveals..."
}
```

**Error Message:**
```json
{
  "type": "error",
  "message": "Failed to load data from URL: Connection refused"
}
```

## Customization & Extension

### Adding New Data Sources

To support additional data formats, modify the `_load_dataframe` function in `ingestion_engine.py`:

```python
def _load_dataframe(path: Path) -> pd.DataFrame:
    """Load *path* into a DataFrame based on its extension."""
    ext = path.suffix.lower()
    if ext in {".csv", ".txt"}:
        return pd.read_csv(path)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext == ".json":
        return pd.read_json(path)
    # Add more formats as needed
    raise ValueError(f"Unsupported file extension: {ext}")
```

### Customizing Transformations

The transformation pipeline is defined in `auto_story_pipeline.py`. You can modify the `auto_pipeline` function to add custom transformations:

```python
# Example: Add a new transformation step to standardize numeric columns
from sklearn.preprocessing import StandardScaler

# Inside auto_pipeline function, after cleaning columns:
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_cols) > 0:
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"Standardized {len(numeric_cols)} numeric columns")
```

### Integrating with Different Databases

To use a different database instead of Snowflake, create a new version of `staging_storage.py`:

1. Replace the `_get_connection` function with your database connector
2. Update the `stage_to_snowflake` function to write to your database
3. Update any references in other modules

### Changing the LLM

The framework uses LLaMa via Ollama, but you can replace it with other models:

1. Replace the `ChatOllama` instantiation with your preferred LLM:
   ```python
   # Before
   llm = ChatOllama(model="llama3:8b", temperature=0)
   
   # After (example using OpenAI)
   from langchain_openai import ChatOpenAI
   llm = ChatOpenAI(model="gpt-4", temperature=0)
   ```

2. Update the Docker Compose file if needed to remove Ollama service

## Troubleshooting

### Common Issues

#### "No ingested table found this session"
- Ensure your data URL is valid and accessible
- Check the Snowflake connection settings in `.env`
- Verify the data format is supported (CSV, Parquet)

#### "Failed to generate figure"
- Check if the analysis request is clear and relevant to the loaded data
- Ensure the Ollama service is running (`docker-compose ps`)
- Check if the LLaMa model was downloaded correctly

#### WebSocket Disconnection
- Refresh the page to reconnect
- Check if the backend server is still running
- Look for error messages in the server logs

### Debugging Tools

1. **Logs Directory**: Check detailed logs in these directories:
   - `prompts/`: LLM prompts for analysis requests
   - `responses/`: Raw LLM responses
   - `codes/`: Generated visualization code
   
2. **Database Inspection**: Connect directly to Snowflake to verify table creation:
   ```sql
   SHOW TABLES LIKE 'RAW_%';
   SELECT * FROM RAW_1 LIMIT 10;
   ```

3. **Container Logs**: View Docker container logs:
   ```bash
   docker-compose logs -f backend
   docker-compose logs -f ollama
   ```

## Best Practices

### Data Quality

- Ensure input data has clear, descriptive column names
- Pre-process large datasets to improve performance
- Use table hints that are descriptive of the data content

### Effective Analysis Requests

Write clear, specific analysis requests:

- **Good**: "Show correlation between height and weight with a regression line"
- **Better**: "Create a scatter plot of height vs weight with a linear regression line and display the R² value"

Include context and domain-specific information:

- **Basic**: "Show distribution of sales"
- **Better**: "Create a histogram of monthly sales with a density curve, highlighting seasonality patterns"

### Security Considerations

- Do not use this framework for sensitive or private data without proper security measures
- Update the Snowflake credentials regularly
- Set proper CORS restrictions in production environments
- Add authentication to the FastAPI application for production use

### Performance Optimization

- Use smaller datasets during development (< 100,000 rows)
- Consider adding data sampling for very large datasets
- Increase Docker resource limits for processing large files
- Add caching for frequently used analytics

---

This documentation is a living document and will be updated as the framework evolves. For questions or issues, please open a GitHub issue or contact the project maintainers.