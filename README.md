# BLOC Summer Sessions 2025 Analysis

An interactive web application to analyze climbing competition data from the Boulder Summer Series. This project provides a modern UI with filtering, searching, and interactive visualization capabilities.

![Boulder Summer Series Dashboard](https://blocsummer.at/wpbs/wp-content/uploads/2023/04/Bloc-Summer-2023_Logo_Header-400x130.webp)

## Features

- **Interactive Dashboard**: Summary statistics and visualizations for overall competition results
- **Climber Analysis**: Search and sort functionality for finding specific climbers and their performance
- **Gym Analysis**: Compare statistics across different gyms with filtering options
- **Boulder Analysis**: Detailed investigation of boulder popularity and completion rates
- **Data Visualization**: Interactive charts and metrics for comprehensive analysis

## Project Structure

```
blocksummer/
├── .streamlit/         # Streamlit configuration
│   └── config.toml     # Streamlit app settings
├── app.py              # Main Streamlit application
├── main.py             # Web scraper for competition data
├── stats.py            # Statistical analysis functions
├── results.json        # Scraped competition data
├── pyproject.toml      # Poetry dependencies
└── README.md           # Project documentation
```

## Installation

This project uses Poetry for dependency management. To install the required dependencies:

```bash
# Install Poetry (if not already installed)
# Windows
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Install dependencies
poetry install
```

## Running the Application

To run the Streamlit application:

```bash
# Activate the poetry environment
poetry shell

# Run the app
streamlit run app.py
```

The application should open automatically in your default web browser at `http://localhost:8501`.

## Application Components

### Data Collection (`main.py`)

The application uses data scraped from the Boulder Summer Series competition website. To collect new data, run:

```bash
poetry run python main.py
```

This will:
1. Scrape the competition rankings page
2. Gather details for each participant
3. Process and save the results to `results.json`

### Data Analysis (`stats.py`)

The statistical analysis module provides functions to:
- Load and parse the results data
- Calculate metrics per gym and climber
- Generate visualization data for boulder popularity and completion rates

### Web Interface (`app.py`)

The Streamlit interface is organized into four main tabs:

1. **Dashboard**: Overall metrics and key insights
2. **Climbers**: Searchable climber database with detailed performance metrics
3. **Gyms**: Comparative analysis between different climbing gyms
4. **Boulder Analysis**: Detailed statistics on boulder popularity and completion rates

## Code Practices

This project follows modern Python best practices:

- **Type Hints**: Type annotations for better code comprehension and IDE support
- **Documentation**: Docstrings and clear function naming
- **Modular Design**: Separate modules for data collection, analysis, and presentation
- **Error Handling**: Graceful error management with informative user messages
- **Caching**: Performance optimization with Streamlit's caching mechanism
- **Testing**: Robust data validation and error handling

## Extending the Project

### Adding New Features

To extend the application with new features:

1. For new data analysis capabilities, add functions to `stats.py`
2. For additional visualizations, add rendering functions to `app.py`
3. For new data sources, create additional scrapers in `main.py` or a new module

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Technologies Used

- **Python 3.12+**: Modern Python features including type annotations
- **Streamlit**: Web interface framework with reactive design
- **Plotly**: Interactive data visualizations
- **Pandas**: Data analysis and manipulation
- **Selenium**: Web scraping for data collection
- **Poetry**: Dependency management

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Boulder Summer Series for providing the competition data
- The open source community for the excellent libraries that made this project possible 