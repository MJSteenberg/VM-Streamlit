# User Metrics Dashboard

A comprehensive analytics dashboard built with Streamlit and Plotly to visualize user acquisition and payment patterns.

## Features

- **User Acquisition Metrics**
  - New user growth tracking
  - New paying users monitoring
  - User conversion analysis

- **Payment Analytics**
  - Cumulative growth visualization
  - Monthly new paying users tracking
  - User payment behavior breakdown
  - Payment type distribution analysis

## Prerequisites

```bash
python >= 3.7
streamlit
pandas
plotly
numpy
python-dateutil
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MJSteenberg/VM-Streamlit.git
cd VM-Streamlit
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data files:
   - Place `Users.csv` and `Purchase Data.csv` in the project directory
   - Ensure the CSV files have the required columns:
     - Users.csv: 'Id', 'Created at'
     - Purchase Data.csv: 'Id [User]', 'Created at [Route Purchase]', 'Type [Payment]'

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the dashboard at `http://localhost:8501`

## Data Format

### Users.csv
```csv
Id,Created at
1,2023-01-01 10:00:00
2,2023-01-02 15:30:00
```

### Purchase Data.csv
```csv
Id [User],Created at [Route Purchase],Type [Payment]
1,2023-01-15 12:00:00,StripePayment
2,2023-01-20 14:30:00,InAppPurchase
```

## Dashboard Sections

1. **Top Level Metrics**
   - Total New Users
   - Total Paying Users
   - Overall Conversion Rate
   - Average Repeat Purchase Rate

2. **User Acquisition Tab**
   - New Users vs New Paying Users trend

3. **Payment Metrics Tab**
   - Payment Growth Trends (Cumulative and Monthly)
   - User Payment Behavior Breakdown
   - Distribution of User Payment Types

4. **Conversion Analysis Tab**
   - Monthly Conversion Rate Trends
   - Conversion Rate Analysis

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Tech

- Built with [Streamlit](https://streamlit.io/)
- Visualization powered by [Plotly](https://plotly.com/)
- Data analysis with [Pandas](https://pandas.pydata.org/)
