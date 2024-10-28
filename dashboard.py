import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
from dateutil import parser

# Set page config
st.set_page_config(page_title="User Metrics Dashboard", layout="wide")

# Title and description
st.title("User Metrics Dashboard")
st.markdown("Analysis of user acquisition and purchase patterns")

# Function to load data (replace this with your actual data loading logic)
@st.cache_data
def load_data():

    users = pd.read_csv("Users.csv")

    purchases = pd.read_csv("Purchase Data.csv")

    def parse_and_standardize_date(date_val):
        if isinstance(date_val, pd.Timestamp):
            return date_val.tz_localize(None)
        elif isinstance(date_val, str):
            try:
                return parser.parse(date_val).replace(tzinfo=None)
            except:
                return pd.NaT
        else:  # Assuming it's already a datetime object
            return date_val.replace(tzinfo=None)

    # Apply the function to both DataFrames
    users['Created at'] = users['Created at'].apply(parse_and_standardize_date)
    purchases['Created at [Route Purchase]'] = purchases['Created at [Route Purchase]'].apply(parse_and_standardize_date)

    # Drop 'None1' and 'None2' columns if they exist
    users = users.drop(columns=['None1', 'None2'], errors='ignore')

    # Ensure 'Id' column in users DataFrame is numeric
    users['Id'] = pd.to_numeric(users['Id'], errors='coerce')

    # Ensure 'Id [User]' column in purchases DataFrame is numeric
    purchases['Id [User]'] = pd.to_numeric(purchases['Id [User]'], errors='coerce')

    # Define payment types to include
    payment_types = ['InAppPurchase', 'AndroidPayment', 'StripePayment', 'CouponRedemptionCreditReseller']

    # Filter purchases based on payment types
    purchases = purchases[purchases['Type [Payment]'].isin(payment_types)]

    # Calculate New Users for each month
    new_users = users.groupby(users['Created at'].dt.to_period('M')).size()

    # Function to get lowest and highest user IDs for each month
    def get_user_id_range(month):
        month_users = users[users['Created at'].dt.to_period('M') == month]
        if len(month_users) > 0:
            numeric_ids = pd.to_numeric(month_users['Id'], errors='coerce')
            return numeric_ids.min(), numeric_ids.max()
        return np.nan, np.nan

    # Get user ID ranges for each month
    id_ranges = [get_user_id_range(month) for month in new_users.index]
    id_range_df = pd.DataFrame(id_ranges, columns=['Lowest ID', 'Highest ID'], index=new_users.index)

    # Calculate New Paying Users, First Month
    def new_paying_users_first_month(month):
        month_start = month.to_timestamp()
        month_end = month.to_timestamp(how='end')
        new_users_ids = users[users['Created at'].dt.to_period('M') == month]['Id']
        new_paying_users = purchases[
            (purchases['Id [User]'].isin(new_users_ids)) & 
            (purchases['Created at [Route Purchase]'] >= month_start) & 
            (purchases['Created at [Route Purchase]'] <= month_end)
        ]['Id [User]'].nunique()
        return new_paying_users

    new_paying_users = pd.Series([new_paying_users_first_month(month) for month in new_users.index], index=new_users.index)

    # Calculate Returning, First Purchase
    def returning_first_purchase(month):
        month_start = month.to_timestamp()
        month_end = month.to_timestamp(how='end')
        
        # Get the highest ID for users created before this month
        previous_month = month - 1
        highest_id_before_month = id_range_df.loc[:previous_month, 'Highest ID'].max()
        
        # Get users who made their first purchase this month
        first_time_purchasers = purchases[
            (purchases['Created at [Route Purchase]'] >= month_start) & 
            (purchases['Created at [Route Purchase]'] <= month_end) &
            (~purchases['Id [User]'].isin(purchases[purchases['Created at [Route Purchase]'] < month_start]['Id [User]']))
        ]['Id [User]'].unique()
        
        # Convert IDs to integers for comparison
        first_time_purchasers = pd.to_numeric(first_time_purchasers, errors='coerce')
        highest_id_before_month = pd.to_numeric(highest_id_before_month, errors='coerce')
        
        # Count returning users (created before this month) who made their first purchase this month
        returning_first_purchase_count = sum(id <= highest_id_before_month for id in first_time_purchasers if pd.notnull(id))
        
        return returning_first_purchase_count

    returning_first_purchase_users = pd.Series([returning_first_purchase(month) for month in new_users.index], index=new_users.index)

    # Calculate All Returning users
    def all_returning_users(month):
        month_start = month.to_timestamp()
        month_end = month.to_timestamp(how='end')
        
        # Get all users who made a purchase this month
        all_purchasers = purchases[
            (purchases['Created at [Route Purchase]'] >= month_start) & 
            (purchases['Created at [Route Purchase]'] <= month_end)
        ]['Id [User]'].unique()
        
        # Get new users for this month
        new_users_ids = users[users['Created at'].dt.to_period('M') == month]['Id']
        
        # Count returning users (all purchasers minus new paying users)
        returning_users_count = len(set(all_purchasers) - set(new_users_ids))
        
        return returning_users_count, len(all_purchasers)

    all_returning, total_paying = zip(*[all_returning_users(month) for month in new_users.index])
    all_returning = pd.Series(all_returning, index=new_users.index)
    total_paying = pd.Series(total_paying, index=new_users.index)

    # Calculate All New Paying
    all_new_paying = new_paying_users + returning_first_purchase_users

    # Calculate Returning, Repeat
    returning_repeat = all_returning - returning_first_purchase_users

    # Calculate Percentage Returning, Repeat
    percentage_returning_repeat = (returning_repeat / total_paying * 100).round(2)

    # Calculate Cumulative All New Paying
    cumulative_all_new_paying = all_new_paying.cumsum()

    # Create the final dataframe
    result_df = pd.DataFrame({
        'New Users': new_users,
        'New Paying Users': new_paying_users,
        'Returning, First Purchase': returning_first_purchase_users,
        'All New Paying': all_new_paying,
        'Cumulative All New Paying': cumulative_all_new_paying,
        'All Returning': all_returning,
        'Returning, Repeat': returning_repeat,
        'Total Paying': total_paying,
        'Percentage Returning, Repeat': percentage_returning_repeat
    })

    # Add the ID range information
    result_df['Lowest ID'] = id_range_df['Lowest ID']
    result_df['Highest ID'] = id_range_df['Highest ID']

    # Fill NaN values with 0 and convert to integer (except for the percentage column)
    result_df = result_df.fillna(0)
    for column in result_df.columns:
        if column not in ['Percentage Returning, Repeat', 'Created at']:
            result_df[column] = result_df[column].astype(int)

    # Reset index to make the date a column
    result_df = result_df.reset_index()
    result_df['Created at'] = result_df['Created at'].dt.to_timestamp()

    # Display the result
    return result_df

# Load the data
df = load_data()

# Convert 'Created at' to datetime if it isn't already
df['Created at'] = pd.to_datetime(df['Created at'])

# Sidebar filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Created at'].min(), df['Created at'].max()),
    min_value=df['Created at'].min().date(),
    max_value=df['Created at'].max().date()
)

# Filter data based on date range
mask = (df['Created at'].dt.date >= date_range[0]) & (df['Created at'].dt.date <= date_range[1])
filtered_df = df[mask]

# Top level metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total New Users", f"{filtered_df['New Users'].sum():,}")
with col2:
    st.metric("Total Paying Users", f"{filtered_df['Total Paying'].sum():,}")
with col3:
    conversion_rate = (filtered_df['All New Paying'].sum() / filtered_df['New Users'].sum() * 100)
    st.metric("Overall Conversion Rate", f"{conversion_rate:.1f}%")
with col4:
    repeat_rate = filtered_df['Percentage Returning, Repeat'].mean()
    st.metric("Avg Repeat Purchase Rate", f"{repeat_rate:.1f}%")

# Main visualizations
st.subheader("User Growth Trends")
tab1, tab2, tab3 = st.tabs(["User Acquisition", "Payment Metrics", "Conversion Analysis"])

with tab1:
    # User acquisition trends
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=filtered_df['Created at'], y=filtered_df['New Users'],
                         name='New Users', marker_color='#2E86C1'))
    fig1.add_trace(go.Bar(x=filtered_df['Created at'], y=filtered_df['New Paying Users'],
                         name='New Paying Users', marker_color='#28B463'))
    fig1.update_layout(
        title='New Users vs New Paying Users Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Users',
        barmode='group'
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    # Payment metrics visualization
    fig2 = go.Figure()
    
    # First trace (line) - Cumulative New Paying Users
    fig2.add_trace(go.Scatter(
        x=filtered_df['Created at'], 
        y=filtered_df['Cumulative All New Paying'],
        name='Cumulative New Paying Users',
        mode='lines',
        line=dict(width=3)
    ))
    
    # Second trace (bar) - New Paying Users Monthly (on secondary y-axis)
    fig2.add_trace(go.Bar(
        x=filtered_df['Created at'],
        y=filtered_df['All New Paying'],
        name='New Paying Users (Monthly)',
        marker_color='#AF7AC5',
        yaxis='y2'  # Assign to secondary y-axis
    ))
    
    # Update layout with secondary y-axis
    fig2.update_layout(
        title='Payment Growth Trends',
        xaxis_title='Date',
        yaxis_title='Cumulative Users',
        yaxis2=dict(
            title='Monthly New Paying Users',
            overlaying='y',
            side='right'
        ),
        # Adjust legend position to avoid overlap with the secondary y-axis
        legend=dict(x=0.02, y=1.15, orientation='h')
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # User Payment Behavior Breakdown
    fig3 = px.bar(filtered_df, x='Created at',
                 y=['New Paying Users', 'Returning, First Purchase', 'Returning, Repeat'],
                 title='User Payment Behavior Breakdown',
                 labels={'value': 'Number of Users', 'variable': 'User Type'})
    st.plotly_chart(fig3, use_container_width=True)
    
    # Distribution of User Payment Types
    avg_percentages = pd.DataFrame({
        'Category': ['New Paying', 'Returning (First)', 'Returning (Repeat)'],
        'Percentage': [
            filtered_df['New Paying Users'].sum() / filtered_df['Total Paying'].sum() * 100,
            filtered_df['Returning, First Purchase'].sum() / filtered_df['Total Paying'].sum() * 100,
            filtered_df['Returning, Repeat'].sum() / filtered_df['Total Paying'].sum() * 100
        ]
    })
    fig4 = px.pie(avg_percentages, values='Percentage', names='Category',
                 title='Distribution of User Payment Types')
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    # Add description
    st.markdown("""
    The conversion rate shows the percentage of new users who become paying users. It is calculated as:
    ```
    Conversion Rate = (All New Paying Users / Total New Users) × 100
    ```
    - **All New Paying Users** includes both users who paid in their first month and those who converted later
    - A higher conversion rate indicates more effective user monetization
    """)
    # Conversion metrics
    monthly_conversion = (filtered_df['All New Paying'] / filtered_df['New Users'] * 100).round(2)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=filtered_df['Created at'],
                             y=monthly_conversion,
                             mode='lines+markers',
                             name='Monthly Conversion Rate',
                             line=dict(width=2)))
    fig5.update_layout(
        title='Monthly Conversion Rate Trend',
        xaxis_title='Date',
        yaxis_title='Conversion Rate (%)',
        yaxis_range=[0, max(monthly_conversion) * 1.1]
    )
    st.plotly_chart(fig5, use_container_width=True)

# Detailed metrics table
st.subheader("Detailed Metrics")

# Create a copy for display
display_df = filtered_df.copy()

# Format the 'Created at' column to show month and year and set it as index
display_df['Created at'] = display_df['Created at'].dt.strftime('%B %Y')
display_df = display_df.set_index('Created at')

# Display the dataframe with index (which will be frozen)
st.dataframe(
    data=display_df.style.format({
        'New Users': '{:,.0f}',
        'New Paying Users': '{:,.0f}',
        'Returning, First Purchase': '{:,.0f}',
        'All New Paying': '{:,.0f}',
        'Cumulative All New Paying': '{:,.0f}',
        'All Returning': '{:,.0f}',
        'Returning, Repeat': '{:,.0f}',
        'Total Paying': '{:,.0f}',
        'Percentage Returning, Repeat': '{:.1f}%'
    }),
    use_container_width=True
)