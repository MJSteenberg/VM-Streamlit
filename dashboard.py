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

# Define all payment types
ALL_PAYMENT_TYPES = ['InAppPurchase', 'AndroidPayment', 'StripePayment', 'CouponRedemptionCreditReseller']

@st.cache_data
def process_user_data_cached(payment_types_key):
    """
    Cached version of user data processing.
    payment_types_key should be a tuple (immutable) of the selected payment types
    """
    users, purchases = load_data()
    return process_user_data(users, purchases, payment_types_key)

@st.cache_data
def process_revenue_data_cached(payment_types_key):
    """
    Cached version of revenue data processing.
    payment_types_key should be a tuple (immutable) of the selected payment types
    """
    users, purchases = load_data()
    return process_revenue_data(users, purchases, payment_types_key)

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
        else:
            return date_val.replace(tzinfo=None)

    # Apply the function to both DataFrames
    users['Created at'] = users['Created at'].apply(parse_and_standardize_date)
    purchases['Created at [Route Purchase]'] = purchases['Created at [Route Purchase]'].apply(parse_and_standardize_date)

    # Drop 'None1' and 'None2' columns if they exist
    users = users.drop(columns=['None1', 'None2'], errors='ignore')

    # Ensure IDs are numeric
    users['Id'] = pd.to_numeric(users['Id'], errors='coerce')
    purchases['Id [User]'] = pd.to_numeric(purchases['Id [User]'], errors='coerce')
    purchases['Price [Route Purchase]'] = pd.to_numeric(purchases['Price [Route Purchase]'], errors='coerce')

    return users, purchases

def get_user_id_range(users, month):
    month_users = users[users['Created at'].dt.to_period('M') == month]
    if len(month_users) > 0:
        numeric_ids = pd.to_numeric(month_users['Id'], errors='coerce')
        return numeric_ids.min(), numeric_ids.max()
    return np.nan, np.nan

def process_user_data(users, purchases, selected_payment_types):
    # Filter purchases based on payment types
    filtered_purchases = purchases[purchases['Type [Payment]'].isin(selected_payment_types)]
    
    # Calculate New Users for each month
    new_users = users.groupby(users['Created at'].dt.to_period('M')).size()
    
    # Get user ID ranges
    id_ranges = [get_user_id_range(users, month) for month in new_users.index]
    id_range_df = pd.DataFrame(id_ranges, columns=['Lowest ID', 'Highest ID'], index=new_users.index)
    
    def calculate_monthly_metrics(month):
        month_start = month.to_timestamp()
        month_end = month.to_timestamp(how='end')
        
        # New users for this month
        new_users_ids = users[users['Created at'].dt.to_period('M') == month]['Id']
        
        # New paying users (first month)
        new_paying = filtered_purchases[
            (filtered_purchases['Id [User]'].isin(new_users_ids)) & 
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end)
        ]['Id [User]'].nunique()
        
        # First-time purchasers this month
        first_time_purchasers = filtered_purchases[
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end) &
            (~filtered_purchases['Id [User]'].isin(
                filtered_purchases[filtered_purchases['Created at [Route Purchase]'] < month_start]['Id [User]']
            ))
        ]['Id [User]'].unique()
        
        # Returning first purchase
        highest_id_before_month = id_range_df.loc[:month - 1, 'Highest ID'].max()
        returning_first = sum(id <= highest_id_before_month for id in first_time_purchasers if pd.notnull(id))
        
        # All purchasers this month
        all_purchasers = filtered_purchases[
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end)
        ]['Id [User]'].unique()
        
        all_returning = len(set(all_purchasers) - set(new_users_ids))
        total_paying = len(all_purchasers)
        
        return pd.Series({
            'New Paying Users': new_paying,
            'Returning, First Purchase': returning_first,
            'All Returning': all_returning,
            'Total Paying': total_paying
        })
    
    # Calculate metrics for each month
    monthly_metrics = pd.DataFrame([calculate_monthly_metrics(month) for month in new_users.index],
                                 index=new_users.index)
    
    # Add new users to the dataframe
    monthly_metrics['New Users'] = new_users
    
    # Calculate derived metrics
    monthly_metrics['All New Paying'] = monthly_metrics['New Paying Users'] + monthly_metrics['Returning, First Purchase']
    monthly_metrics['Returning, Repeat'] = monthly_metrics['All Returning'] - monthly_metrics['Returning, First Purchase']
    monthly_metrics['Percentage Returning, Repeat'] = (monthly_metrics['Returning, Repeat'] / monthly_metrics['Total Paying'] * 100).round(2)
    monthly_metrics['Cumulative All New Paying'] = monthly_metrics['All New Paying'].cumsum()
    
    # Add ID ranges
    monthly_metrics['Lowest ID'] = id_range_df['Lowest ID']
    monthly_metrics['Highest ID'] = id_range_df['Highest ID']
    
    # Save the period index before resetting
    period_index = monthly_metrics.index
    
    # Convert to timestamps
    monthly_metrics = monthly_metrics.reset_index(drop=True)  # Drop the old index
    monthly_metrics['Created at'] = [period.to_timestamp() for period in period_index]
    
    return monthly_metrics

def process_revenue_data(users, purchases, selected_payment_types):
    filtered_purchases = purchases[purchases['Type [Payment]'].isin(selected_payment_types)]
    
    def calculate_monthly_revenue(month):
        month_start = month.to_timestamp()
        month_end = month.to_timestamp(how='end')
        
        # New users for this month
        new_users_ids = users[users['Created at'].dt.to_period('M') == month]['Id']
        
        # Total revenue from new users
        new_users_revenue = filtered_purchases[
            filtered_purchases['Id [User]'].isin(new_users_ids)
        ]['Price [Route Purchase]'].sum()
        
        # New paying users revenue (first month)
        new_paying_revenue = filtered_purchases[
            (filtered_purchases['Id [User]'].isin(new_users_ids)) & 
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end)
        ]['Price [Route Purchase]'].sum()
        
        # First-time purchasers this month
        first_time_purchasers = filtered_purchases[
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end) &
            (~filtered_purchases['Id [User]'].isin(
                filtered_purchases[filtered_purchases['Created at [Route Purchase]'] < month_start]['Id [User]']
            ))
        ]
        
        # Revenue from returning first-time purchasers
        previous_users = users[users['Created at'] < month_start]['Id']
        returning_first_revenue = first_time_purchasers[
            first_time_purchasers['Id [User]'].isin(previous_users)
        ]['Price [Route Purchase]'].sum()
        
        # All revenue this month
        month_revenue = filtered_purchases[
            (filtered_purchases['Created at [Route Purchase]'] >= month_start) & 
            (filtered_purchases['Created at [Route Purchase]'] <= month_end)
        ]
        
        returning_revenue = month_revenue[~month_revenue['Id [User]'].isin(new_users_ids)]['Price [Route Purchase]'].sum()
        total_revenue = month_revenue['Price [Route Purchase]'].sum()
        
        return pd.Series({
            'New Users Revenue': new_users_revenue,
            'New Paying Users Revenue': new_paying_revenue,
            'Returning First Purchase Revenue': returning_first_revenue,
            'All Returning Revenue': returning_revenue,
            'Total Revenue': total_revenue
        })
    
    # Calculate metrics for each month
    period_index = users['Created at'].dt.to_period('M').unique()
    monthly_revenue = pd.DataFrame([
        calculate_monthly_revenue(month) 
        for month in period_index
    ], index=period_index)
    
    # Calculate derived metrics
    monthly_revenue['All New Paying Revenue'] = monthly_revenue['New Paying Users Revenue'] + monthly_revenue['Returning First Purchase Revenue']
    monthly_revenue['Returning Repeat Revenue'] = monthly_revenue['All Returning Revenue'] - monthly_revenue['Returning First Purchase Revenue']
    monthly_revenue['Percentage Returning Repeat Revenue'] = (monthly_revenue['Returning Repeat Revenue'] / monthly_revenue['Total Revenue'] * 100).round(2)
    monthly_revenue['Cumulative All New Paying Revenue'] = monthly_revenue['All New Paying Revenue'].cumsum()
    
    # Convert the period index to timestamp
    monthly_revenue = monthly_revenue.reset_index(drop=True)  # Drop the old index
    monthly_revenue['Created at'] = [period.to_timestamp() for period in period_index]
    
    return monthly_revenue

# Load base data
users, purchases = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Add payment type filter
selected_payment_types = st.sidebar.multiselect(
    "Select Payment Types",
    options=ALL_PAYMENT_TYPES,
    default=ALL_PAYMENT_TYPES
)

if not selected_payment_types:
    st.warning("Please select at least one payment type.")
    st.stop()

# Convert selected payment types to tuple for caching
payment_types_key = tuple(sorted(selected_payment_types)) 

# Process data with selected payment types using cached functions
df = process_user_data_cached(payment_types_key)
df_revenue = process_revenue_data_cached(payment_types_key)

# # Process data with selected payment types
# df = process_user_data(users, purchases, selected_payment_types)
# df_revenue = process_revenue_data(users, purchases, selected_payment_types)

# Add date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(df['Created at'].min(), df['Created at'].max()),
    min_value=df['Created at'].min().date(),
    max_value=df['Created at'].max().date()
)

# Filter data based on date range
mask = (df['Created at'].dt.date >= date_range[0]) & (df['Created at'].dt.date <= date_range[1])
mask2 = (df_revenue['Created at'].dt.date >= date_range[0]) & (df_revenue['Created at'].dt.date <= date_range[1])
filtered_df = df[mask]
filtered_df_revenue = df_revenue[mask2]

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
tab1, tab2, tab3, tab4 = st.tabs(["User Acquisition", "Payment Metrics", "Conversion Analysis", "Revenue Metrics"])

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
    
    fig2.add_trace(go.Scatter(
        x=filtered_df['Created at'], 
        y=filtered_df['Cumulative All New Paying'],
        name='Cumulative New Paying Users',
        mode='lines',
        line=dict(width=3)
    ))
    
    fig2.add_trace(go.Bar(
        x=filtered_df['Created at'],
        y=filtered_df['All New Paying'],
        name='New Paying Users (Monthly)',
        marker_color='#AF7AC5',
        yaxis='y2'
    ))
    
    fig2.update_layout(
        title='Payment Growth Trends',
        xaxis_title='Date',
        yaxis_title='Cumulative Users',
        yaxis2=dict(
            title='Monthly New Paying Users',
            overlaying='y',
            side='right'
        ),
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

with tab4:
    # Revenue Growth
    fig_revenue = go.Figure()
    fig_revenue.add_trace(go.Bar(
        x=filtered_df_revenue['Created at'],
        y=filtered_df_revenue['Total Revenue'],
        name='Monthly Revenue'
    ))
    fig_revenue.add_trace(go.Scatter(
        x=filtered_df_revenue['Created at'],
        y=filtered_df_revenue['Total Revenue'].cumsum(),
        name='Cumulative Revenue',
        yaxis='y2'
    ))
    fig_revenue.update_layout(
        title='Revenue Growth',
        yaxis_title='Monthly Revenue ($)',
        yaxis2=dict(title='Cumulative Revenue ($)', overlaying='y', side='right'),
        legend=dict(x=0.02, y=1.15, orientation='h')
    )
    st.plotly_chart(fig_revenue, use_container_width=True)

    # Revenue Breakdown
    fig_rev_breakdown = px.bar(filtered_df_revenue, x='Created at',
        y=['New Paying Users Revenue', 'Returning First Purchase Revenue', 'Returning Repeat Revenue'],
        title='Revenue Breakdown by User Type',
        labels={'value': 'Revenue ($)', 'variable': 'Revenue Type'}
    )
    st.plotly_chart(fig_rev_breakdown, use_container_width=True)

    # Revenue per User Trend
    monthly_arpu = filtered_df_revenue['Total Revenue'] / filtered_df['New Users']
    monthly_arppu = filtered_df_revenue['Total Revenue'] / filtered_df['Total Paying']
    
    st.markdown("""
    ARPU and ARPPU is calculated as:
    ```
    ARPU = (Total Revenue / New Users) × 100
    ```
    ```
    ARPPU = (Total Revenue / All Paying Users) × 100
    ```
    """)
    fig_arpu = go.Figure()
    fig_arpu.add_trace(go.Scatter(x=filtered_df['Created at'], y=monthly_arpu, name='ARPU'))
    fig_arpu.add_trace(go.Scatter(x=filtered_df['Created at'], y=monthly_arppu, name='ARPPU'))
    fig_arpu.update_layout(
        title='Revenue per User Trends',
        yaxis_title='Revenue per User ($)'
    )
    st.plotly_chart(fig_arpu, use_container_width=True)

# Detailed metrics table
st.subheader("Detailed Metrics")

# Create a copy for display
display_df = filtered_df.copy()
display_revenue_df = filtered_df_revenue.copy()

# Format the 'Created at' column to show month and year and set it as index
display_df['Created at'] = display_df['Created at'].dt.strftime('%B %Y')
display_revenue_df['Created at'] = display_revenue_df['Created at'].dt.strftime('%B %Y')

display_df = display_df.set_index('Created at')
display_revenue_df = display_revenue_df.set_index('Created at')

# Add revenue columns to display_df
display_df['Total Revenue'] = display_revenue_df['Total Revenue']
display_df['ARPU'] = display_revenue_df['Total Revenue'] / display_df['New Users']
display_df['ARPPU'] = display_revenue_df['Total Revenue'] / display_df['Total Paying']

# Reorder the columns
display_df = display_df[['New Users', 'New Paying Users', 'Returning, First Purchase', 
                        'All New Paying', 'Cumulative All New Paying', 'All Returning', 
                        'Returning, Repeat', 'Total Paying', 'Percentage Returning, Repeat',
                        'Total Revenue', 'ARPU', 'ARPPU']]


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
        'Percentage Returning, Repeat': '{:.1f}%',
        'Total Revenue': '${:,.2f}',
        'ARPU': '${:,.2f}',
        'ARPPU': '${:,.2f}'
    }),
    use_container_width=True
)