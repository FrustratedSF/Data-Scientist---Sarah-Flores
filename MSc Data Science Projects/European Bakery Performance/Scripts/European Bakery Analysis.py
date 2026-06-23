# =====================================
# EUROPEAN BAKERY PERFORMANCE DASHBOARD 
# =====================================

# |||| 1. IMPORTING LIBRARIES ||||

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from dash import Dash, dcc, html, Input, Output


sns.set(style='whitegrid')


# |||| 2. LOAD THE DATAS ||||

df= pd.read_csv('Bakery_dataset_4566.csv', sep=';')


# |||| 3. DATA CLEANING ||||

df.columns = df.columns.str.strip()  # clean column names
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')  #convert date column
print("\nMissing values:\n", df.isnull().sum())  

df = df.drop_duplicates()  # remove duplicates

df['Revenue(£)'] = pd.to_numeric(df['Revenue(£)'].replace('[£,]', '', regex=True), errors='coerce')
df['Cost(£)'] = pd.to_numeric(df['Cost(£)'].replace('[£,]', '', regex=True), errors='coerce')
df['Profit(£)'] = pd.to_numeric(df['Profit(£)'].replace('[£,]', '', regex=True), errors='coerce')
                                 
# |||| 4. FEATURE ENGINEERING - PROFIT MARGIN

df['Profit Margin'] = df['Profit(£)'] / df['Revenue(£)'].replace(0, pd.NA)

# DASH APP

app = Dash(__name__)

app.layout = html.Div([
    html.H1("European Bakery Performance Dashboard"),

    dcc.Dropdown(
        id='city_filter',
        options=[{'label': c, 'value': c} for c in df['City'].unique()],
        value=df['City'].iloc[0]
    ),

    dcc.Dropdown(
        id='product_filter',
        options=[{'label': p, 'value': p} for p in df['Confectionary'].unique()],
        value=df['Confectionary'].iloc[0]
    ),

    dcc.Graph(id='profit_city_chart'),
    dcc.Graph(id='profit_product_chart'),
    dcc.Graph(id='time_chart'),
    dcc.Graph(id='scatter_chart')

])

# CALLBACK - INTERACTIVE ANALYSIS

@app.callback(
    [Output('profit_city_chart', 'figure'),
     Output('profit_product_chart', 'figure'),
     Output('time_chart', 'figure'),
     Output('scatter_chart', 'figure')],
    [Input('city_filter', 'value'),
     Input('product_filter', 'value')]

)
def update_dashboard(selected_city, selected_product):

    filtered = df[
        (df['City'] == selected_city) |   
        (df['Confectionary'] == selected_product)
    ]

    if filtered.empty:
        empty_fig = px.scatter(title='No data available for selected filters')
        return empty_fig, empty_fig, empty_fig, empty_fig


    # Profit by City (Filtered)
    city_df = filtered.groupby('City', as_index=False)['Profit(£)'].sum()
    
    fig1 = px.bar(
        city_df,
        x='City', 
        y='Profit(£)', 
        title='Profit by City (Filtered)'
    )

    # Profit by Product (Filtered)
    product_df = filtered.groupby('Confectionary', as_index=False)['Profit(£)'].sum()

    fig2 = px.bar(
        product_df,
        x='Confectionary', 
        y='Profit(£)', 
        title='Profit by Product (Filtered)'
    )

    #Time Trend
    time_df = (
        filtered.groupby('Date', as_index=False)['Profit(£)']
        .sum()
        .sort_values('Date')
    )

    fig3 = px.line(
        time_df, 
        x='Date', y='Profit(£)', 
        title='Profit Over Time'
    )

    # Revenue vs Cost
    fig4 = px.scatter(
        filtered,
        x='Revenue(£)',
        y='Cost(£)',
        color='City',
        hover_data =['City', 'Confectionary'],
        title='Revenue vs Cost'
    )

    return fig1, fig2, fig3, fig4

        
# RUN SERVER

if __name__ == '__main__':
    app.run(debug=True)

        
# DATA OVERVIEW

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nFirst 5 rows:")
print(df.head())


# 5. ANALYSIS CALCULATIONS - BUSINESS INSIGHTS

# Profit by City
city_profit = df.groupby('City')['Profit(£)'].sum().sort_values()

# Profit by product
product_profit = df.groupby('Confectionary')['Profit(£)'].sum().sort_values()

#Average Profit Margin by City
margin_city = df.groupby('City')['Profit Margin'].mean()

# Revenue by City
city_revenue = df.groupby('City')['Revenue(£)'].sum()

# Cost by City
city_cost = df.groupby('City')['Cost(£)'].sum()

# Profit by City and Product
city_product_profit = df.pivot_table(
    values='Profit(£)',
    index='City',
    columns='Confectionary',
    aggfunc='sum'

)

print("\n  >>> KEY BUSINESS INSIGHTS <<<")

print("\n --- City Performance ---")
print(f"\nMost profitable city: {city_profit.idxmax()} (£{city_profit.max():,.2f})")
print(f"\nLeast profitable city: {city_profit.idxmin()} (£{city_profit.min():,.2f})")

print("\n --- Product Performance ---")
print(f"\nMost profitable product: {product_profit.idxmax()} (£{product_profit.max():,.2f})")
print(f"\nLeast profitable product: {product_profit.idxmin()} (£{product_profit.min():,.2f})")

print("\n --- Efficiency ---")
print(f"\nHighest profit margin city: {margin_city.idxmax()} ({margin_city.max():,.2%})")
print(f"\nLowest profit margin city: {margin_city.idxmin()} ({margin_city.min():,.2%})")


# 6. VISUALISATIONS

# PROFIT BY CITY [CITY PERFORMANCE]

plt.figure(figsize=(10,6))
city_profit.plot(kind='bar')
plt.title('Total Profit by City')
plt.ylabel('Profit (£)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# PROFIT BY PRODUCT [PRODUCT PERFORMANCE]

plt.figure(figsize=(10,6))
product_profit.plot(kind='bar', color='orange')
plt.title('Total Profit by Product')
plt.ylabel('Profit(£)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# HEATMAP (CITY vs PRODUCT)

pivot = df.pivot_table(values='Profit(£)',
                       index='City',
                       columns='Confectionary',
                       aggfunc='sum')

plt.figure(figsize=(12,6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap='coolwarm')
plt.title('Profit by City and Product')
plt.tight_layout()
plt.show()

# REVENUE vs COST [EFFICIENCY ANALYSIS]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Revenue(£)', y='Cost(£)', hue='City')
plt.title('Revenue vs Cost by City')
plt.tight_layout()
plt.show()


# PROFIT OVER TIME

time_profit = df.groupby('Date')['Profit(£)'].sum()

plt.figure(figsize=(10,6))
time_profit.plot()
plt.title('Profit Over Time')
plt.ylabel('Profit(£)')
plt.tight_layout()
plt.show()


# PROFIT MARGIN BY CITY

plt.figure(figsize=(10,6))
margin_city.plot(kind='bar', color='green')
plt.title('Average Profit Margin by City')
plt.ylabel('Profit Margin')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# EUROPE MAP USING CITY NAMES

city_summary = df.groupby('City').agg({
    'Revenue(£)': 'sum',
    'Profit(£)' : 'sum'
}).reset_index()

city_coords = {
    'London': [51.5074, -0.1278],
    'Bonn'  : [50.7373, 7.0982],
    'Napoli': [40.8518, 14.2681],
    'Paris' : [48.8566, 2.3522],
    'Seville': [37.3891, -5.9845]
}

city_summary['lat'] = city_summary['City'].map(lambda x: city_coords.get(x, [None, None])[0])
city_summary['lon'] = city_summary['City'].map(lambda x: city_coords.get(x, [None, None])[1])

city_summary = city_summary.dropna()

fig_map = px.scatter_geo(
    city_summary,
    lat='lat',
    lon='lon',
    size = 'Profit(£)',
    color = 'Revenue(£)',
    color_continuous_scale='viridis',
    hover_name='City',
    projection='natural earth',
    title='European Bakery Performance Overview'
) 

fig_map.update_traces(marker=dict(line=dict(width=1, color='black')))

fig_map.show()
