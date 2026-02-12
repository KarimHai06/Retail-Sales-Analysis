import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Information about DS
def summarize(DataFrame):
    summary = pd.DataFrame()
    summary['type'] = DataFrame.dtypes
    summary['sum_nan'] = DataFrame.isna().sum()
    summary['%_nan'] = round(DataFrame.isna().sum() / len(DataFrame) * 100, 2)
    summary['unique'] = DataFrame.apply(lambda x: ', '.join(map(str, x.unique())) if len(x.unique()) <= 8 else len(x.unique()))
    summary['mode'] = DataFrame.apply(lambda x: x.mode().iloc[0] if not x.mode().empty else '-')

    # Check that the data type is numeric (int or float) and NOT Boolean
    def is_numeric_non_bool(dtype):
        return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)

    # We apply numerical statistics only to suitable columns
    summary['max'] = DataFrame.apply(lambda x: x.max() if is_numeric_non_bool(x) else '-')
    summary['75_quantile'] = DataFrame.apply(lambda x: round(x.quantile(0.75), 2) if is_numeric_non_bool(x) else '-')
    summary['median'] = DataFrame.apply(lambda x: round(x.median(), 2) if is_numeric_non_bool(x) else '-')
    summary['25_quantile'] = DataFrame.apply(lambda x: round(x.quantile(0.25), 2) if is_numeric_non_bool(x) else '-')
    summary['min'] = DataFrame.apply(lambda x: x.min() if is_numeric_non_bool(x) else '-')
    summary['mean'] = DataFrame.apply(lambda x: round(x.mean(), 2) if is_numeric_non_bool(x) else '-')
    summary['standard_deviation'] = DataFrame.apply(lambda x: round(x.std(), 2) if is_numeric_non_bool(x) else '-')

    # Sorting by type
    summary['type'] = summary['type'].astype(str)
    type_order = ['int64', 'float64', 'bool', 'category', 'datetime64[ns]', 'object']
    summary['type'] = pd.Categorical(summary['type'], categories=type_order, ordered=True)
    summary = summary.sort_values(by='type')
    summary['type'] = summary['type'].astype(str)

    return summary

df = pd.read_csv('Data.csv')

# display(df.shape)
# summarize(df)


dupl = df[df.duplicated(subset=['Order ID', 'Product ID'], keep=False)]

# df[df.isnull().any(axis=1)]
# df[df['State'] == 'Vermont']
# df[df['Postal Code'] == 5401]
df.fillna(5401,inplace=True)

# Connect the split payments. Let's group all columns except Row_ID and Sales
group = [
    'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
    'Customer ID', 'Customer Name', 'Segment',
    'Country', 'City', 'State', 'Postal Code', 'Region',
    'Product ID', 'Category', 'Sub-Category', 'Product Name'
]
df = (
    df.groupby(group, as_index=False)
      .agg({'Sales': 'sum'})
)
df = df.reset_index(drop=True)
df.insert(0, 'Row_ID', df.index + 1)

for col in ['Order Date', 'Ship Date']:
    df[col] = pd.to_datetime(df[col], dayfirst=True)


# Analysis of sales dynamics
df_v1 = df.copy()

df_v1['Year'] = df_v1['Order Date'].dt.year
df_v1['Month'] = df_v1['Order Date'].dt.month

monthly_sales = (
    df_v1.groupby(['Year', 'Month'], as_index=False)['Sales']
      .sum()
)

pivot = (
    monthly_sales
    .pivot(index='Month', columns='Year', values='Sales')
    .sort_index()
)

fig = go.Figure()

for year in pivot.columns:
    fig.add_trace(
        go.Scatter(
            x=pivot.index,
            y=pivot[year],
            mode='lines+markers',
            name=str(year)
        )
    )

fig.update_layout(
    title='Sales Dynamics by Month',
    xaxis_title='Month',
    yaxis_title='Total Sales',
    xaxis=dict(
        tickmode='linear',
        tick0=1,
        dtick=1
    ),
    template='plotly_white',
    height=600,          
    margin=dict(l=40, r=40, t=60, b=60)  
)
fig.show()

# pivot.mean()

# Analysis by products and categories
cat = (
    df_v1.groupby('Category', as_index=False)['Sales']
      .sum()
      .sort_values('Sales', ascending=False)
)
cat['(%)'] = (cat['Sales'] / cat['Sales'].sum() * 100).round(2)


fig = go.Figure(go.Bar(
    x=cat['Category'],
    y=cat['Sales'],
))

fig.update_layout(
    title='Total Sales by Category',
    xaxis_title='Category',
    yaxis_title='Total Sales',
    template='plotly_white'
)

fig.show()


sub = (
    df_v1.groupby('Sub-Category', as_index=False)['Sales']
      .sum()
      .sort_values('Sales', ascending=False)
)

sub['cum_share'] = sub['Sales'].cumsum() / sub['Sales'].sum()
sub['color'] = np.where(sub['cum_share'] <= 0.8, 'green', 'lightgray')


fig1 = go.Figure(go.Bar(
    x=sub['Sales'],
    y=sub['Sub-Category'],
    orientation='h',
    marker_color=sub['color']
))

fig1.update_layout(
    title='Sub-Categories Contributing 80% of Total Sales',
    xaxis_title='Total Sales',
    yaxis_title='Sub-Category',
    template='plotly_white',
    height=650
)

fig1.show()


cat_year = (
    df_v1.groupby(['Year', 'Category'], as_index=False)['Sales']
      .sum()
)

sub_year = (
    df_v1.groupby(['Year', 'Sub-Category'], as_index=False)['Sales']
      .sum()
)
fig = go.Figure()

for category in cat_year['Category'].unique():
    data = cat_year[cat_year['Category'] == category]

    fig.add_trace(go.Scatter(
        x=data['Year'],
        y=data['Sales'],
        mode='lines+markers',
        name=category
    ))

fig.update_layout(
    title='Category Sales Trends Over Time',
    xaxis_title='Year',
    yaxis_title='Total Sales',
    template='plotly_white',
    xaxis=dict(tickmode='linear')
)

fig.show()


highlight = ['Machines', 'Tables']

fig = go.Figure()

for subcat in sub_year['Sub-Category'].unique():
    data = sub_year[sub_year['Sub-Category'] == subcat]

    color = 'red' if subcat in highlight else 'lightgray'
    width = 3 if subcat in highlight else 2
    opacity = 1 if subcat in highlight else 0.7

    fig.add_trace(go.Scatter(
        x=data['Year'],
        y=data['Sales'],
        mode='lines+markers',
        name=subcat,
        line=dict(color=color, width=width),
        opacity=opacity
    ))

fig.update_layout(
    title='Sub-Category Sales Trends by Year (Machines & Tables Show Decline)',
    xaxis_title='Year',
    yaxis_title='Total Sales',
    template='plotly_white',
    xaxis=dict(tickmode='linear'),
    height=600
)


# Customer Analysis¶
df_v1.groupby('Customer Name')['Customer ID'].nunique() \
  .sort_values(ascending=False).head(1)

# There are no duplicate names

segments = ['Consumer', 'Corporate', 'Home Office']
categories = ['Technology', 'Furniture', 'Office Supplies']
threshold = 0.2

fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=[f"{seg} - {cat}" for seg in segments for cat in categories],
    horizontal_spacing=0.08,
    vertical_spacing=0.12
)

for r, seg in enumerate(segments, start=1):
    df_seg = df_v1[df_v1['Segment'] == seg].copy()
    tmp = (
        df_seg.groupby('Customer Name', as_index=False)['Sales']
              .sum()
              .sort_values('Sales', ascending=False)
    )
    tmp['cum_share'] = tmp['Sales'].cumsum() / tmp['Sales'].sum()
    key_names = tmp.loc[tmp['cum_share'] <= threshold, 'Customer Name']

    df_key = df_seg[df_seg['Customer Name'].isin(key_names)].copy()

    for c, cat in enumerate(categories, start=1):
        sub = (
            df_key[df_key['Category'] == cat]
            .groupby('Sub-Category', as_index=False)['Sales']
            .sum()
            .sort_values('Sales', ascending=True)
        )

        if sub.empty:
            fig.add_trace(
                go.Bar(x=[], y=[], orientation='h', showlegend=False),
                row=r, col=c
            )
            continue

        fig.add_trace(
            go.Bar(
                x=sub['Sales'],
                y=sub['Sub-Category'],
                orientation='h',
                showlegend=False
            ),
            row=r, col=c
        )

        fig.update_xaxes(title_text="Sales", row=r, col=c)
        fig.update_yaxes(title_text="", row=r, col=c)

fig.update_layout(
    title=f"Top-Tier Customers (Top 20% of Sales) by Sub-Category (Segment × Category)",
    template="plotly_white",
    height=1100,
    width=1200
)

fig.show()


fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=[f"{seg} - {cat}" for seg in segments for cat in categories],
    horizontal_spacing=0.08,
    vertical_spacing=0.12
)

for r, seg in enumerate(segments, start=1):
    df_seg = df_v1[df_v1['Segment'] == seg].copy()

    tmp = (
        df_seg.groupby('Customer Name', as_index=False)['Sales']
              .sum()
              .sort_values('Sales', ascending=False)
    )
    tmp['cum_share'] = tmp['Sales'].cumsum() / tmp['Sales'].sum()

    key_names = tmp.loc[
        (tmp['cum_share'] > 0.2) & (tmp['cum_share'] <= 0.8),
        'Customer Name'
    ]

    df_mid = df_seg[df_seg['Customer Name'].isin(key_names)].copy()

    for c, cat in enumerate(categories, start=1):
        sub = (
            df_mid[df_mid['Category'] == cat]
            .groupby('Sub-Category', as_index=False)['Sales']
            .sum()
            .sort_values('Sales', ascending=True)
        )

        fig.add_trace(
            go.Bar(
                x=sub['Sales'],
                y=sub['Sub-Category'],
                orientation='h',
                showlegend=False
            ),
            row=r, col=c
        )

fig.update_layout(
    title=f"Mid-Tier Customers (20–80% of Sales) by Sub-Category (Segment × Category)",
    template="plotly_white",
    height=1100,
    width=1200
)

fig.show()


fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=[f"{seg} - {cat}" for seg in segments for cat in categories],
    horizontal_spacing=0.08,
    vertical_spacing=0.12
)

for r, seg in enumerate(segments, start=1):
    df_seg = df_v1[df_v1['Segment'] == seg].copy()

    tmp = (
        df_seg.groupby('Customer Name', as_index=False)['Sales']
              .sum()
              .sort_values('Sales', ascending=False)
    )
    tmp['cum_share'] = tmp['Sales'].cumsum() / tmp['Sales'].sum()

    tail_names = tmp.loc[tmp['cum_share'] > 0.8, 'Customer Name']

    df_tail = df_seg[df_seg['Customer Name'].isin(tail_names)].copy()

    for c, cat in enumerate(categories, start=1):
        sub = (
            df_tail[df_tail['Category'] == cat]
            .groupby('Sub-Category', as_index=False)['Sales']
            .sum()
            .sort_values('Sales', ascending=True)
        )

        fig.add_trace(
            go.Bar(
                x=sub['Sales'],
                y=sub['Sub-Category'],
                orientation='h',
                showlegend=False
            ),
            row=r, col=c
        )

fig.update_layout(
    title=f"Lower-Tier Customers (80–100% of Sales) by Sub-Category (Segment × Category)",
    template="plotly_white",
    height=1100,
    width=1200
)

fig.show()


machines_freq = (
    df_v1[df_v1['Sub-Category'] == 'Machines']
    .groupby('Year')
    .size()
    .reset_index(name='Number of Machines Sold by Year')
)
# print(machines_freq)

# Geographical analysis
region_cat = (
    df_v1.groupby(['Region', 'Category'], as_index=False)['Sales']
      .sum()
)

pivot_rc = (
    region_cat.pivot(index='Region', columns='Category', values='Sales')
              .fillna(0)
)

categories = ['Technology', 'Furniture', 'Office Supplies']

fig = go.Figure()
for cat in categories:
    if cat in pivot_rc.columns:
        fig.add_trace(go.Bar(
            x=pivot_rc.index,
            y=pivot_rc[cat],
            name=cat
        ))

fig.update_layout(
    title="Sales by Category and Region",
    xaxis_title="Region",
    yaxis_title="Total Sales",
    barmode="stack",
    template="plotly_white",
    height=550
)

fig.show()


state_cat = (
    df_v1.groupby(['State', 'Category'], as_index=False)['Sales']
      .sum()
)

pivot_sc = (
    state_cat
    .pivot(index='State', columns='Category', values='Sales')
    .fillna(0)
)

share_sc = pivot_sc.div(pivot_sc.sum(axis=1), axis=0)

states_50 = (
    pd.DataFrame({
        'Dominant Category': share_sc.idxmax(axis=1),
        'Dominant Share': share_sc.max(axis=1),
        'Total Sales': pivot_sc.sum(axis=1)
    })
    .query('`Dominant Share` >= 0.5')
    .sort_values(['Dominant Share', 'Total Sales'], ascending=False)
)

top_states = states_50.sort_values('Total Sales', ascending=False).index
share_top = share_sc.loc[top_states] * 100

fig = go.Figure()
for cat in categories:
    if cat in share_top.columns:
        fig.add_trace(go.Bar(
            x=share_top.index,
            y=share_top[cat],
            name=cat
        ))

fig.update_layout(
    title="States Where One Category Accounts for More Than 50% of Sales",
    xaxis_title="State",
    yaxis_title="Share of Sales (%)",
    barmode="stack",
    template="plotly_white",
    height=650,
    xaxis=dict(tickangle=-45),
    yaxis=dict(ticksuffix='%')
)

fig.show()


regions = df_v1['Region'].unique()
region = regions[0]

tmp = (
    df_v1[df_v1['Region'] == region]
    .groupby(['City', 'Category'], as_index=False)['Sales']
    .sum()
)
pivot = (
    tmp.pivot(index='City', columns='Category', values='Sales')
       .fillna(0)
)
pivot['Total'] = pivot.sum(axis=1)
pivot = (
    pivot[pivot['Total'] > 10000]
    .sort_values('Total', ascending=False)
    .drop(columns='Total')
)

fig = go.Figure()
for cat in categories:
    if cat in pivot.columns:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[cat],
            name=cat
        ))

fig.update_layout(
    title=f"Sales by City and Category (Total > 10,000) - {region}",
    xaxis_title="City",
    yaxis_title="Sales",
    barmode="group",
    template="plotly_white",
    height=650,
    xaxis=dict(tickangle=-45)
)

fig.show()


region = regions[1]

tmp = (
    df_v1[df_v1['Region'] == region]
    .groupby(['City', 'Category'], as_index=False)['Sales']
    .sum()
)
pivot = (
    tmp.pivot(index='City', columns='Category', values='Sales')
       .fillna(0)
)
pivot['Total'] = pivot.sum(axis=1)
pivot = (
    pivot[pivot['Total'] > 10000]
    .sort_values('Total', ascending=False)
    .drop(columns='Total')
)

fig = go.Figure()
for cat in categories:
    if cat in pivot.columns:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[cat],
            name=cat
        ))

fig.update_layout(
    title=f"Sales by City and Category (Total > 10,000) - {region}",
    xaxis_title="City",
    yaxis_title="Sales",
    barmode="group",
    template="plotly_white",
    height=650,
    xaxis=dict(tickangle=-45)
)


fig.show()


region = regions[2]

tmp = (
    df_v1[df_v1['Region'] == region]
    .groupby(['City', 'Category'], as_index=False)['Sales']
    .sum()
)
pivot = (
    tmp.pivot(index='City', columns='Category', values='Sales')
       .fillna(0)
)
pivot['Total'] = pivot.sum(axis=1)
pivot = (
    pivot[pivot['Total'] > 10000]
    .sort_values('Total', ascending=False)
    .drop(columns='Total')
)

fig = go.Figure()
for cat in categories:
    if cat in pivot.columns:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[cat],
            name=cat
        ))

fig.update_layout(
    title=f"Sales by City and Category (Total > 10,000) - {region}",
    xaxis_title="City",
    yaxis_title="Sales",
    barmode="group",
    template="plotly_white",
    height=650,
    xaxis=dict(tickangle=-45)
)

fig.show()


region = regions[3]

tmp = (
    df_v1[df_v1['Region'] == region]
    .groupby(['City', 'Category'], as_index=False)['Sales']
    .sum()
)
pivot = (
    tmp.pivot(index='City', columns='Category', values='Sales')
       .fillna(0)
)
pivot['Total'] = pivot.sum(axis=1)
pivot = (
    pivot[pivot['Total'] > 10000]
    .sort_values('Total', ascending=False)
    .drop(columns='Total')
)

fig = go.Figure()
for cat in categories:
    if cat in pivot.columns:
        fig.add_trace(go.Bar(
            x=pivot.index,
            y=pivot[cat],
            name=cat
        ))

fig.update_layout(
    title=f"Sales by City and Category (Total > 10,000) - {region}",
    xaxis_title="City",
    yaxis_title="Sales",
    barmode="group",
    template="plotly_white",
    height=650,
    xaxis=dict(tickangle=-45)
)

fig.show()


# Logistics analysis
ship_order = ['Standard Class', 'Second Class', 'First Class', 'Same Day']

df_v1['delivery_days'] = (df_v1['Ship Date'] - df_v1['Order Date']).dt.days
df_ok = df_v1[df_v1['delivery_days'] >= 0].copy()

fig = px.box(
    df_ok,
    x='Ship Mode',
    y='delivery_days',
    category_orders={'Ship Mode': ship_order},
    title='Delivery Time (Days) by Shipping Mode'
)
fig.show()


region_ship = (
    df_ok.groupby(['Region', 'Ship Mode'])
        .agg(avg_delivery_days=('delivery_days', 'mean'))
        .reset_index()
)

heatmap_data = (
    region_ship
    .pivot(index='Region', columns='Ship Mode', values='avg_delivery_days')
    .reindex(columns=ship_order)
)

fig = px.imshow(
    heatmap_data,
    aspect='auto',
    title='Average Delivery Time by Region and Shipping Mode'
)
fig.show()


label = ['total amount','average amount']
col = ['total_sales', 'avg_order_sales']

ship_sales = (
    df_v1.groupby('Ship Mode')
      .agg(
          total_sales=('Sales', 'sum'),
          orders=('Order ID', 'nunique'),
          avg_order_sales=('Sales', 'mean')
      )
      .reset_index()
)

ship_sales['sales_share'] = ship_sales['total_sales'] / ship_sales['total_sales'].sum()

ship_sales[['total_sales','avg_order_sales','sales_share']] = (
    ship_sales[['total_sales','avg_order_sales','sales_share']].round(2)
)

ship_sales['Ship Mode'] = pd.Categorical(
    ship_sales['Ship Mode'],
    categories=ship_order,
    ordered=True
)
ship_sales = ship_sales.sort_values('Ship Mode')

fig = px.bar(
    ship_sales,
    x='Ship Mode',
    y='total_sales',
    title='Total Sales by Shipping Mode'
)
fig.show()


fig = px.bar(
    ship_sales,
    x='Ship Mode',
    y='avg_order_sales',
    title='Average Sales by Shipping Mode'
)
fig.show()


data = (
        df_v1.groupby(['Segment', 'Ship Mode'])
      .agg(total_sales=('Sales', 'sum'))
      .reset_index()
)

data['Ship Mode'] = pd.Categorical(
    data['Ship Mode'],
    categories=ship_order,
    ordered=True
)

data['share_within_ship'] = (
    data['total_sales'] /
    data.groupby('Ship Mode', observed=True)['total_sales'].transform('sum')
)

data = data.sort_values('Ship Mode')

fig = px.bar(
    data,
    x='Ship Mode',
    y='total_sales',
    color='Segment',
    title=f'Total Sales by Shipping Mode and Segment',
    hover_data={
        'total_sales': ':.2f',
        'share_within_ship': ':.1%',
        'Ship Mode': False
    }
)

fig.show()


data = (
        df_v1.groupby(['Category', 'Ship Mode'])
          .agg(total_sales=('Sales', 'sum'))
          .reset_index()
)

data['Ship Mode'] = pd.Categorical(
    data['Ship Mode'],
    categories=ship_order,
    ordered=True
)

data['share_within_ship'] = (
    data['total_sales'] /
    data.groupby('Ship Mode', observed=True)['total_sales'].transform('sum')
)

data = data.sort_values('Ship Mode')

fig = px.bar(
    data,
    x='Ship Mode',
    y='total_sales',
    color='Category',
    title=f'Total Sales by Shipping Mode and Category',
    hover_data={
        'total_sales': ':.2f',
        'share_within_ship': ':.1%',
        'Ship Mode': False
    }
)

fig.show()