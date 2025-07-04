import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load & Inspect Dataset
# -------------------------------
df = pd.read_csv('train.csv')
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

# -------------------------------
# Date Conversion & Feature Engineering
# -------------------------------
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
df['Year'] = df['Order Date'].dt.year

# -------------------------------
# Monthly Sales
# -------------------------------
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
print("\nMonthly Sales:")
print(monthly_sales)

monthly_sales.plot(
    x='Month', y='Sales', kind='line',
    title='Monthly Sales Trend', figsize=(12, 5),
    legend=False, rot=45
)
plt.tight_layout()
plt.show()

# -------------------------------
# Top 10 Products
# -------------------------------
top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Best-Selling Products:\n", top_products)

top_products.plot(
    kind='bar', title='Top 10 Best-Selling Products',
    figsize=(12, 5), rot=45
)
plt.tight_layout()
plt.show()

# -------------------------------
# Sales by Region
# -------------------------------
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
print("\nSales by Region:\n", region_sales)

region_sales.plot(
    kind='bar', title='Sales by Region',
    figsize=(8, 5)
)
plt.tight_layout()
plt.show()

# -------------------------------
# Top 10 Customers
# -------------------------------
top_customers = df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Customers by Sales:\n", top_customers)

top_customers.plot(
    kind='barh', title='Top 10 Customers by Sales',
    figsize=(10, 6)
)
plt.tight_layout()
plt.show()

# -------------------------------
# Sales by Category/Sub-Category
# -------------------------------
cat_subcat_sales = df.groupby(['Category', 'Sub-Category'])['Sales'].sum().sort_values(ascending=False)
print("\nSales by Category/Sub-Category:\n", cat_subcat_sales)

cat_plot = df.groupby(['Category', 'Sub-Category'])['Sales'].sum().unstack()
cat_plot.plot(
    kind='bar', stacked=True,
    figsize=(12, 6), title="Sales by Category and Sub-Category"
)
plt.tight_layout()
plt.show()

# -------------------------------
# Average Sales per Order
# -------------------------------
avg_order_sales = df.groupby('Order ID')['Sales'].sum().mean()
print(f"\nAverage sales per order: ${avg_order_sales:.2f}")

# -------------------------------
# Top 10 Cities by Sales
# -------------------------------
top_cities = df.groupby('City')['Sales'].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Cities by Sales:\n", top_cities)

top_cities.plot(
    kind='bar', title='Top 10 Cities by Sales',
    figsize=(10, 5), rot=45
)
plt.tight_layout()
plt.show()

# -------------------------------
# Repeat Customers
# -------------------------------
repeat_customers = df['Customer ID'].value_counts()
frequent_buyers = repeat_customers[repeat_customers > 1].count()
print(f"\nNumber of repeat customers: {frequent_buyers}")

# -------------------------------
# Yearly Sales
# -------------------------------
yearly_sales = df.groupby('Year')['Sales'].sum()
print("\nYearly Sales:\n", yearly_sales)

yearly_sales.plot(
    kind='bar', title='Yearly Sales',
    figsize=(8, 5)
)
plt.tight_layout()
plt.show()
