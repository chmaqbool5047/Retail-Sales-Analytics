import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title="Retail Sales Analytics", layout="wide")
st.title("🛒 Retail Sales Dashboard & ML Insights")


# Database se data load krna
@st.cache_data
def load_data():
  conn = sqlite3.connect("sales_analytics.db")
  df = pd.read_sql_query("SELECT * FROM sales", conn)
  conn.close()
  df["Date"] = pd.to_datetime(df["Date"])
  return df


df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filters")
categories = st.sidebar.multiselect(
    "Category:",
    options=df["Product Category"].unique(),
    default=df["Product Category"].unique(),
)
genders = st.sidebar.multiselect(
    "Gender:", options=df["Gender"].unique(), default=df["Gender"].unique()
)

filtered_df = df[
    (df["Product Category"].isin(categories)) & (df["Gender"].isin(genders))
]

# KPIs
k1, k2, k3 = st.columns(3)
k1.metric("Total Revenue", f"${filtered_df['Total Amount'].sum():,.2f}")
k2.metric("Total Transactions", f"{len(filtered_df):,}")
k3.metric("Avg Order Value", f"${filtered_df['Total Amount'].mean():,.2f}")

st.markdown("---")

# Visualizations
c1, c2 = st.columns(2)
with c1:
  cat_sales = (
      filtered_df.groupby("Product Category")["Total Amount"]
      .sum()
      .reset_index()
  )
  fig_cat = px.bar(
      cat_sales,
      x="Product Category",
      y="Total Amount",
      color="Product Category",
      title="Revenue by Category",
  )
  st.plotly_chart(fig_cat, use_container_width=True)

with c2:
  gender_sales = (
      filtered_df.groupby(["Gender", "Product Category"])["Total Amount"]
      .sum()
      .reset_index()
  )
  fig_gen = px.bar(
      gender_sales,
      x="Product Category",
      y="Total Amount",
      color="Gender",
      barmode="group",
      title="Sales by Gender & Category",
  )
  st.plotly_chart(fig_gen, use_container_width=True)

st.markdown("---")

# ML Model
st.subheader("🤖 Predict Customer Spending")

ml_df = pd.get_dummies(
    df.copy(), columns=["Product Category", "Gender"], drop_first=True
)
X = ml_df.drop(
    columns=["Transaction ID", "Date", "Customer ID", "Total Amount"]
)
y = ml_df["Total Amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

col1, col2, col3 = st.columns(3)
age = col1.number_input("Age", 18, 80, 30)
qty = col2.number_input("Quantity", 1, 10, 2)
price = col3.number_input("Price per Unit ($)", 10, 1000, 50)

cat_opt = st.selectbox("Product Category", df["Product Category"].unique())
gen_opt = st.selectbox("Gender", df["Gender"].unique())

if st.button("Predict"):
  input_data = pd.DataFrame(
      [[
          age,
          qty,
          price,
          1 if cat_opt == "Clothing" else 0,
          1 if cat_opt == "Electronics" else 0,
          1 if gen_opt == "Male" else 0,
      ]],
      columns=X.columns,
  )
  pred = model.predict(input_data)[0]
  st.success(f"Predicted Total Spent: **${pred:.2f}**")