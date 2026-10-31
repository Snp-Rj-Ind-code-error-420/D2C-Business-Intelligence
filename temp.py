import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="D2C Business Intelligence", page_icon="📊", layout="wide")

DATA_PATH = Path(__file__).parent / "left_df2.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for col in ["quantity", "unit_price", "discount_pct", "delivery_days",
                "net_revenue", "cost_price", "gross_profit", "margin_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def options(df, col):
    return sorted(df[col].dropna().unique().tolist(), key=str) if col in df.columns else []

def unique_orders(df):
    return df["order_id"].nunique() if "order_id" in df.columns else len(df)

def order_level(df):
    """Aggregate item rows into one row per order for order-level metrics."""
    if "order_id" not in df.columns:
        return df.copy()
    agg = {}
    for col in ["order_date", "customer_id", "payment_method", "channel",
                "delivery_days", "order_status", "city", "segment"]:
        if col in df.columns:
            agg[col] = "first"
    for col in ["net_revenue", "gross_profit", "quantity"]:
        if col in df.columns:
            agg[col] = "sum"
    return df.groupby("order_id", as_index=False).agg(agg) if agg else df.drop_duplicates("order_id")

def money(x):
    """Format a number as Indian rupees without decimal places."""
    return "—" if pd.isna(x) else f"₹{format_indian(x)}"


def format_indian(x):
    """Format a number using Indian digit grouping, preserving two decimals."""
    s = f"{x:.2f}"
    int_part, dec_part = s.split(".")

    if len(int_part) <= 3:
        return s

    last_three = int_part[-3:]
    remaining = int_part[:-3]

    groups = []
    while remaining:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]

    return ",".join(groups) + "," + last_three + "." + dec_part

def apply_global_filters(df, start, end, categories, channels, cities):
    out = df.copy()
    if "order_date" in out.columns:
        if start:
            out = out[out["order_date"].dt.date >= start]
        if end:
            out = out[out["order_date"].dt.date <= end]
    for col, selected in [("category", categories), ("channel", channels), ("city", cities)]:
        if selected and col in out.columns:
            out = out[out[col].isin(selected)]
    return out

def kpis(df):
    revenue = df["net_revenue"].sum() if "net_revenue" in df.columns else float("nan")
    profit = df["gross_profit"].sum() if "gross_profit" in df.columns else float("nan")
    margin = profit / revenue * 100 if pd.notna(revenue) and revenue else float("nan")
    orders = unique_orders(df)
    aov = revenue / orders if orders else float("nan")
    cols = st.columns(4)
    cols[0].metric("Net Revenue", money(revenue))
    cols[1].metric("Gross Profit", money(profit))
    cols[2].metric("Gross Margin", "—" if pd.isna(margin) else f"{margin:.1f}%")
    cols[3].metric("AOV", money(aov))

if not DATA_PATH.exists():
    st.title("📊 D2C Business Intelligence")
    st.error("Dataset not found. Put your prepared `left_df2.csv` next to `app.py`.")
    st.code("left_df2.to_csv('left_df2.csv', index=False)")
    st.stop()

df = load_data(str(DATA_PATH))
if df.empty:
    st.error("The dataset is empty.")
    st.stop()

st.title("📊 D2C Business Intelligence")
st.caption("Financial performance • Operational diagnostics • Customer insights")

pages = ["Financial Review", "Returns Analysis", "Delivery & Payments", "Customer Insights"]
page = st.radio("Dashboard section", pages, horizontal=True, label_visibility="collapsed")
st.divider()

# Shared top filter bar: same position on every page
st.markdown("#### Global filters")
if st.button("Reset all filters"):
    for key in [
        "global_dates", "global_categories", "global_channels", "global_cities",
        "fin_segment", "fin_payment", "ret_reason", "ret_subcat",
        "ops_payment", "ops_status", "cust_segment",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

f1, f2, f3, f4 = st.columns([1.6, 1.3, 1.3, 1.3])

if "order_date" in df.columns and df["order_date"].notna().any():
    min_date, max_date = df["order_date"].min().date(), df["order_date"].max().date()
    dates = f1.date_input("Order date range", value=(min_date, max_date),
                          min_value=min_date, max_value=max_date, key="global_dates")
    if isinstance(dates, (tuple, list)) and len(dates) == 2:
        start_date, end_date = dates
    elif isinstance(dates, (tuple, list)) and len(dates) == 1:
        start_date = end_date = dates[0]
    else:
        start_date, end_date = min_date, max_date
else:
    f1.caption("Date unavailable")
    start_date = end_date = None

categories = f2.multiselect("Category", options(df, "category"),
                            placeholder="All categories", key="global_categories")
channels = f3.multiselect("Channel", options(df, "channel"),
                          placeholder="All channels", key="global_channels")
cities = f4.multiselect("City", options(df, "city"),
                        placeholder="All cities", key="global_cities")

filtered = apply_global_filters(df, start_date, end_date, categories, channels, cities)

# Page-specific filters stay in the same expander position.
with st.expander("Advanced filters", expanded=False):
    c1, c2, c3 = st.columns(3)
    if page == "Financial Review":
        segments = c1.multiselect("Customer segment", options(filtered, "segment"), key="fin_segment")
        payments = c2.multiselect("Payment method", options(filtered, "payment_method"), key="fin_payment")
        if segments and "segment" in filtered: filtered = filtered[filtered["segment"].isin(segments)]
        if payments and "payment_method" in filtered: filtered = filtered[filtered["payment_method"].isin(payments)]
    elif page == "Returns Analysis":
        reasons = c1.multiselect("Return reason", options(filtered, "return_reason"), key="ret_reason")
        subcats = c2.multiselect("Subcategory", options(filtered, "sub_category"), key="ret_subcat")
        if reasons and "return_reason" in filtered: filtered = filtered[filtered["return_reason"].isin(reasons)]
        if subcats and "sub_category" in filtered: filtered = filtered[filtered["sub_category"].isin(subcats)]
    elif page == "Delivery & Payments":
        payments = c1.multiselect("Payment method", options(filtered, "payment_method"), key="ops_payment")
        statuses = c2.multiselect("Order status", options(filtered, "order_status"), key="ops_status")
        if payments and "payment_method" in filtered: filtered = filtered[filtered["payment_method"].isin(payments)]
        if statuses and "order_status" in filtered: filtered = filtered[filtered["order_status"].isin(statuses)]
    else:
        segments = c1.multiselect("Customer segment", options(filtered, "segment"), key="cust_segment")
        if segments and "segment" in filtered: filtered = filtered[filtered["segment"].isin(segments)]

st.caption(f"{len(filtered):,} item rows · {unique_orders(filtered):,} unique orders after filters")
st.divider()

if page == "Financial Review":
    st.header("Financial Performance Review")
    kpis(filtered)

    st.subheader("Monthly Revenue Trend")
    if {"order_date", "net_revenue"}.issubset(filtered.columns):
        monthly = (filtered.dropna(subset=["order_date"])
                   .assign(month=lambda x: x["order_date"].dt.to_period("M").dt.to_timestamp())
                   .groupby("month", as_index=False)["net_revenue"].sum())
        st.plotly_chart(px.line(monthly, x="month", y="net_revenue", markers=True,
                                labels={"month": "Month", "net_revenue": "Net Revenue"}),
                        width='stretch')
    else:
        st.info("Requires order_date and net_revenue columns.")

    left, right = st.columns(2)
    with left:
        st.subheader("Revenue vs Gross Profit")
        if {"net_revenue", "gross_profit"}.issubset(filtered.columns):
            totals = pd.DataFrame({"Metric": ["Net Revenue", "Gross Profit"],
                                   "Amount": [filtered["net_revenue"].sum(), filtered["gross_profit"].sum()]})
            st.plotly_chart(px.bar(totals, x="Metric", y="Amount"), width='stretch')
        else:
            st.info("Requires net_revenue and gross_profit.")
    with right:
        st.subheader("Revenue by Category")
        if {"category", "net_revenue"}.issubset(filtered.columns):
            data = filtered.groupby("category", as_index=False)["net_revenue"].sum()
            st.plotly_chart(px.bar(data, x="category", y="net_revenue"), width='stretch')
        else:
            st.info("Requires category and net_revenue.")

    left, right = st.columns(2)
    with left:
        st.subheader("Margin by Category")
        if {"category", "net_revenue", "gross_profit"}.issubset(filtered.columns):
            data = filtered.groupby("category", as_index=False).agg(
                revenue=("net_revenue", "sum"), profit=("gross_profit", "sum"))
            data["margin_pct"] = data["profit"] / data["revenue"] * 100
            st.plotly_chart(px.bar(data, x="category", y="margin_pct",
                                   labels={"margin_pct": "Gross Margin (%)"}), width='stretch')
        else:
            st.info("Requires category, net_revenue, and gross_profit.")
    with right:
        st.subheader("Revenue by Channel")
        if {"channel", "net_revenue"}.issubset(filtered.columns):
            data = filtered.groupby("channel", as_index=False)["net_revenue"].sum()
            st.plotly_chart(px.bar(data, x="channel", y="net_revenue"), width='stretch')
        else:
            st.info("Requires channel and net_revenue.")

    st.subheader("Customer Segment Performance")
    if {"segment", "net_revenue"}.issubset(filtered.columns):
        data = filtered.groupby("segment", as_index=False)["net_revenue"].sum()
        st.plotly_chart(px.bar(data, x="segment", y="net_revenue"), width='stretch')
    else:
        st.info("Requires segment and net_revenue.")

elif page == "Returns Analysis":
    st.header("Returns Analysis")
    if "return_id" in filtered.columns:
        returned = filtered[filtered["return_id"].notna()]
        total_orders = unique_orders(filtered)
        returned_orders = unique_orders(returned)
        rate = returned_orders / total_orders * 100 if total_orders else float("nan")
        a, b, c = st.columns(3)
        a.metric("Unique Orders", f"{total_orders:,}")
        b.metric("Returned Orders", f"{returned_orders:,}")
        c.metric("Return Rate", "—" if pd.isna(rate) else f"{rate:.2f}%")
        st.warning("Returns are attributed at order level. Do not treat this as an item-level return rate unless returned product IDs are available.")
        if "return_reason" in returned.columns:
            reason_data = (returned.dropna(subset=["return_reason"])
                           .groupby("return_reason", as_index=False)["order_id"].nunique()
                           .rename(columns={"order_id": "returned_orders"}))
            if not reason_data.empty:
                st.subheader("Return Reasons")
                st.plotly_chart(px.bar(reason_data, x="return_reason", y="returned_orders"),
                                width='stretch')
    else:
        st.info("This page requires return fields such as return_id and return_reason. Use a prepared returns-merged dataset if they are absent.")

elif page == "Delivery & Payments":
    st.header("Delivery & Payments")
    odf = order_level(filtered)
    a, b, c = st.columns(3)
    a.metric("Unique Orders", f"{unique_orders(filtered):,}")
    if "delivery_days" in odf.columns:
        avg_days = odf["delivery_days"].mean()
        b.metric("Average Delivery Time", "—" if pd.isna(avg_days) else f"{avg_days:.2f} days")
    else:
        b.metric("Average Delivery Time", "—")
    if "order_status" in odf.columns:
        cancelled = odf["order_status"].astype(str).str.lower().str.contains("cancel").sum()
        total = odf["order_id"].nunique() if "order_id" in odf else len(odf)
        c.metric("Cancellation Rate", f"{cancelled / total * 100:.2f}%" if total else "—")
    else:
        c.metric("Cancellation Rate", "—")

    left, right = st.columns(2)
    with left:
        st.subheader("Average Delivery Days by City")
        if {"city", "delivery_days"}.issubset(odf.columns):
            city_data = odf.groupby("city", as_index=False)["delivery_days"].mean()
            st.plotly_chart(px.bar(city_data, x="city", y="delivery_days"), width='stretch')
        else:
            st.info("Requires city and delivery_days.")
    with right:
        st.subheader("Orders by Payment Method")
        if "payment_method" in odf.columns and "order_id" in odf.columns:
            pay = odf.groupby("payment_method", as_index=False)["order_id"].nunique().rename(columns={"order_id": "orders"})
            st.plotly_chart(px.bar(pay, x="payment_method", y="orders"), width='stretch')
        else:
            st.info("Requires payment_method and order_id.")
    st.caption("These comparisons are descriptive; they do not establish causation.")

elif page == "Customer Insights":
    st.header("Customer Insights")
    odf = order_level(filtered)
    if {"customer_id", "order_id"}.issubset(odf.columns):
        per_customer = odf.groupby("customer_id")["order_id"].nunique()
        repeat_rate = per_customer.gt(1).mean() * 100 if len(per_customer) else float("nan")
        a, b, c = st.columns(3)
        a.metric("Unique Customers", f"{odf['customer_id'].nunique():,}")
        b.metric("Repeat Customer Rate", f"{repeat_rate:.2f}%" if pd.notna(repeat_rate) else "—")
        c.metric("Orders per Customer", f"{per_customer.mean():.2f}" if len(per_customer) else "—")
    else:
        st.info("Requires customer_id and order_id.")

    left, right = st.columns(2)
    with left:
        st.subheader("Revenue by Customer Segment")
        if {"segment", "net_revenue"}.issubset(filtered.columns):
            data = filtered.groupby("segment", as_index=False)["net_revenue"].sum()
            st.plotly_chart(px.bar(data, x="segment", y="net_revenue"), width='stretch')
        else:
            st.info("Requires segment and net_revenue.")
    with right:
        st.subheader("Customer Order Frequency")
        if {"customer_id", "order_id"}.issubset(odf.columns):
            freq = odf.groupby("customer_id")["order_id"].nunique().value_counts().sort_index()
            freq = freq.rename_axis("orders_per_customer").reset_index(name="customers")
            st.plotly_chart(px.bar(freq, x="orders_per_customer", y="customers"), width='stretch')
        else:
            st.info("Requires customer_id and order_id.")

st.divider()
st.caption("Validate all metrics against your notebook. Item-level revenue and profit should be summed from line items; order-level KPIs should use unique orders.")
