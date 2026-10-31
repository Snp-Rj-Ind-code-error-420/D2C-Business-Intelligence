from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="D2C Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
ITEM_DATA_PATH = BASE_DIR / "left_df2.csv"
RETURN_DATA_CANDIDATES = [
    BASE_DIR / "left_df3.csv",   # optional order-level returns-merged extract
    BASE_DIR / "returns.csv",    # optional original returns table
]

NAV_GROUPS = {
    "Financial Review": [
        "Overview",
        "Products & Categories",
        "Customers & Segments",
        "Discount Analysis",
    ],
    "Operational Diagnostics": [
        "Overview",
        "Returns",
        "Delivery & Payments",
        "Customer Retention",
    ],
}

NUMERIC_COLUMNS = [
    "quantity", "unit_price", "discount_pct", "delivery_days",
    "net_revenue", "cost_price", "gross_profit", "margin_pct",
]
DATE_COLUMNS = ["order_date", "signup_date", "return_date"]


# -------------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------------
@st.cache_data(show_spinner="Loading prepared data...")
def load_csv(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    for col in DATE_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def available_options(data: pd.DataFrame, column: str) -> list:
    if column not in data.columns:
        return []
    return sorted(data[column].dropna().unique().tolist(), key=lambda value: str(value))


def money(value) -> str:
    if pd.isna(value):
        return "—"
    # Python's Indian locale grouping is not built in, so format explicitly.
    value = float(value)
    sign = "-" if value < 0 else ""
    rounded = f"{abs(value):.0f}"
    if len(rounded) > 3:
        rounded = rounded[:-3] + "," + rounded[-3:]
        head, tail = rounded.rsplit(",", 1)
        chunks = []
        while len(head) > 2:
            chunks.insert(0, head[-2:])
            head = head[:-2]
        if head:
            chunks.insert(0, head)
        rounded = ",".join(chunks + [tail])
    return f"{sign}₹{rounded}"


def number(value, decimals=0) -> str:
    return "—" if pd.isna(value) else f"{value:,.{decimals}f}"


def pct(numerator, denominator):
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return float("nan")
    return numerator / denominator * 100


def unique_orders(data: pd.DataFrame) -> int:
    return int(data["order_id"].nunique()) if "order_id" in data.columns else len(data)


def order_level(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse line items to one row/order for order-level metrics."""
    if "order_id" not in data.columns:
        return data.copy()
    agg = {}
    for col in [
        "order_date", "customer_id", "payment_method", "channel", "delivery_days",
        "order_status", "city", "segment", "signup_date",
    ]:
        if col in data.columns:
            agg[col] = "first"
    for col in ["net_revenue", "gross_profit", "quantity"]:
        if col in data.columns:
            agg[col] = "sum"
    if not agg:
        return data.drop_duplicates("order_id").copy()
    return data.groupby("order_id", as_index=False, dropna=False).agg(agg)


def prepare_returns(item_data: pd.DataFrame) -> pd.DataFrame | None:
    """Load a return table separately so return joins never inflate financial totals."""
    if "return_id" in item_data.columns and "order_id" in item_data.columns:
        source = item_data.copy()
    else:
        source = None
        for path in RETURN_DATA_CANDIDATES:
            if path.exists():
                candidate = load_csv(str(path))
                if "order_id" in candidate.columns and (
                    "return_id" in candidate.columns or "return_reason" in candidate.columns
                ):
                    source = candidate
                    break
    if source is None or "order_id" not in source.columns:
        return None

    keep = ["order_id"]
    for col in ["return_id", "return_date", "return_reason", "refund_status"]:
        if col in source.columns:
            keep.append(col)
    returns = source[keep].copy()
    returns = returns[returns["order_id"].notna()]
    if "return_id" in returns.columns:
        returns = returns[returns["return_id"].notna()]
    # One row per order is enough for order-return rate. Keep the first recorded
    # reason/date/status for the order; reason counts below use the source records.
    return returns.drop_duplicates("order_id")


def return_reason_records(item_data: pd.DataFrame) -> pd.DataFrame | None:
    if "return_reason" in item_data.columns and "order_id" in item_data.columns:
        source = item_data.copy()
    else:
        source = None
        for path in RETURN_DATA_CANDIDATES:
            if path.exists():
                candidate = load_csv(str(path))
                if "order_id" in candidate.columns and "return_reason" in candidate.columns:
                    source = candidate
                    break
    if source is None:
        return None
    cols = [c for c in ["order_id", "return_id", "return_reason", "return_date", "refund_status"] if c in source]
    source = source[cols].dropna(subset=["order_id", "return_reason"]).copy()
    if "return_id" in source.columns:
        source = source[source["return_id"].notna()]
    return source.drop_duplicates()


def apply_filters(
    data: pd.DataFrame,
    start_date,
    end_date,
    categories: list,
    channels: list,
    cities: list,
) -> pd.DataFrame:
    out = data.copy()
    if "order_date" in out.columns:
        if start_date is not None:
            out = out[out["order_date"].dt.date >= start_date]
        if end_date is not None:
            out = out[out["order_date"].dt.date <= end_date]
    for col, selected in [
        ("category", categories), ("channel", channels), ("city", cities)
    ]:
        if selected and col in out.columns:
            out = out[out[col].isin(selected)]
    return out


def filter_return_data(returns: pd.DataFrame | None, item_subset: pd.DataFrame) -> pd.DataFrame | None:
    if returns is None or "order_id" not in item_subset.columns:
        return None
    order_ids = set(item_subset["order_id"].dropna().unique())
    return returns[returns["order_id"].isin(order_ids)].copy()


def category_order_summary(data: pd.DataFrame, returns: pd.DataFrame | None) -> pd.DataFrame:
    """Return rate by category using orders that contain that category."""
    if not {"category", "order_id"}.issubset(data.columns):
        return pd.DataFrame()
    order_category = data[["category", "order_id"]].dropna().drop_duplicates()
    counts = order_category.groupby("category")["order_id"].nunique().rename("orders")
    if returns is None:
        result = counts.reset_index()
        result["returned_orders"] = pd.NA
        result["return_rate_pct"] = float("nan")
        return result
    returned_ids = set(returns["order_id"].dropna().unique())
    order_category["is_returned"] = order_category["order_id"].isin(returned_ids)
    returned = order_category[order_category["is_returned"]].groupby("category")["order_id"].nunique()
    result = counts.to_frame().join(returned.rename("returned_orders")).fillna({"returned_orders": 0}).reset_index()
    result["return_rate_pct"] = result.apply(
        lambda row: pct(row["returned_orders"], row["orders"]), axis=1
    )
    return result


def financial_summary(data: pd.DataFrame) -> dict:
    revenue = data["net_revenue"].sum(min_count=1) if "net_revenue" in data else float("nan")
    profit = data["gross_profit"].sum(min_count=1) if "gross_profit" in data else float("nan")
    orders = unique_orders(data)
    margin = pct(profit, revenue)
    aov = revenue / orders if orders else float("nan")
    return {
        "revenue": revenue,
        "profit": profit,
        "margin": margin,
        "aov": aov,
        "orders": orders,
    }


def period_previous_dates(start_date, end_date):
    if start_date is None or end_date is None:
        return None, None
    days = (end_date - start_date).days + 1
    prev_end = start_date - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    return prev_start, prev_end


def add_page_filters(data: pd.DataFrame, section: str, page: str) -> pd.DataFrame:
    out = data.copy()
    if section == "Financial Review" and page in ["Overview", "Products & Categories", "Customers & Segments", "Discount Analysis"]:
        segments = st.session_state.get("fin_segment", [])
        payments = st.session_state.get("fin_payment", [])
        if segments and "segment" in out:
            out = out[out["segment"].isin(segments)]
        if payments and "payment_method" in out:
            out = out[out["payment_method"].isin(payments)]
    elif section == "Operational Diagnostics" and page == "Returns":
        subcats = st.session_state.get("ret_subcat", [])
        if subcats and "sub_category" in out:
            out = out[out["sub_category"].isin(subcats)]
    elif section == "Operational Diagnostics" and page == "Delivery & Payments":
        payments = st.session_state.get("ops_payment", [])
        statuses = st.session_state.get("ops_status", [])
        if payments and "payment_method" in out:
            out = out[out["payment_method"].isin(payments)]
        if statuses and "order_status" in out:
            out = out[out["order_status"].isin(statuses)]
    elif section == "Operational Diagnostics" and page == "Customer Retention":
        segments = st.session_state.get("cust_segment", [])
        if segments and "segment" in out:
            out = out[out["segment"].isin(segments)]
    return out


def chart(fig, height=350):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="",
        hoverlabel=dict(namelength=-1),
    )
    st.plotly_chart(fig, width='stretch')


def show_table(data: pd.DataFrame, money_columns=None, pct_columns=None):
    display = data.copy()
    for col in money_columns or []:
        if col in display.columns:
            display[col] = display[col].map(money)
    for col in pct_columns or []:
        if col in display.columns:
            display[col] = display[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}%")
    st.dataframe(display, width='stretch', hide_index=True)


# -------------------------------------------------------------------
# Load inputs and validate
# -------------------------------------------------------------------
if not ITEM_DATA_PATH.exists():
    st.title("D2C Business Intelligence")
    st.error("Prepared dataset not found.")
    st.markdown(
        "Place **`left_df2.csv`** in the same folder as `app.py`. "
        "This app expects the prepared item-level dataframe; it does not repeat notebook cleaning."
    )
    st.code("left_df2.to_csv('left_df2.csv', index=False)")
    st.stop()

df = load_csv(str(ITEM_DATA_PATH))
if df.empty:
    st.error("The prepared dataset is empty.")
    st.stop()
if "order_id" not in df.columns:
    st.error("Required column `order_id` is missing from left_df2.csv.")
    st.stop()

returns_all = prepare_returns(df)
reason_records_all = return_reason_records(df)

# -------------------------------------------------------------------
# App shell and persistent global filter bar
# -------------------------------------------------------------------
st.title("D2C Business Intelligence")
st.caption("Executive overview · Financial performance · Operational diagnosis")

with st.sidebar:
    st.markdown("## Navigation")
    section = st.radio(
        "Business area",
        list(NAV_GROUPS.keys()),
        key="nav_section",
        label_visibility="collapsed",
    )
    page = st.radio(
        "Analysis",
        NAV_GROUPS[section],
        key=f"nav_page_{section}",
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Prepared data")
    st.write(f"**{len(df):,}** item rows")
    st.write(f"**{unique_orders(df):,}** unique orders")
    st.write(f"**{df['customer_id'].nunique():,}** customers" if "customer_id" in df else "Customer count unavailable")
    st.caption("Metrics are recalculated from the selected data.")

st.markdown("### Global filters")
filter_cols = st.columns([1.6, 1.2, 1.2, 1.2, 0.7])

if st.button("Reset filters", type="secondary"):
    for key in [
        "global_dates", "global_categories", "global_channels", "global_cities",
        "fin_segment", "fin_payment", "ret_subcat", "ops_payment", "ops_status",
        "cust_segment",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

if "order_date" in df.columns and df["order_date"].notna().any():
    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()
    selected_dates = filter_cols[0].date_input(
        "Order date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="global_dates",
    )
    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    elif isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 1:
        start_date = end_date = selected_dates[0]
    else:
        start_date, end_date = min_date, max_date
else:
    filter_cols[0].caption("Order dates unavailable")
    start_date = end_date = None

categories = filter_cols[1].multiselect(
    "Category", available_options(df, "category"),
    placeholder="All categories", key="global_categories",
)
channels = filter_cols[2].multiselect(
    "Channel", available_options(df, "channel"),
    placeholder="All channels", key="global_channels",
)
cities = filter_cols[3].multiselect(
    "City", available_options(df, "city"),
    placeholder="All cities", key="global_cities",
)
filter_cols[4].write("")
filter_cols[4].write("")
filter_cols[4].caption("Shared filters")

base_filtered = apply_filters(df, start_date, end_date, categories, channels, cities)

with st.expander("Advanced filters", expanded=False):
    ac1, ac2, ac3 = st.columns(3)
    if section == "Financial Review":
        ac1.multiselect("Customer segment", available_options(base_filtered, "segment"), key="fin_segment")
        ac2.multiselect("Payment method", available_options(base_filtered, "payment_method"), key="fin_payment")
        ac3.caption("These filters apply to all Financial Review subpages.")
    elif section == "Operational Diagnostics" and page == "Returns":
        ac1.multiselect("Subcategory", available_options(base_filtered, "sub_category"), key="ret_subcat")
        ac2.caption("Return reason is explored in the chart/table below.")
    elif section == "Operational Diagnostics" and page == "Delivery & Payments":
        ac1.multiselect("Payment method", available_options(base_filtered, "payment_method"), key="ops_payment")
        ac2.multiselect("Order status", available_options(base_filtered, "order_status"), key="ops_status")
    elif section == "Operational Diagnostics" and page == "Customer Retention":
        ac1.multiselect("Customer segment", available_options(base_filtered, "segment"), key="cust_segment")
        ac2.caption("Cohorts are based on first observed purchase month.")
    else:
        ac1.caption("No additional filters for this view.")

filtered = add_page_filters(base_filtered, section, page)
filtered_returns = filter_return_data(returns_all, filtered)
filtered_reason_records = filter_return_data(reason_records_all, filtered)
st.caption(
    f"Selected scope: **{len(filtered):,} item rows** · "
    f"**{unique_orders(filtered):,} unique orders**"
)
st.divider()


# -------------------------------------------------------------------
# Financial Review: overview
# -------------------------------------------------------------------
if section == "Financial Review" and page == "Overview":
    st.header("Financial Performance Review")
    current = financial_summary(filtered)

    prev_start, prev_end = period_previous_dates(start_date, end_date)
    prev_base = apply_filters(df, prev_start, prev_end, categories, channels, cities)
    prev_data = add_page_filters(prev_base, section, page)
    previous = financial_summary(prev_data)
    growth = pct(current["revenue"] - previous["revenue"], previous["revenue"])

    k1, k2, k3 = st.columns(3)
    k4, k5, k6 = st.columns(3)
    k1.metric("Total Net Revenue", money(current["revenue"]),
              delta=f"{growth:+.2f}% vs previous period" if pd.notna(growth) else None)
    k2.metric("Total Gross Profit", money(current["profit"]))
    k3.metric("Overall Gross Margin", f"{current['margin']:.2f}%" if pd.notna(current["margin"]) else "—")
    k4.metric("Average Order Value", money(current["aov"]))
    k5.metric("Total Orders", f"{current['orders']:,}",
              delta=f"{current['orders'] - previous['orders']:+,} vs previous period"
              if previous["orders"] else None)
    k6.metric("Revenue Growth", f"{growth:+.2f}%" if pd.notna(growth) else "—")

    if pd.notna(growth):
        st.caption("Revenue growth compares the selected date range with the immediately preceding range of equal length.")
    else:
        st.caption("Revenue growth is unavailable when the preceding comparison period has no revenue.")

    st.subheader("Monthly Revenue Trend")
    if {"order_date", "net_revenue"}.issubset(filtered.columns):
        monthly = (
            filtered.dropna(subset=["order_date"])
            .assign(month=lambda x: x["order_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month", as_index=False)
            .agg(net_revenue=("net_revenue", "sum"), gross_profit=("gross_profit", "sum")
                 if "gross_profit" in filtered.columns else ("net_revenue", "sum"))
        )
        chart(px.line(monthly, x="month", y="net_revenue", markers=True,
                      labels={"month": "Month", "net_revenue": "Net Revenue"}))
    else:
        st.info("Monthly trend requires `order_date` and `net_revenue`.")

    left, right = st.columns(2)
    with left:
        st.subheader("Revenue vs. Gross Profit by Month")
        if {"order_date", "net_revenue", "gross_profit"}.issubset(filtered.columns):
            monthly_gp = (
                filtered.dropna(subset=["order_date"])
                .assign(month=lambda x: x["order_date"].dt.to_period("M").dt.to_timestamp())
                .groupby("month", as_index=False)[["net_revenue", "gross_profit"]].sum()
                .melt(id_vars="month", var_name="Metric", value_name="Amount")
            )
            chart(px.line(monthly_gp, x="month", y="Amount", color="Metric", markers=True))
        else:
            st.info("Requires `order_date`, `net_revenue`, and `gross_profit`.")
    with right:
        st.subheader("Revenue by Product Category")
        if {"category", "net_revenue"}.issubset(filtered.columns):
            cat = filtered.groupby("category", as_index=False)["net_revenue"].sum().sort_values("net_revenue")
            chart(px.bar(cat, x="net_revenue", y="category", orientation="h",
                         labels={"net_revenue": "Net Revenue", "category": "Category"}))
        else:
            st.info("Requires `category` and `net_revenue`.")

    left, right = st.columns(2)
    with left:
        st.subheader("Gross Margin by Category")
        if {"category", "net_revenue", "gross_profit"}.issubset(filtered.columns):
            cat = filtered.groupby("category", as_index=False).agg(
                revenue=("net_revenue", "sum"), profit=("gross_profit", "sum")
            )
            cat["gross_margin_pct"] = cat.apply(lambda r: pct(r["profit"], r["revenue"]), axis=1)
            chart(px.bar(cat.sort_values("gross_margin_pct"), x="gross_margin_pct", y="category",
                         orientation="h", labels={"gross_margin_pct": "Gross Margin (%)"}))
        else:
            st.info("Requires category, net revenue, and gross profit.")
    with right:
        st.subheader("Revenue by Sales Channel")
        if {"channel", "net_revenue"}.issubset(filtered.columns):
            channel = filtered.groupby("channel", as_index=False)["net_revenue"].sum().sort_values("net_revenue")
            chart(px.bar(channel, x="net_revenue", y="channel", orientation="h",
                         labels={"net_revenue": "Net Revenue", "channel": "Channel"}))
        else:
            st.info("Requires channel and net revenue.")

    st.subheader("Revenue Contribution by Customer Segment")
    if {"segment", "net_revenue"}.issubset(filtered.columns):
        seg = filtered.groupby("segment", as_index=False)["net_revenue"].sum().sort_values("net_revenue")
        chart(px.bar(seg, x="segment", y="net_revenue", labels={"net_revenue": "Net Revenue"}))
    else:
        st.info("Requires segment and net revenue.")

    st.subheader("Priority business insight")
    if filtered_returns is not None and {"category", "order_id"}.issubset(filtered.columns):
        cat_returns = category_order_summary(filtered, filtered_returns)
        apparel = cat_returns[cat_returns["category"].astype(str).str.casefold() == "apparel"]
        if not apparel.empty and pd.notna(apparel.iloc[0]["return_rate_pct"]):
            row = apparel.iloc[0]
            st.info(
                f"Apparel return rate in the selected scope: **{row['return_rate_pct']:.2f}%** "
                f"({int(row['returned_orders']):,} returned orders out of {int(row['orders']):,} orders). "
                "Investigate sizing and product-description issues before changing discounts."
            )
    else:
        st.info("Load return data to show the evidence-backed return insight.")


# -------------------------------------------------------------------
# Financial Review: Products & Categories
# -------------------------------------------------------------------
elif section == "Financial Review" and page == "Products & Categories":
    st.header("Products & Categories")
    if {"category", "order_id", "net_revenue", "gross_profit"}.issubset(filtered.columns):
        summary = filtered.groupby("category", dropna=False).agg(
            orders=("order_id", "nunique"),
            revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
        ).reset_index()
        summary["gross_margin_pct"] = summary.apply(lambda r: pct(r["gross_profit"], r["revenue"]), axis=1)
        st.subheader("Category Performance")
        show_table(summary.sort_values("revenue", ascending=False),
                   money_columns=["revenue", "gross_profit"], pct_columns=["gross_margin_pct"])
        left, right = st.columns(2)
        with left:
            chart(px.bar(summary.sort_values("revenue"), x="revenue", y="category", orientation="h",
                         labels={"revenue": "Net Revenue", "category": "Category"}))
        with right:
            chart(px.bar(summary.sort_values("gross_margin_pct"), x="gross_margin_pct", y="category",
                         orientation="h", labels={"gross_margin_pct": "Gross Margin (%)"}))
    else:
        st.info("Category profitability requires category, order_id, net_revenue, and gross_profit.")

    st.subheader("Top Products")
    required = {"product_name", "category", "order_id", "net_revenue", "gross_profit"}
    if required.issubset(filtered.columns):
        products = filtered.groupby(["product_name", "category"], dropna=False).agg(
            orders=("order_id", "nunique"),
            revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
        ).reset_index()
        products["gross_margin_pct"] = products.apply(lambda r: pct(r["gross_profit"], r["revenue"]), axis=1)
        top_n = st.slider("Products to display", min_value=5, max_value=30, value=10, step=5)
        show_table(products.sort_values("revenue", ascending=False).head(top_n),
                   money_columns=["revenue", "gross_profit"], pct_columns=["gross_margin_pct"])
    else:
        st.info("Top-products table requires product_name, category, order_id, net_revenue, and gross_profit.")


# -------------------------------------------------------------------
# Financial Review: Customers & Segments
# -------------------------------------------------------------------
elif section == "Financial Review" and page == "Customers & Segments":
    st.header("Customers & Segments")
    odf = order_level(filtered)
    if {"customer_id", "order_id"}.issubset(odf.columns):
        per_customer = odf.groupby("customer_id")["order_id"].nunique()
        customers = int(odf["customer_id"].nunique())
        repeat_rate = pct(int(per_customer.gt(1).sum()), customers)
        a, b, c = st.columns(3)
        a.metric("Unique Customers", f"{customers:,}")
        b.metric("Repeat Customer Rate", f"{repeat_rate:.2f}%" if pd.notna(repeat_rate) else "—")
        c.metric("Average Orders per Customer", f"{per_customer.mean():.2f}" if len(per_customer) else "—")
    else:
        st.info("Customer metrics require customer_id and order_id.")

    if {"segment", "customer_id", "order_id", "net_revenue"}.issubset(filtered.columns):
        seg = filtered.groupby("segment", dropna=False).agg(
            customers=("customer_id", "nunique"),
            orders=("order_id", "nunique"),
            revenue=("net_revenue", "sum"),
        ).reset_index()
        seg["aov"] = (seg["revenue"] / seg["orders"].where(seg["orders"] != 0))
        st.subheader("Customer Segment Performance")
        show_table(seg, money_columns=["revenue", "aov"])
        left, right = st.columns(2)
        with left:
            chart(px.bar(seg, x="segment", y="revenue", labels={"revenue": "Net Revenue"}))
        with right:
            chart(px.bar(seg, x="segment", y="aov", labels={"aov": "Average Order Value"}))
    else:
        st.info("Segment performance requires segment, customer_id, order_id, and net_revenue.")

    if {"customer_id", "order_id", "net_revenue"}.issubset(filtered.columns):
        customer_revenue = filtered.groupby("customer_id", as_index=False)["net_revenue"].sum()
        total_revenue = customer_revenue["net_revenue"].sum()
        customer_revenue = customer_revenue.sort_values("net_revenue", ascending=False)
        top_20_n = max(1, math.ceil(len(customer_revenue) * 0.20)) if len(customer_revenue) else 0
        top_share = pct(customer_revenue.head(top_20_n)["net_revenue"].sum(), total_revenue) if top_20_n else float("nan")
        st.metric("Revenue from Top 20% of Customers", f"{top_share:.2f}%" if pd.notna(top_share) else "—")
        st.caption("This is a concentration measure, not a claim that revenue must follow an 80/20 rule.")


# -------------------------------------------------------------------
# Financial Review: Discount Analysis
# -------------------------------------------------------------------
elif section == "Financial Review" and page == "Discount Analysis":
    st.header("Discount Analysis")
    if {"order_id", "discount_pct", "net_revenue", "gross_profit"}.issubset(filtered.columns):
        # Use the revenue-weighted average line-item discount per order only if
        # list_price exists; otherwise the mean discount across line items is shown.
        work = filtered.copy()
        if "discount_pct" not in work:
            st.info("Discount column unavailable.")
        else:
            work["discount_band"] = pd.cut(
                work["discount_pct"],
                bins=[-float("inf"), 0, 10, 20, float("inf")],
                labels=["No discount / 0%", "1–10%", "11–20%", "21%+"],
                include_lowest=True,
            )
            # Aggregate financial values to order level. For the order's discount,
            # use list-price value (quantity × list_price) as the weight when available.
            if {"quantity", "list_price"}.issubset(work.columns):
                work["_discount_weight"] = (
                    pd.to_numeric(work["quantity"], errors="coerce").fillna(0)
                    * pd.to_numeric(work["list_price"], errors="coerce").fillna(0)
                )
                work["_weighted_discount"] = work["discount_pct"] * work["_discount_weight"]
                discount_order = work.groupby("order_id", as_index=False).agg(
                    revenue=("net_revenue", "sum"),
                    gross_profit=("gross_profit", "sum"),
                    weighted_discount=("_weighted_discount", "sum"),
                    discount_weight=("_discount_weight", "sum"),
                )
                discount_order["discount_pct"] = (
                    discount_order["weighted_discount"]
                    / discount_order["discount_weight"].replace(0, float("nan"))
                )
            else:
                # Fallback when list-price weights are unavailable.
                discount_order = work.groupby("order_id", as_index=False).agg(
                    revenue=("net_revenue", "sum"),
                    gross_profit=("gross_profit", "sum"),
                    discount_pct=("discount_pct", "mean"),
                )
            discount_order["discount_band"] = pd.cut(
                discount_order["discount_pct"],
                bins=[-float("inf"), 0, 10, 20, float("inf")],
                labels=["No discount / 0%", "1–10%", "11–20%", "21%+"],
                include_lowest=True,
            )
            band = discount_order.groupby("discount_band", observed=False).agg(
                orders=("order_id", "nunique"),
                revenue=("revenue", "sum"),
                gross_profit=("gross_profit", "sum"),
            ).reset_index()
            band["aov"] = band["revenue"] / band["orders"].replace(0, pd.NA)
            band["gross_margin_pct"] = band.apply(lambda r: pct(r["gross_profit"], r["revenue"]), axis=1)
            show_table(band, money_columns=["revenue", "gross_profit", "aov"],
                       pct_columns=["gross_margin_pct"])
            left, right = st.columns(2)
            with left:
                chart(px.bar(band, x="discount_band", y="aov",
                             labels={"discount_band": "Discount Band", "aov": "AOV"}))
            with right:
                chart(px.bar(band, x="discount_band", y="gross_margin_pct",
                             labels={"discount_band": "Discount Band", "gross_margin_pct": "Gross Margin (%)"}))
            st.warning(
                "Discount bands are assigned using the mean discount across each order's line items. "
                "The relationship is descriptive; it does not prove that discounts cause lower AOV or margin."
            )
    else:
        st.info("Discount analysis requires order_id, discount_pct, net_revenue, and gross_profit.")


# -------------------------------------------------------------------
# Operational Diagnostics: overview
# -------------------------------------------------------------------
elif section == "Operational Diagnostics" and page == "Overview":
    st.header("Operational & Business Problem Diagnosis")
    odf = order_level(filtered)
    returned_ids = set(filtered_returns["order_id"].dropna().unique()) if filtered_returns is not None else set()
    orders = unique_orders(filtered)
    returned_orders = len(set(filtered["order_id"].dropna().unique()) & returned_ids)
    return_rate = pct(returned_orders, orders)

    apparel_rate = float("nan")
    if "category" in filtered.columns:
        apparel_data = filtered[filtered["category"].astype(str).str.casefold() == "apparel"]
        if len(apparel_data):
            apparel_ids = set(apparel_data["order_id"].dropna().unique())
            apparel_returned = len(apparel_ids & returned_ids)
            apparel_rate = pct(apparel_returned, len(apparel_ids))

    avg_delivery = odf["delivery_days"].mean() if "delivery_days" in odf.columns else float("nan")
    if "customer_id" in odf.columns and "order_id" in odf.columns:
        cust_orders = odf.groupby("customer_id")["order_id"].nunique()
        repeat_rate = pct(int(cust_orders.gt(1).sum()), len(cust_orders))
    else:
        repeat_rate = float("nan")
    marketplace_rate = float("nan")
    if "channel" in filtered.columns and len(filtered[filtered["channel"].astype(str).str.casefold() == "marketplace"]):
        market = filtered[filtered["channel"].astype(str).str.casefold() == "marketplace"]
        market_ids = set(market["order_id"].dropna().unique())
        marketplace_rate = pct(len(market_ids & returned_ids), len(market_ids))
    if "order_status" in odf.columns:
        cancelled_n = int(odf["order_status"].astype(str).str.contains("cancel", case=False, na=False).sum())
        cancel_rate = pct(cancelled_n, len(odf))
    else:
        cancel_rate = float("nan")

    r1 = st.columns(3)
    r2 = st.columns(3)
    r1[0].metric("Overall Return Rate", f"{return_rate:.2f}%" if pd.notna(return_rate) else "—")
    r1[1].metric("Apparel Return Rate", f"{apparel_rate:.2f}%" if pd.notna(apparel_rate) else "—")
    r1[2].metric("Average Delivery Time", f"{avg_delivery:.2f} days" if pd.notna(avg_delivery) else "—")
    r2[0].metric("Repeat Customer Rate", f"{repeat_rate:.2f}%" if pd.notna(repeat_rate) else "—")
    r2[1].metric("Marketplace Return Rate", f"{marketplace_rate:.2f}%" if pd.notna(marketplace_rate) else "—")
    r2[2].metric("Cancellation Rate", f"{cancel_rate:.2f}%" if pd.notna(cancel_rate) else "—")

    if filtered_returns is None:
        st.warning("Return metrics need `return_id`/`return_reason` data keyed by `order_id`. Add `left_df3.csv` or `returns.csv` beside app.py.")
    else:
        st.caption("Returns are counted by unique order. The source data does not identify returned product IDs, so product-level return rates are not presented as exact.")

    left, right = st.columns(2)
    with left:
        st.subheader("Return Rate by Category")
        cat_ret = category_order_summary(filtered, filtered_returns)
        if not cat_ret.empty and cat_ret["return_rate_pct"].notna().any():
            chart(px.bar(cat_ret.sort_values("return_rate_pct"), x="return_rate_pct", y="category",
                         orientation="h", labels={"return_rate_pct": "Return Rate (%)", "category": "Category"}))
        else:
            st.info("Requires return data and category/order IDs.")
    with right:
        st.subheader("Return Reasons")
        if filtered_reason_records is not None and not filtered_reason_records.empty:
            reason = filtered_reason_records.groupby("return_reason", as_index=False)["order_id"].nunique()
            reason = reason.rename(columns={"order_id": "returned_orders"}).sort_values("returned_orders")
            chart(px.bar(reason, x="returned_orders", y="return_reason", orientation="h",
                         labels={"returned_orders": "Returned Orders", "return_reason": "Reason"}))
        else:
            st.info("Return reason data unavailable.")

    left, right = st.columns(2)
    with left:
        st.subheader("Return Rate: Delivery Within 7 Days vs. Over 7 Days")
        if filtered_returns is not None and {"delivery_days", "order_id"}.issubset(odf.columns):
            odf_return = odf.copy()
            odf_return["is_returned"] = odf_return["order_id"].isin(returned_ids)
            odf_return["delivery_group"] = odf_return["delivery_days"].apply(
                lambda x: "Within 7 days" if pd.notna(x) and x <= 7
                else ("Over 7 days" if pd.notna(x) else None)
            )
            delay = odf_return.dropna(subset=["delivery_group"]).groupby("delivery_group", as_index=False).agg(
                orders=("order_id", "nunique"), returned_orders=("is_returned", "sum")
            )
            delay["return_rate_pct"] = delay.apply(lambda r: pct(r["returned_orders"], r["orders"]), axis=1)
            chart(px.bar(delay, x="delivery_group", y="return_rate_pct",
                         labels={"delivery_group": "Delivery Group", "return_rate_pct": "Return Rate (%)"}))
        else:
            st.info("Requires delivery days and return data.")
    with right:
        st.subheader("Average Delivery Days by City")
        if {"city", "delivery_days"}.issubset(odf.columns):
            city_data = odf.groupby("city", as_index=False).agg(
                orders=("order_id", "nunique"), avg_delivery_days=("delivery_days", "mean")
            ).sort_values("avg_delivery_days")
            chart(px.bar(city_data, x="avg_delivery_days", y="city", orientation="h",
                         labels={"avg_delivery_days": "Average Delivery Days", "city": "City"}))
        else:
            st.info("Requires city and delivery_days.")

    st.subheader("Channel Performance Diagnostics")
    if {"channel", "order_id", "net_revenue", "gross_profit"}.issubset(filtered.columns):
        channel = filtered.groupby("channel", as_index=False).agg(
            orders=("order_id", "nunique"),
            revenue=("net_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
        channel["gross_margin_pct"] = channel.apply(lambda r: pct(r["gross_profit"], r["revenue"]), axis=1)
        if "discount_pct" in filtered.columns:
            disc = filtered.groupby("channel")["discount_pct"].mean().rename("avg_discount_pct")
            channel = channel.merge(disc, on="channel", how="left")
        if "delivery_days" in odf.columns and "channel" in odf.columns:
            delivery = odf.groupby("channel")["delivery_days"].mean().rename("avg_delivery_days")
            channel = channel.merge(delivery, on="channel", how="left")
        if filtered_returns is not None:
            returned_per_channel = filtered[["order_id", "channel"]].drop_duplicates()
            returned_per_channel["is_returned"] = returned_per_channel["order_id"].isin(returned_ids)
            rr = returned_per_channel.groupby("channel").agg(
                returned_orders=("is_returned", "sum"), orders=("order_id", "nunique")
            )
            rr["return_rate_pct"] = rr.apply(lambda r: pct(r["returned_orders"], r["orders"]), axis=1)
            channel = channel.merge(rr[["return_rate_pct"]], on="channel", how="left")
        show_table(channel, money_columns=["revenue", "gross_profit"],
                   pct_columns=["gross_margin_pct", "avg_discount_pct", "return_rate_pct"])
    else:
        st.info("Channel diagnostics require channel, order_id, net_revenue, and gross_profit.")


# -------------------------------------------------------------------
# Operational Diagnostics: Returns
# -------------------------------------------------------------------
elif section == "Operational Diagnostics" and page == "Returns":
    st.header("Returns Investigation")
    if filtered_returns is None:
        st.warning("This view needs return data keyed by `order_id`. Place `left_df3.csv` or `returns.csv` beside app.py.")
    else:
        order_ids = set(filtered["order_id"].dropna().unique())
        returned_ids = set(filtered_returns["order_id"].dropna().unique()) & order_ids
        total = len(order_ids)
        returned = len(returned_ids)
        a, b, c = st.columns(3)
        a.metric("Orders in Scope", f"{total:,}")
        b.metric("Returned Orders", f"{returned:,}")
        a_rate = pct(returned, total)
        c.metric("Order Return Rate", f"{a_rate:.2f}%" if pd.notna(a_rate) else "—")

        cat_ret = category_order_summary(filtered, filtered_returns)
        left, right = st.columns(2)
        with left:
            st.subheader("Return Rate by Category")
            if not cat_ret.empty:
                chart(px.bar(cat_ret.sort_values("return_rate_pct"), x="return_rate_pct", y="category",
                             orientation="h", labels={"return_rate_pct": "Return Rate (%)"}))
        with right:
            st.subheader("Return Reasons")
            if filtered_reason_records is not None and not filtered_reason_records.empty:
                reason = filtered_reason_records.groupby("return_reason", as_index=False)["order_id"].nunique()
                reason = reason.rename(columns={"order_id": "returned_orders"}).sort_values("returned_orders")
                reason["share_of_reason_records_pct"] = reason["returned_orders"] / reason["returned_orders"].sum() * 100
                chart(px.bar(reason, x="returned_orders", y="return_reason", orientation="h",
                             labels={"returned_orders": "Returned Orders", "return_reason": "Return Reason"}))
                show_table(reason.sort_values("returned_orders", ascending=False),
                           pct_columns=["share_of_reason_records_pct"])

        st.subheader("Return Rate by Subcategory")
        if "sub_category" in filtered.columns:
            subcat = filtered[["sub_category", "order_id"]].dropna().drop_duplicates()
            subcat["is_returned"] = subcat["order_id"].isin(returned_ids)
            sub_summary = subcat.groupby("sub_category", as_index=False).agg(
                orders=("order_id", "nunique"), returned_orders=("is_returned", "sum")
            )
            sub_summary["return_rate_pct"] = sub_summary.apply(lambda r: pct(r["returned_orders"], r["orders"]), axis=1)
            chart(px.bar(sub_summary.sort_values("return_rate_pct"), x="return_rate_pct", y="sub_category",
                         orientation="h", labels={"return_rate_pct": "Return Rate (%)"}))
        else:
            st.info("Subcategory column unavailable.")

        st.subheader("Return Reason Breakdown")
        if filtered_reason_records is not None and not filtered_reason_records.empty:
            show_table(
                filtered_reason_records.groupby("return_reason", as_index=False)["order_id"].nunique()
                .rename(columns={"order_id": "returned_orders"})
                .sort_values("returned_orders", ascending=False)
            )

        st.warning(
            "Returns are linked to orders, not specific product line items. Category/subcategory rates describe "
            "orders containing those categories; they do not prove which item in a multi-item order was returned."
        )


# -------------------------------------------------------------------
# Operational Diagnostics: Delivery & Payments
# -------------------------------------------------------------------
elif section == "Operational Diagnostics" and page == "Delivery & Payments":
    st.header("Delivery & Payments")
    odf = order_level(filtered)
    total_orders = unique_orders(odf)
    if "delivery_days" in odf:
        avg_delivery = odf["delivery_days"].mean()
        median_delivery = odf["delivery_days"].median()
    else:
        avg_delivery = median_delivery = float("nan")
    if "order_status" in odf:
        cancelled_mask = odf["order_status"].astype(str).str.contains("cancel", case=False, na=False)
        cancelled_orders = int(cancelled_mask.sum())
        cancellation_rate = pct(cancelled_orders, total_orders)
    else:
        cancelled_orders = 0
        cancellation_rate = float("nan")

    a, b, c = st.columns(3)
    a.metric("Unique Orders", f"{total_orders:,}")
    b.metric("Average Delivery Time", f"{avg_delivery:.2f} days" if pd.notna(avg_delivery) else "—")
    c.metric("Cancellation Rate", f"{cancellation_rate:.2f}%" if pd.notna(cancellation_rate) else "—")
    d, e, f = st.columns(3)
    d.metric("Median Delivery Time", f"{median_delivery:.2f} days" if pd.notna(median_delivery) else "—")
    e.metric("Cancelled Orders", f"{cancelled_orders:,}")
    if "payment_method" in odf:
        cod = odf["payment_method"].astype(str).str.contains("cod|cash on delivery", case=False, na=False)
        f.metric("COD Share", f"{pct(int(cod.sum()), total_orders):.2f}%" if total_orders else "—")
    else:
        f.metric("COD Share", "—")

    left, right = st.columns(2)
    with left:
        st.subheader("Average Delivery Days by City")
        if {"city", "delivery_days"}.issubset(odf.columns):
            city_data = odf.groupby("city", as_index=False).agg(
                orders=("order_id", "nunique"), avg_delivery_days=("delivery_days", "mean")
            ).sort_values("avg_delivery_days")
            chart(px.bar(city_data, x="avg_delivery_days", y="city", orientation="h",
                         labels={"avg_delivery_days": "Average Delivery Days", "city": "City"}))
            show_table(city_data.sort_values("avg_delivery_days", ascending=False))
        else:
            st.info("Requires city and delivery_days.")
    with right:
        st.subheader("Orders by Payment Method")
        if {"payment_method", "order_id"}.issubset(odf.columns):
            pay = odf.groupby("payment_method", as_index=False)["order_id"].nunique().rename(columns={"order_id": "orders"})
            chart(px.bar(pay.sort_values("orders"), x="orders", y="payment_method", orientation="h",
                         labels={"orders": "Orders", "payment_method": "Payment Method"}))
        else:
            st.info("Requires payment_method and order_id.")

    st.subheader("Cancellation Rate by Payment Method")
    if {"payment_method", "order_status", "order_id"}.issubset(odf.columns):
        payment_status = odf.copy()
        payment_status["is_cancelled"] = payment_status["order_status"].astype(str).str.contains(
            "cancel", case=False, na=False
        )
        pay_cancel = payment_status.groupby("payment_method", as_index=False).agg(
            orders=("order_id", "nunique"), cancelled_orders=("is_cancelled", "sum")
        )
        pay_cancel["cancellation_rate_pct"] = pay_cancel.apply(
            lambda r: pct(r["cancelled_orders"], r["orders"]), axis=1
        )
        chart(px.bar(pay_cancel, x="payment_method", y="cancellation_rate_pct",
                     labels={"cancellation_rate_pct": "Cancellation Rate (%)"}))
        show_table(pay_cancel, pct_columns=["cancellation_rate_pct"])
    else:
        st.info("Requires payment method and order status.")

    st.subheader("COD Usage by City Group")
    group_col = next((c for c in ["city_group", "city_type", "city_tier", "region_type"] if c in odf.columns), None)
    if group_col and "payment_method" in odf:
        odf["is_cod"] = odf["payment_method"].astype(str).str.contains("cod|cash on delivery", case=False, na=False)
        cod_summary = odf.groupby(group_col, as_index=False).agg(
            orders=("order_id", "nunique"), cod_orders=("is_cod", "sum")
        )
        cod_summary["cod_share_pct"] = cod_summary.apply(lambda r: pct(r["cod_orders"], r["orders"]), axis=1)
        chart(px.bar(cod_summary, x=group_col, y="cod_share_pct",
                     labels={"cod_share_pct": "COD Share (%)", group_col: "City Group"}))
    else:
        st.info("A verified city-group column (for example, Metro/Tier-2) is not present. The app will not guess city classifications.")

    st.caption("These comparisons are descriptive and do not establish that delivery delays or payment method cause returns/cancellations.")


# -------------------------------------------------------------------
# Operational Diagnostics: Customer Retention
# -------------------------------------------------------------------
elif section == "Operational Diagnostics" and page == "Customer Retention":
    st.header("Customer Retention")
    odf = order_level(filtered)
    if {"customer_id", "order_id"}.issubset(odf.columns):
        per_customer = odf.groupby("customer_id")["order_id"].nunique()
        customers = len(per_customer)
        repeat_customers = int(per_customer.gt(1).sum())
        a, b, c = st.columns(3)
        a.metric("Customers", f"{customers:,}")
        a_rate = pct(repeat_customers, customers)
        b.metric("Repeat Customer Rate", f"{a_rate:.2f}%" if pd.notna(a_rate) else "—")
        c.metric("Average Orders per Customer", f"{per_customer.mean():.2f}" if customers else "—")
        freq = per_customer.value_counts().sort_index().rename_axis("orders_per_customer").reset_index(name="customers")
        left, right = st.columns(2)
        with left:
            st.subheader("Customer Order Frequency")
            chart(px.bar(freq, x="orders_per_customer", y="customers",
                         labels={"orders_per_customer": "Orders per Customer", "customers": "Customers"}))
        with right:
            st.subheader("Repeat vs. One-Time Customers")
            split = pd.DataFrame({
                "Customer Type": ["Repeat", "One-time"],
                "Customers": [repeat_customers, customers - repeat_customers],
            })
            chart(px.pie(split, names="Customer Type", values="Customers", hole=0.55))
    else:
        st.info("Retention metrics require customer_id and order_id.")

    # Cohort retention: cohort is the first observed purchase month, not signup_date.
    if {"customer_id", "order_id", "order_date"}.issubset(filtered.columns):
        cohort_source = filtered.dropna(subset=["customer_id", "order_id", "order_date"]).copy()
        cohort_source["order_month"] = cohort_source["order_date"].dt.to_period("M")
        first_purchase = cohort_source.groupby("customer_id")["order_month"].min().rename("cohort_month")
        cohort_source = cohort_source.join(first_purchase, on="customer_id")
        cohort_source["cohort_age"] = (
            (cohort_source["order_month"].dt.year - cohort_source["cohort_month"].dt.year) * 12
            + (cohort_source["order_month"].dt.month - cohort_source["cohort_month"].dt.month)
        )
        cohort_counts = cohort_source.groupby(["cohort_month", "cohort_age"])["customer_id"].nunique().unstack()
        if not cohort_counts.empty:
            cohort_sizes = cohort_counts.iloc[:, 0]
            retention = cohort_counts.divide(cohort_sizes, axis=0) * 100
            retention.columns = [f"Month {int(c)}" for c in retention.columns]
            retention.index = retention.index.astype(str)
            st.subheader("Customer Cohort Retention")
            st.caption("Cohorts use first observed purchase month. Blank cells mean the cohort has not had enough observed follow-up time.")
            chart(px.imshow(
                retention,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "Months Since First Purchase", "y": "Cohort Month", "color": "Retention (%)"},
                text_auto=".1f",
            ), height=max(350, min(650, 35 * len(retention) + 150)))
            st.dataframe(retention.round(2), width='stretch')
    else:
        st.info("Cohort analysis requires customer_id, order_id, and order_date.")


# -------------------------------------------------------------------
# Footer and data caveats
# -------------------------------------------------------------------
st.divider()
with st.expander("Metric definitions & data limitations"):
    st.markdown(
        """
        - **Net revenue and gross profit:** summed from prepared item-level rows.
        - **Gross margin:** total gross profit ÷ total net revenue; row-level margins are not averaged.
        - **AOV:** item-level net revenue ÷ unique orders in the selected scope.
        - **Order counts:** distinct `order_id`, not item-row counts.
        - **Return rate:** unique returned orders ÷ unique orders in scope. Return data is kept separate from item revenue to avoid duplicate-inflated totals.
        - **Product return limitation:** returns are linked to orders. Without returned product IDs, product-specific return rates cannot be claimed as exact.
        - **Growth:** selected period versus the immediately preceding period of equal duration; missing prior-period data makes growth unavailable.
        - **Interpretation:** observed associations do not establish causation. Gross profit is not net profit because operating expenses are not included.
        """
    )
st.caption("D2C Business Intelligence · Prepared-data dashboard")
