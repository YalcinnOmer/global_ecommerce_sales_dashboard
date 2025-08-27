from __future__ import annotations
from pathlib import Path
import os
import sqlite3
from typing import Optional, List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from matplotlib.patches import ConnectionPatch
sns.set_theme(style="whitegrid")


def resolve_base_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

base_dir = resolve_base_dir()
data_dir = base_dir / "data" / "raw"
db_path = base_dir / 'db' / 'olist_ecom.db'
fig_dir = base_dir / 'figures'

bases_to_table = {
    "olist_customers_dataset": "customers",
    "olist_geolocation_dataset": "geolocation",
    "olist_order_items_dataset": "order_items",
    "olist_order_payments_dataset": "payments",
    "olist_order_reviews_dataset": "reviews",
    "olist_orders_dataset": "orders",
    "olist_products_dataset": "products",
    "olist_sellers_dataset": "sellers",
    "product_category_name_translation": "product_category_name_translation",}

def ensure_dirs() -> None:
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(db_path.parent, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

def connect_fast(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript("""
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA cache_size = -20000;
    PRAGMA foreign_keys = ON;
    """)
    return conn



def find_file(base:str) -> Optional[Path]:
    for ext in (".csv",".xlsx"):
        p = data_dir / f"{base}{ext}"
        if p.exists():
            return p
    hits = sorted(data_dir.glob(f"{base}*"))
    return hits[0] if hits else None

def safe_chunksize(n_cols : int, target: int = 900) -> int:
    return max(1,target // max(1, n_cols))

def load_one(conn: sqlite3.Connection, path: Path, table: str, replace: bool = True) -> None:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df = pd.read_excel(path)
        cs = safe_chunksize(len(df.columns))
        df.to_sql(table, conn, if_exists="replace" if replace else "append", index=False, chunksize = cs,method = None)
        print(f'{path.name} -> {table} (rows = {len(df):,},cs = {cs}')
        return

    if "olist_geolocation_dataset" in path.stem:
        total = 0
        first = True
        for chunk in pd.read_csv(path, chunksize = 100_000, low_memory=False):
            cs = safe_chunksize(len(chunk.columns))
            chunk.to_sql(table, conn, if_exists = 'replace' if (replace and first) else 'append', index = False, chunksize = cs,method = None)
            total += len(chunk)
            first = False
        print(f'{path.name} -> {table} (rows = {total:,},CHUNKED)')
    else:
        df = pd.read_csv(path, low_memory=False)
        cs = safe_chunksize(len(df.columns))
        df.to_sql(table, conn, if_exists = 'replace' if replace else 'append', index = False, chunksize = cs, method = None)
        print(f'{path.name} -> {table} (rows = {len(df):,} cs = {cs})')

def load_all(conn : sqlite3.Connection) -> None:
    for base ,table in bases_to_table.items():
        path = find_file(base)
        if not path:
            print(f'[WARN] Missing file for base: {base} (expected in {data_dir}')
            continue
        load_one(conn, path, table,replace=True)

def create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_orders_customer      ON orders(customer_id);
            CREATE INDEX IF NOT EXISTS idx_orders_purchase_date ON orders(order_purchase_timestamp);
            CREATE INDEX IF NOT EXISTS idx_items_order          ON order_items(order_id);
            CREATE INDEX IF NOT EXISTS idx_items_product        ON order_items(product_id);
            CREATE INDEX IF NOT EXISTS idx_payments_order       ON payments(order_id);
            CREATE INDEX IF NOT EXISTS idx_reviews_order        ON reviews(order_id);
            CREATE INDEX IF NOT EXISTS idx_customers_state      ON customers(customer_state);
            CREATE INDEX IF NOT EXISTS idx_sellers_state        ON sellers(seller_state);
            CREATE INDEX IF NOT EXISTS idx_products_cat         ON products(product_category_name);""")
    conn.commit()
    print("Indexes created")

def list_tables(conn: sqlite3.Connection) -> List[str]:
    df = pd.read_sql('SELECT name FROM sqlite_master WHERE type = "table" ORDER BY name', conn)
    return df['name'].tolist()

def table_count(conn: sqlite3.Connection, tables: list[str]) -> pd.DataFrame:
    rows = []
    for t in tables:
        try:
            n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {t}", conn).iloc[0, 0]
        except Exception as e:
            n = None
        rows.append({"table":t, "rows": n})
    return pd.DataFrame(rows).sort_values("table")

def quick_eda(conn: sqlite3.Connection) -> None:
    print("\n === Quick Eda ====")

    q_orders =  """
        SELECT
            strftime('%Y-%m', order_purchase_timestamp) AS ym,
            COUNT(*) AS n_orders
        FROM orders
        GROUP BY 1
        ORDER BY 1"""
    orders_month = pd.read_sql(q_orders, conn)
    print("\nOrders by month (head)")
    print(orders_month.head())

    q_pay_per_order = """
            SELECT order_id, SUM(payment_value) AS order_value
            FROM payments
            GROUP BY order_id"""
    pay = pd.read_sql(q_pay_per_order, conn)
    aov = pay['order_value'].mean()
    print(f"\nOverall AOV = {aov:,.2f}")

    q_items_prod = """
        SELECT oi.product_id, COUNT(*) AS n_items
        FROM order_items oi
        GROUP BY 1;"""

    items_prod = pd.read_sql(q_items_prod, conn)
    products = pd.read_sql("SELECT product_id, product_category_name FROM products",conn)
    top_cat = (items_prod.merge(products, on='product_id', how='left')
               .groupby("product_category_name",dropna=False)["n_items"].sum()
               .sort_values(ascending=False).head(10))
    print("\nTop products by category by items sold:")
    print(top_cat)

    q_rev_by_state = """
           SELECT c.customer_state, SUM(p.payment_value) AS order_value
           FROM payments p
           JOIN orders o   ON o.order_id = p.order_id
           JOIN customers c ON c.customer_id = o.customer_id
           GROUP BY 1
           ORDER BY order_value DESC
           LIMIT 10;"""
    rev = pd.read_sql(q_rev_by_state, conn).set_index('customer_state')['order_value']
    print("\nRevenue by customer_state(top 10)")
    print(rev)

def _money_fmt(x, pos):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}k"
    return f"${x:.0f}"

def _maybe_save(fig_dir: Optional[Path], name: str):
    if fig_dir:
        plt.savefig(fig_dir / name, dpi=150, bbox_inches="tight")

def plot_order_revenue_trend(conn: sqlite3.Connection, out_dir: Optional[Path] = None) -> None:
    q_orders = """
        SELECT strftime('%Y-%m', order_purchase_timestamp) AS ym,
               COUNT(DISTINCT order_id) AS n_orders
        FROM orders
        GROUP BY 1
        ORDER BY 1
    """
    df_o = pd.read_sql(q_orders, conn)
    df_o["ym"] = pd.to_datetime(df_o["ym"] + "-01")
    df_o = df_o[df_o["ym"] <= "2018-08-01"].sort_values("ym")

    plt.figure(figsize=(12,5))
    ax = sns.lineplot(data=df_o, x="ym", y="n_orders", marker="o", linewidth=2, color="tab:blue", label="Orders")
    ax.set_title("Monthly Orders Trend", fontsize=14, pad=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Orders"); ax.grid(True, alpha=.25)
    for t in ax.get_xticklabels():
        t.set_rotation(30); t.set_ha("right")
    plt.tight_layout()
    _maybe_save(out_dir, "orders_trend.png")
    plt.show()

    q_rev = """
        SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS ym,
               SUM(p.payment_value) AS revenue
        FROM orders o
        JOIN payments p ON p.order_id = o.order_id
        GROUP BY 1
        ORDER BY 1
    """
    df_r = pd.read_sql(q_rev, conn)
    df_r["ym"] = pd.to_datetime(df_r["ym"] + "-01")
    df_r = df_r[df_r["ym"] <= "2018-08-01"].sort_values("ym")

    plt.figure(figsize=(12,5))
    ax = sns.lineplot(data=df_r, x="ym", y="revenue", marker="o", linewidth=2, color="tab:green", label="Revenue")
    ax.set_title("Monthly Revenue Trend", fontsize=14, pad=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax.grid(True, alpha=.25)
    for t in ax.get_xticklabels():
        t.set_rotation(30); t.set_ha("right")
    plt.tight_layout()
    _maybe_save(out_dir, "revenue_trend.png")
    plt.show()


def plot_top_categories_by_items(conn: sqlite3.Connection, out_dir: Optional[Path] = None, top_n: int = 15) -> None:
    q = """
        SELECT p.product_category_name AS category,
               SUM(oi.price) AS revenue
        FROM order_items oi
        LEFT JOIN products p ON p.product_id = oi.product_id
        GROUP BY 1
        ORDER BY revenue DESC
        LIMIT ?;
    """
    df = pd.read_sql(q, conn, params=(top_n,))
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(x="category", y="revenue", data=df, color="tab:green")
    ax.set_title(f"Top {top_n} Categories by Revenue (from items.price)", fontsize=14, pad=12)
    ax.set_xlabel("Category")
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    for lab in ax.get_xticklabels():
        lab.set_rotation(35)
        lab.set_ha("right")
    plt.tight_layout()
    if out_dir: plt.savefig(out_dir / "top_categories_revenue.png", dpi=150)
    plt.show()


def plot_top_categories_by_revenue(conn: sqlite3.Connection, out_dir: Optional[Path] = None, top_n: int = 15) -> None:
    q = """
        SELECT p.product_category_name AS category,
               SUM(oi.price) AS revenue
        FROM order_items oi
        LEFT JOIN products p ON p.product_id = oi.product_id
        GROUP BY 1
        ORDER BY revenue DESC
        LIMIT ?
    """
    df = pd.read_sql(q, conn, params=(top_n,))
    plt.figure(figsize=(12,6))
    ax = sns.barplot(data=df, y="category", x="revenue", color="tab:green")  # tek renk -> deprecation yok
    ax.set_title(f"Top {top_n} Categories by Revenue (from items.price)", fontsize=14, pad=12)
    ax.set_xlabel("Revenue"); ax.set_ylabel("Category")
    ax.xaxis.set_major_formatter(FuncFormatter(_money_fmt))
    plt.tight_layout()
    _maybe_save(out_dir, "top_categories_revenue.png")
    plt.show()



def plot_state_revenue(conn: sqlite3.Connection, out_dir: Optional[Path] = None, top_n: int = 10) -> None:
    q = """
        SELECT c.customer_state AS state, SUM(p.payment_value) AS revenue
        FROM payments p
        JOIN orders o   ON o.order_id   = p.order_id
        JOIN customers c ON c.customer_id = o.customer_id
        GROUP BY 1
        ORDER BY revenue DESC
        LIMIT ?
    """
    df = pd.read_sql(q, conn, params=(top_n,))
    plt.figure(figsize=(12,5))
    ax = sns.barplot(data=df, x="state", y="revenue", color="tab:blue")
    ax.set_title(f"Top {top_n} States by Revenue", fontsize=14, pad=12)
    ax.set_xlabel("State"); ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(FuncFormatter(_money_fmt))
    ax.grid(True, axis="y", alpha=.25)
    plt.tight_layout()
    _maybe_save(out_dir, "state_revenue.png")
    plt.show()



def plot_payment_type_share(
    conn: sqlite3.Connection,
    out_dir: Optional[Path] = None,
    by: str = "revenue",
    min_share: float = 0.02,
    donut: bool = True,
    min_pct_label: float = 3.0,
) -> None:
    assert by in {"revenue", "count"}

    if by == "revenue":
        q = "SELECT payment_type, SUM(payment_value) AS value FROM payments GROUP BY 1"
        title_metric = "Revenue"
    else:
        q = "SELECT payment_type, COUNT(*) AS value FROM payments GROUP BY 1"
        title_metric = "Count"

    raw = pd.read_sql(q, conn)
    raw = raw.sort_values("value", ascending=False).reset_index(drop=True)
    total = raw["value"].sum()
    raw["share"] = raw["value"] / total


    major = raw[raw["share"] >= min_share].copy()
    minor = raw[raw["share"] <  min_share]
    if not minor.empty:
        other_row = pd.DataFrame([{
            "payment_type": "other",
            "value": minor["value"].sum(),
            "share":  minor["value"].sum() / total
        }])
        df = pd.concat([major, other_row], ignore_index=True)
    else:
        df = major.copy()


    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_aspect("equal", "box")

    wedgeprops = dict(edgecolor="white")
    if donut:
        wedgeprops["width"] = 0.38


    wedges, _ = ax.pie(
        df["value"],
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops=wedgeprops
    )


    ax.set_title(f"Payment Type {title_metric} Share", fontsize=14, pad=14)


    def fmt_value(v):
        return _money_fmt(v, None) if by == "revenue" else f"{int(v):,}"
    legends = [
        f"{pt} — {fmt_value(v)} ({s:.1%})"
        for pt, v, s in zip(df["payment_type"], df["value"], df["share"])
    ]
    ax.legend(
        wedges, legends, title=title_metric,
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True
    )




    angles = []
    for w in wedges:
        ang = np.deg2rad((w.theta1 + w.theta2) / 2.0)
        angles.append(ang)


    show = (df["share"] * 100.0 >= min_pct_label).to_list()


    r_text = 1.18
    r_anchor = 1.00 if not donut else 0.99 + wedgeprops["width"]

    min_dy = 0.06


    right_idx = [i for i, a in enumerate(angles) if np.cos(a) >= 0]
    left_idx  = [i for i, a in enumerate(angles) if np.cos(a) <  0]

    def place_side(idxs, side=1):
        if not idxs:
            return
        idxs = sorted(idxs, key=lambda i: -np.sin(angles[i]))
        ys = [np.sin(angles[i]) for i in idxs]
        for j in range(1, len(ys)):
            if ys[j-1] - ys[j] < min_dy:
                ys[j] = ys[j-1] - min_dy

        for i, y in zip(idxs, ys):
            if not show[i]:
                continue
            a = angles[i]
            x_text = side * r_text
            y_text = y
            pct = df.loc[i, "share"] * 100.0
            label = f"{pct:.1f}%"
            ha = "left" if side > 0 else "right"
            ax.text(x_text, y_text, label, ha=ha, va="center", fontsize=11)

            x0 = np.cos(a) * r_anchor
            y0 = np.sin(a) * r_anchor
            con = ConnectionPatch(
                xyA=(x_text - 0.03*side, y_text), coordsA=ax.transData,
                xyB=(x0, y0), coordsB=ax.transData,
                arrowstyle="-", lw=1.0, color="#666666"
            )
            ax.add_artist(con)

    place_side(right_idx, side=+1)
    place_side(left_idx,  side=-1)

    fig.subplots_adjust(left=0.06, right=0.80, top=0.92, bottom=0.08)

    if out_dir:
        name = f"payment_type_share_{by}{'_donut' if donut else ''}.png"
        plt.savefig(out_dir / name, dpi=150)
    plt.show()



def main() -> None:
    print(">>> Olist pipeline starting...")
    ensure_dirs()
    print(f"Data dir: {data_dir}")
    print(f"DB path:  {db_path}")

    conn = connect_fast(db_path)

    print("\n--- Loading files into SQLite ---")
    load_all(conn)

    print("\n--- Creating indexes ---")
    create_indexes(conn)

    print("\n--- Verifying tables ---")
    tables = list_tables(conn)
    print("tables:", tables)
    print(table_count(conn, tables))

    print("\n--- Quick EDA ---")
    quick_eda(conn)

    print("\n--- Plots ---")
    plot_order_revenue_trend(conn, out_dir=fig_dir)
    plot_top_categories_by_items(conn, out_dir=fig_dir)
    plot_top_categories_by_revenue(conn, out_dir=fig_dir)
    plot_state_revenue(conn, out_dir=fig_dir)
    plot_payment_type_share(conn, out_dir=fig_dir)

    conn.close()
    print("\n--- Done ---")


if __name__ == "__main__":
    main()

