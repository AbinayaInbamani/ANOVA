import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import t
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="ANOVA App", layout="wide")
st.title("ANOVA App (Variety / Trait / Treatment / Rep / Value)")

REQUIRED = ["Variety", "Trait", "Treatment", "Rep", "Value"]

# -----------------------------
# Formatting helpers
# -----------------------------
def fmt2(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

def fmt_p(x):
    if pd.isna(x):
        return ""
    try:
        x = float(x)
        # Avoid displaying tiny p-values as 0.00
        if x < 0.001:
            return f"{x:.2e}"
        return f"{x:.4f}"
    except Exception:
        return str(x)

def format_anova_table(aov: pd.DataFrame) -> pd.DataFrame:
    """Format ANOVA table with 2 decimals; PR(>F) in scientific when tiny."""
    a = aov.copy()
    for col in a.columns:
        if col == "PR(>F)":
            a[col] = a[col].apply(fmt_p)
        else:
            a[col] = a[col].apply(fmt2)
    return a

# -----------------------------
# Data loading
# -----------------------------
def load_excel_safely(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)

    # Drop columns completely empty
    df = df.dropna(axis=1, how="all")

    # If extra columns exist (Unnamed:*), keep first 5 columns (your format)
    if df.shape[1] > 5:
        df = df.iloc[:, :5]

    # If header missing, pandas may use integer column names
    if all(isinstance(c, int) for c in df.columns) and df.shape[1] == 5:
        df.columns = REQUIRED

    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]

    # If first row is actually header words stored as values
    if df.shape[0] > 0:
        first_row = df.iloc[0].astype(str).str.strip().tolist()
        if set(first_row) >= set(REQUIRED):
            df.columns = first_row[:5]
            df = df.iloc[1:].reset_index(drop=True)

    # Keep only required columns
    df = df[[c for c in REQUIRED if c in df.columns]].copy()

    # Clean text columns
    for c in ["Variety", "Trait", "Treatment", "Rep"]:
        df[c] = df[c].astype(str).str.strip()

    # Numeric Value
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Drop missing rows
    df = df.dropna(subset=REQUIRED).reset_index(drop=True)
    return df

def validate_df(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}. Required: {REQUIRED}"
    if df["Variety"].nunique() < 2:
        return False, "Need at least TWO varieties (e.g., Freedom and Ouchitta) to compare."
    if df["Trait"].nunique() < 1:
        return False, "No Trait values found."
    if df["Treatment"].nunique() < 1 or df["Rep"].nunique() < 1:
        return False, "Treatment/Rep columns look empty."
    return True, "OK"

# -----------------------------
# ANOVA + conclusions
# -----------------------------
def run_anova(df_trait: pd.DataFrame):
    # Must have >=2 varieties at app level; still safe
    formula = "Value ~ C(Variety) * C(Treatment) + C(Rep)"
    model = ols(formula, data=df_trait).fit()
    aov = anova_lm(model, typ=2)
    return formula, model, aov

def get_aov(aov, row, col, default=np.nan):
    if row in aov.index and col in aov.columns:
        try:
            return float(aov.loc[row, col])
        except Exception:
            return default
    return default

def better_word(trait_name: str, direction: str) -> str:
    # direction is "Higher is better" or "Lower is better"
    if direction.lower().startswith("lower"):
        return f"lower mean {trait_name}"
    return f"higher mean {trait_name}"

def treatment_wise_statements(df_trait: pd.DataFrame, trait_name: str, direction: str) -> list:
    """
    Create sentences for each treatment comparing the two varieties using means.
    direction:
      - "Lower is better" -> chooses smaller mean as better
      - "Higher is better" -> chooses larger mean as better
    """
    varieties = sorted(df_trait["Variety"].unique().tolist())
    if len(varieties) < 2:
        return []

    v1, v2 = varieties[0], varieties[1]
    stmts = []

    g = (
        df_trait
        .groupby(["Treatment", "Variety"])["Value"]
        .mean()
        .reset_index()
        .pivot(index="Treatment", columns="Variety", values="Value")
    )

    for tmt in sorted(g.index.astype(str).tolist(), key=lambda x: (len(x), x)):
        if tmt not in g.index:
            continue
        # robust access
        try:
            m1 = float(g.loc[tmt, v1])
            m2 = float(g.loc[tmt, v2])
        except Exception:
            continue

        if direction.lower().startswith("lower"):
            # lower mean is better
            if m1 < m2:
                better, worse, mb, mw = v1, v2, m1, m2
            else:
                better, worse, mb, mw = v2, v1, m2, m1
            stmt = (f"Under Treatment {tmt}, {better} had lower mean {trait_name} "
                    f"({mb:.2f}) than {worse} ({mw:.2f}), indicating better {trait_name.lower()} performance under this treatment.")
        else:
            # higher mean is better
            if m1 > m2:
                better, worse, mb, mw = v1, v2, m1, m2
            else:
                better, worse, mb, mw = v2, v1, m2, m1
            stmt = (f"Under Treatment {tmt}, {better} had higher mean {trait_name} "
                    f"({mb:.2f}) than {worse} ({mw:.2f}), indicating better {trait_name.lower()} performance under this treatment.")
        stmts.append(stmt)

    return stmts

def build_conclusions(df_trait: pd.DataFrame, aov: pd.DataFrame, trait_name: str, direction: str, alpha=0.05):
    means = df_trait.groupby("Variety")["Value"].mean().sort_values(ascending=False)

    # p-values
    p_var = get_aov(aov, "C(Variety)", "PR(>F)")
    p_int = get_aov(aov, "C(Variety):C(Treatment)", "PR(>F)")

    # overall better based on direction
    if direction.lower().startswith("lower"):
        # lower mean overall is better -> sort ascending
        means2 = df_trait.groupby("Variety")["Value"].mean().sort_values(ascending=True)
        best = means2.index[0]
        other = means2.index[1]
    else:
        best = means.index[0]
        other = means.index[1]

    # 9 values (variety effect)
    SS = get_aov(aov, "C(Variety)", "sum_sq")
    dfv = get_aov(aov, "C(Variety)", "df")
    MS = SS / dfv if (not np.isnan(SS) and not np.isnan(dfv) and dfv != 0) else np.nan
    F = get_aov(aov, "C(Variety)", "F")

    nine = pd.DataFrame(
        [
            ("N (observations)", int(df_trait.shape[0])),
            ("Grand Mean", float(df_trait["Value"].mean())),
            (f"Mean (Freedom)" if "Freedom" in df_trait["Variety"].unique() else "Mean (Variety 1)",
             float(df_trait[df_trait["Variety"] == ( "Freedom" if "Freedom" in df_trait["Variety"].unique() else sorted(df_trait["Variety"].unique())[0])]["Value"].mean())),
            (f"Mean (Ouchitta)" if "Ouchitta" in df_trait["Variety"].unique() else "Mean (Variety 2)",
             float(df_trait[df_trait["Variety"] == ( "Ouchitta" if "Ouchitta" in df_trait["Variety"].unique() else sorted(df_trait["Variety"].unique())[1])]["Value"].mean())),
            ("SS(Variety)", SS),
            ("df(Variety)", dfv),
            ("MS(Variety)", MS),
            ("F(Variety)", F),
            ("p-value(Variety)", p_var),
        ],
        columns=["Metric", "Value"]
    )

    # conclusions
    treatment_statements = []

    if not np.isnan(p_int) and p_int < alpha:
        # interaction significant -> depends on treatment + per-treatment statements
        overall = (f"No single variety (Ouchitta or Freedom) is better for {trait_name} across all treatments. "
                   f"The better variety depends on the treatment applied (Variety × Treatment interaction p = {fmt_p(p_int)}).")
        treatment_statements = treatment_wise_statements(df_trait, trait_name, direction)
    else:
        # no significant interaction -> can interpret main effect
        if not np.isnan(p_var) and p_var < alpha:
            if direction.lower().startswith("lower"):
                overall = (f"{best} is better than {other} for {trait_name} because it has a lower mean {trait_name} overall "
                           f"(p = {fmt_p(p_var)}).")
            else:
                overall = (f"{best} is better than {other} for {trait_name} because it has a higher mean {trait_name} overall "
                           f"(p = {fmt_p(p_var)}).")
        else:
            overall = (f"There is no statistically significant difference between the varieties for {trait_name} "
                       f"(p = {fmt_p(p_var)}).")

    return nine, overall, treatment_statements

# -----------------------------
# Plots
# -----------------------------
def plot_ci95_by_variety(df_trait, ylabel):
    g = df_trait.groupby("Variety")["Value"].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"])
    g["tcrit"] = g["count"].apply(lambda n: t.ppf(0.975, df=n - 1) if n > 1 else np.nan)
    g["ci95"] = g["tcrit"] * g["se"]

    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, g["mean"])
    ax.errorbar(x, g["mean"], yerr=g["ci95"], fmt="o", capsize=6, linewidth=2, color="black")

    for i, (mean, ci, n) in enumerate(zip(g["mean"], g["ci95"], g["count"])):
        ax.text(i + 0.05, mean + (ci if not np.isnan(ci) else 0), f"n = {int(n)}", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(g["Variety"].astype(str))
    ax.set_ylabel(ylabel)
    ax.set_title("Confidence Interval (95%)")
    ax.legend(["Confidence Interval (95%)"], loc="upper left", frameon=True)
    ax.grid(False)
    st.pyplot(fig, clear_figure=True)

def interaction_plot(df_trait, ylabel):
    pivot = df_trait.pivot_table(index="Treatment", columns="Variety", values="Value", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in pivot.columns:
        ax.plot(pivot.index.astype(str), pivot[v], marker="o", label=str(v))
    ax.set_title(f"Interaction Plot (Means): {ylabel}")
    ax.set_xlabel("Treatment")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    st.pyplot(fig, clear_figure=True)

def readable_boxplot(df_trait, trait_name):
    df = df_trait.copy()
    df["TxV"] = df["Treatment"].astype(str) + " – " + df["Variety"].astype(str)

    # Order: T1–Freedom, T1–Ouchitta, T2–Freedom, ...
    order = []
    for tmt in sorted(df["Treatment"].unique(), key=lambda x: (len(str(x)), str(x))):
        for v in sorted(df["Variety"].unique()):
            lbl = f"{tmt} – {v}"
            if (df["TxV"] == lbl).any():
                order.append(lbl)

    data = [df.loc[df["TxV"] == lbl, "Value"].values for lbl in order]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=order, showmeans=True)
    ax.set_title(f"{trait_name}: Treatment × Variety")
    ax.set_xlabel("Treatment – Variety")
    ax.set_ylabel(trait_name)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    st.pyplot(fig, clear_figure=True)

# -----------------------------
# UI
# -----------------------------
uploaded = st.file_uploader(
    "Upload Excel (.xlsx) with 5 columns: Variety, Trait, Treatment, Rep, Value (header optional)",
    type=["xlsx"]
)

if not uploaded:
    st.info("Upload your Excel file to continue.")
    st.stop()

df = load_excel_safely(uploaded)
ok, msg = validate_df(df)
if not ok:
    st.error(msg)
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(30), use_container_width=True)

traits = sorted(df["Trait"].unique().tolist())
analyze_all = st.checkbox("Analyze ALL traits in this file", value=False)

# Direction control (for correct 'better' wording)
direction = st.selectbox(
    "When we say 'better', which direction should we use for this trait?",
    options=["Higher is better", "Lower is better"],
    index=0
)
st.caption("Example: Yield → Higher is better.  Acidity → Lower is better.")

st.markdown(
    "**Model formula used:**  \n"
    "`Value ~ C(Variety) * C(Treatment) + C(Rep)`  \n"
    "**What is `C()`?** `C(Variety)` means Variety is treated as a **categorical factor** (group labels), so ANOVA compares **group means**."
)

def render_trait_block(trait_name: str):
    st.markdown(f"---\n## Trait: {trait_name}")

    df_trait = df[df["Trait"].str.lower() == trait_name.lower()].copy()

    # Checks table
    st.markdown("### Data checks")
    st.write("Varieties:", sorted(df_trait["Variety"].unique().tolist()))
    st.write("Treatments:", sorted(df_trait["Treatment"].unique().tolist()))
    st.write("Reps:", sorted(df_trait["Rep"].unique().tolist()))
    st.dataframe(df_trait.groupby(["Variety", "Treatment"]).size().unstack(fill_value=0), use_container_width=True)

    # ANOVA
    formula, model, aov = run_anova(df_trait)
    st.markdown("### ANOVA table (formatted)")
    st.dataframe(format_anova_table(aov), use_container_width=True)

    # Conclusions + treatment-wise statements
    nine, overall, t_statements = build_conclusions(df_trait, aov, trait_name, direction, alpha=0.05)

    # Format 9 values: 2 decimals, but p-values in scientific
    nine_show = nine.copy()
    nine_show["Value"] = nine_show.apply(
        lambda r: fmt_p(r["Value"]) if "p-value" in str(r["Metric"]).lower() else fmt2(r["Value"]),
        axis=1
    )

    st.markdown("### Key output values")
    st.dataframe(nine_show, use_container_width=True)

    st.markdown("### Conclusion")
    st.success(overall)

    if t_statements:
        st.markdown("### Treatment-wise conclusions (because interaction is significant)")
        for s in t_statements:
            st.write("- " + s)

    # Plots
    st.markdown("### Plots")
    plot_ci95_by_variety(df_trait, ylabel=trait_name)
    interaction_plot(df_trait, ylabel=trait_name)
    readable_boxplot(df_trait, trait_name=trait_name)

    return {
        "trait": trait_name,
        "anova": aov,
        "nine": nine_show,
        "overall": overall,
        "treatment_statements": t_statements,
        "data": df_trait
    }

results = []

if analyze_all:
    for tr in traits:
        results.append(render_trait_block(tr))
else:
    trait = st.selectbox("Select Trait", traits)
    results.append(render_trait_block(trait))

# -----------------------------
# Download output workbook
# -----------------------------
st.markdown("---\n## Download Output (Excel)")

out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    for r in results:
        tr = r["trait"]
        safe_tr = "".join([c for c in tr if c.isalnum()])[:25] or "Trait"

        r["data"].to_excel(writer, index=False, sheet_name=f"{safe_tr}_Data")
        r["anova"].to_excel(writer, sheet_name=f"{safe_tr}_ANOVA")

        # Key values + conclusion + treatment-wise statements
        r["nine"].to_excel(writer, index=False, sheet_name=f"{safe_tr}_KeyValues")
        pd.DataFrame({"Conclusion": [r["overall"]]}).to_excel(writer, index=False, sheet_name=f"{safe_tr}_Conclusion")

        if r["treatment_statements"]:
            pd.DataFrame({"Treatment-wise statements": r["treatment_statements"]}).to_excel(
                writer, index=False, sheet_name=f"{safe_tr}_ByTreatment"
            )

st.download_button(
    "Download results as Excel",
    data=out.getvalue(),
    file_name="ANOVA_All_Traits_Output.xlsx" if analyze_all else "ANOVA_Trait_Output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
