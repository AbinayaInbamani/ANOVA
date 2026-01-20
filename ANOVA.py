import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import t
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

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
        # show scientific for very small p
        if x < 0.001:
            return f"{x:.2e}"
        return f"{x:.4f}"
    except Exception:
        return str(x)

# -----------------------------
# Data loading
# -----------------------------
def load_excel_safely(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)
    df = df.dropna(axis=1, how="all")

    # if extra empty columns exist, keep first 5 columns (your format)
    if df.shape[1] > 5:
        df = df.iloc[:, :5]

    # if header missing, pandas may use 0..4
    if all(isinstance(c, int) for c in df.columns) and df.shape[1] == 5:
        df.columns = REQUIRED

    df.columns = [str(c).strip() for c in df.columns]

    # if first row contains header words, fix
    if df.shape[0] > 0:
        first_row = df.iloc[0].astype(str).str.strip().tolist()
        if set(first_row) >= set(REQUIRED):
            df.columns = first_row[:5]
            df = df.iloc[1:].reset_index(drop=True)

    # keep required columns
    df = df[[c for c in REQUIRED if c in df.columns]].copy()

    # clean strings
    for c in ["Variety", "Trait", "Treatment", "Rep"]:
        df[c] = df[c].astype(str).str.strip()

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=REQUIRED).reset_index(drop=True)

    return df

def validate_df(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}. Required: {REQUIRED}"
    if df["Variety"].nunique() < 1:
        return False, "No varieties found."
    if df["Trait"].nunique() < 1:
        return False, "No traits found."
    return True, "OK"

# -----------------------------
# ANOVA
# -----------------------------
def run_anova(df_trait: pd.DataFrame):
    n_var = df_trait["Variety"].nunique()
    if n_var >= 2:
        formula = "Value ~ C(Variety) * C(Treatment) + C(Rep)"
    else:
        formula = "Value ~ C(Treatment) + C(Rep)"
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

def nine_values_and_conclusion(df_trait, aov, trait_name, alpha=0.05):
    n_var = df_trait["Variety"].nunique()
    N = int(df_trait.shape[0])
    grand_mean = float(df_trait["Value"].mean())

    # If only one variety: no “better variety”
    if n_var < 2:
        # best treatment within this one variety
        best_t = (
            df_trait.groupby("Treatment")["Value"].mean()
            .sort_values(ascending=False).index[0]
        )
        nine = pd.DataFrame(
            [
                ("N (observations)", N),
                ("Grand Mean", grand_mean),
                ("Note", "Only one variety present → cannot compare varieties."),
            ],
            columns=["Metric", "Value"]
        )
        conclusion = (
            f"Only one variety found in the uploaded file. "
            f"For {trait_name}, the best mean treatment is {best_t} for this variety."
        )
        return nine, conclusion

    # Means by variety
    means = df_trait.groupby("Variety")["Value"].mean().sort_values(ascending=False)
    best = means.index[0]
    other = means.index[1]

    # ANOVA pieces
    SS = get_aov(aov, "C(Variety)", "sum_sq")
    dfv = get_aov(aov, "C(Variety)", "df")
    MS = SS / dfv if (not np.isnan(SS) and not np.isnan(dfv) and dfv != 0) else np.nan
    F = get_aov(aov, "C(Variety)", "F")
    p = get_aov(aov, "C(Variety)", "PR(>F)")

    p_int = get_aov(aov, "C(Variety):C(Treatment)", "PR(>F)")

    nine = pd.DataFrame(
        [
            ("N (observations)", N),
            ("Grand Mean", grand_mean),
            (f"Mean ({best})", float(means.iloc[0])),
            (f"Mean ({other})", float(means.iloc[1])),
            ("SS(Variety)", SS),
            ("df(Variety)", dfv),
            ("MS(Variety)", MS),
            ("F(Variety)", F),
            ("p-value(Variety)", p),
        ],
        columns=["Metric", "Value"]
    )

    # Clear conclusion
    if not np.isnan(p_int) and p_int < alpha:
        conclusion = (
            f"Better variety depends on treatment for {trait_name} "
            f"(Variety × Treatment interaction p = {fmt_p(p_int)})."
        )
    elif not np.isnan(p) and p < alpha:
        conclusion = (
            f"{best} is better than {other} for {trait_name} "
            f"(p = {fmt_p(p)} < {alpha})."
        )
    else:
        conclusion = (
            f"No significant difference between {best} and {other} for {trait_name} "
            f"(p = {fmt_p(p)} ≥ {alpha})."
        )

    return nine, conclusion

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

def qq_plot_only(model, ylabel):
    fig = plt.figure()
    sm.qqplot(model.resid, line="45", fit=True)
    plt.title(f"Q-Q Plot: {ylabel}")
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

def boxplot(df_trait, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5))
    df_trait.boxplot(column="Value", by=["Treatment", "Variety"], grid=False, ax=ax)
    ax.set_title(f"Boxplot: {ylabel} by Treatment × Variety")
    plt.suptitle("")
    ax.set_ylabel(ylabel)
    st.pyplot(fig, clear_figure=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ANOVA App", layout="wide")
st.title("ANOVA App (Variety / Trait / Treatment / Rep / Value)")

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
trait = st.selectbox("Select Trait", traits)

df_trait = df[df["Trait"].str.lower() == trait.lower()].copy()

st.markdown("### Data checks")
st.write("Varieties:", sorted(df_trait["Variety"].unique().tolist()))
st.write("Treatments:", sorted(df_trait["Treatment"].unique().tolist()))
st.write("Reps:", sorted(df_trait["Rep"].unique().tolist()))
st.dataframe(df_trait.groupby(["Variety","Treatment"]).size().unstack(fill_value=0), use_container_width=True)

st.markdown("## ANOVA")
formula, model, aov = run_anova(df_trait)

st.caption(f"Model: `{formula}`")
st.markdown(
    "**What does `C()` mean?**  \n"
    "`C(Variety)` tells the ANOVA model to treat *Variety* as a **categorical factor** (group labels), "
    "so it compares **group means** instead of fitting a numeric trend."
)

st.dataframe(aov, use_container_width=True)

nine_df, conclusion = nine_values_and_conclusion(df_trait, aov, trait, alpha=0.05)

# Format the 9 outputs: 2 decimals for most, scientific for p-values
nine_show = nine_df.copy()
nine_show["Value"] = nine_show.apply(
    lambda r: fmt_p(r["Value"]) if "p-value" in str(r["Metric"]).lower() else fmt2(r["Value"]),
    axis=1
)

st.markdown("## 9 Output Values + Conclusion")
st.dataframe(nine_show, use_container_width=True)
st.success(conclusion)

st.markdown("## Plots")
if df_trait["Variety"].nunique() >= 2:
    plot_ci95_by_variety(df_trait, ylabel=trait)
    interaction_plot(df_trait, ylabel=trait)
    boxplot(df_trait, ylabel=trait)
else:
    st.info("Only one variety present: CI-by-variety and interaction plot not available.")

st.markdown("## Diagnostics")
st.write("Q–Q Plot (normality check)")
qq_plot_only(model, ylabel=trait)

# Download output
st.markdown("## Download Output")
out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    df_trait.to_excel(writer, index=False, sheet_name="Filtered_Data")
    aov.to_excel(writer, sheet_name="ANOVA_Table")
    nine_show.to_excel(writer, index=False, sheet_name="Nine_Values_Formatted")
    pd.DataFrame({"Conclusion": [conclusion]}).to_excel(writer, index=False, sheet_name="Conclusion")

st.download_button(
    "Download results as Excel",
    data=out.getvalue(),
    file_name=f"ANOVA_Output_{trait}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
