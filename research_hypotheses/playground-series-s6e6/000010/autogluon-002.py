def preprocess(df: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    numeric_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if {"alpha", "delta"}.issubset(out.columns):
        a_rad = np.deg2rad(out["alpha"].astype(float))
        d_rad = np.deg2rad(out["delta"].astype(float))
        out["alpha_sin"] = np.sin(a_rad)
        out["alpha_cos"] = np.cos(a_rad)
        out["delta_sin"] = np.sin(d_rad)
        out["delta_cos"] = np.cos(d_rad)

    if {"u", "g", "r", "i", "z", "redshift"}.issubset(out.columns):
        out["c_ug"] = out["u"] - out["g"]
        out["c_gr"] = out["g"] - out["r"]
        out["c_ri"] = out["r"] - out["i"]
        out["c_iz"] = out["i"] - out["z"]
        out["c_ui"] = out["u"] - out["i"]
        out["redshift_log1p"] = np.log1p(np.clip(out["redshift"], 0, None))

    if {"u", "g"}.issubset(out.columns):
        out["u_div_g"] = out["u"] / (out["g"] + 1e-8)
    if {"g", "r"}.issubset(out.columns):
        out["g_div_r"] = out["g"] / (out["r"] + 1e-8)
    if {"r", "i"}.issubset(out.columns):
        out["r_div_i"] = out["r"] / (out["i"] + 1e-8)
    if {"i", "z"}.issubset(out.columns):
        out["i_div_z"] = out["i"] / (out["z"] + 1e-8)

    for col in ["galaxy_population", "spectral_type"]:
        if col in out.columns:
            freq = out[col].value_counts(dropna=False)
            out[f"{col}_freq"] = out[col].map(freq).div(len(out))
            out[f"{col}_code"] = pd.Categorical(out[col], categories=freq.index).codes

    if isinstance(aux, pd.DataFrame) and ("class" in aux.columns):
        aux_clean = aux.copy()
        aux_num = [c for c in numeric_cols if c in aux_clean.columns]
        for c in aux_num:
            aux_clean[c] = pd.to_numeric(aux_clean[c], errors="coerce").replace(
                -9999, np.nan
            )

        if {"u", "g"}.issubset(aux_clean.columns) and {"u", "g"}.issubset(out.columns):
            aux_clean["c_ug"] = aux_clean["u"] - aux_clean["g"]
            out["c_ug"] = out["u"] - out["g"]
        if {"g", "r"}.issubset(aux_clean.columns) and {"g", "r"}.issubset(out.columns):
            aux_clean["c_gr"] = aux_clean["g"] - aux_clean["r"]
            out["c_gr"] = out["g"] - out["r"]
        if {"r", "i"}.issubset(aux_clean.columns) and {"r", "i"}.issubset(out.columns):
            aux_clean["c_ri"] = aux_clean["r"] - aux_clean["i"]
            out["c_ri"] = out["r"] - out["i"]
        if {"i", "z"}.issubset(aux_clean.columns) and {"i", "z"}.issubset(out.columns):
            aux_clean["c_iz"] = aux_clean["i"] - aux_clean["z"]
            out["c_iz"] = out["i"] - out["z"]
        if {"u", "i"}.issubset(aux_clean.columns) and {"u", "i"}.issubset(out.columns):
            aux_clean["c_ui"] = aux_clean["u"] - aux_clean["i"]
            out["c_ui"] = out["u"] - out["i"]

        aux_features = [
            c
            for c in [
                "u",
                "g",
                "r",
                "i",
                "z",
                "redshift",
                "alpha",
                "delta",
                "c_ug",
                "c_gr",
                "c_ri",
                "c_iz",
                "c_ui",
            ]
            if c in aux_clean.columns and c in out.columns
        ]
        aux_valid = aux_clean.dropna(subset=aux_features + ["class"])
        if len(aux_valid) > 10 and len(aux_features) >= 2:
            class_center = aux_valid.groupby("class", dropna=False)[
                aux_features
            ].median(numeric_only=True)
            if len(class_center) >= 2:
                scale = aux_valid[aux_features].mad().replace(0, 1.0)
                scale = np.where(scale <= 0, 1.0, scale.to_numpy())
                X = out[aux_features].to_numpy(float)

                dist_cols = []

                def _safe(s: str) -> str:
                    out_name = []
                    for ch in str(s):
                        out_name.append(ch if ch.isalnum() else "_")
                    return "".join(out_name).strip("_").lower()

                for cls, row in class_center.iterrows():
                    center = row.to_numpy(float)
                    d = X - center
                    d = d / scale
                    out[f"aux_dist_to_{_safe(cls)}"] = np.sqrt(np.nansum(d * d, axis=1))
                    dist_cols.append(f"aux_dist_to_{_safe(cls)}")

                if dist_cols:
                    dist_arr = out[dist_cols].to_numpy()
                    out["aux_dist_min"] = np.nanmin(dist_arr, axis=1)
                    if len(dist_cols) >= 2:
                        out["aux_dist_margin"] = (
                            np.nanmax(dist_arr, axis=1) - out["aux_dist_min"]
                        )

    return out
