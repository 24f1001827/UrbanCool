# src/models/pinn.py

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# ── Constants ────────────────────────────────────────────────────────────────
STEFAN_BOLTZMANN = 5.67e-8   # W/m²/K⁴
EMISSIVITY        = 0.95     # urban surface emissivity
RHO_CP            = 1200.0   # volumetric heat capacity of air J/m³/K
PRIESTLEY_TAYLOR  = 0.6      # LE fraction of Rn for vegetated pixels
BASTIAANSSEN_G    = 0.31     # G fraction of Rn for bare/urban pixels

TARGET_COL = "LST_C"

PHYSICS_COLS = [
    "NET_SOLAR_RADIATION_W_M2",
    "AIR_TEMP_C",
    "WIND_SPEED_M_S",
    "NDVI",
]

DROP_COLS = [
    "longitude", "latitude",
    "LST_C",
    "DW_LABEL",
    "ECOSTRESS_LST_C",
    "SENSIBLE_HEAT_FLUX_W_M2",
    "LATENT_HEAT_FLUX_W_M2",
    "NET_THERMAL_RADIATION_W_M2",
    "DEWPOINT_C",
    "SURFACE_PRESSURE_HPA",
    "SOIL_MOISTURE_L1",
    "DW_SNOW_ICE_PROB",
    "DW_FLOODED_VEGETATION_PROB",
    "DW_CROPS_PROB",
    "PRECIPITATION_MM",
]


# ── Physics Head ─────────────────────────────────────────────────────────────
class PhysicsHead(nn.Module):
    """
    Computes the surface energy balance residual given predicted LST
    and physics context features.

    Energy balance: Rn = H + LE + G
    Residual = |Rn - H - LE - G|  → should be zero if physics is satisfied
    """

    def forward(
        self,
        lst_pred: torch.Tensor,      # (B,) predicted LST in °C
        net_solar: torch.Tensor,     # (B,) NET_SOLAR_RADIATION_W_M2
        air_temp_c: torch.Tensor,    # (B,) AIR_TEMP_C
        wind_speed: torch.Tensor,    # (B,) WIND_SPEED_M_S
        ndvi: torch.Tensor,          # (B,) NDVI
    ) -> torch.Tensor:               # (B,) energy balance residual

        lst_k     = lst_pred + 273.15
        t_air_k   = air_temp_c + 273.15

        # Net radiation
        rn = net_solar - EMISSIVITY * STEFAN_BOLTZMANN * lst_k ** 4

        # Aerodynamic resistance (s/m) — bulk formula
        ra = 1.0 / (0.01 * wind_speed.clamp(min=0.1) + 0.001)

        # Sensible heat flux
        h = RHO_CP * (lst_k - t_air_k) / ra.clamp(min=1.0)

        # Latent heat flux (Priestley-Taylor, NDVI-weighted)
        ndvi_clamped = ndvi.clamp(0.0, 1.0)
        le = PRIESTLEY_TAYLOR * rn * ndvi_clamped

        # Ground heat flux (Bastiaanssen 1995)
        g = BASTIAANSSEN_G * rn * (1.0 - ndvi_clamped)

        # Residual — energy that is unaccounted for
        residual = rn - h - le - g
        return residual


# ── Neural Network ───────────────────────────────────────────────────────────
class LSTNet(nn.Module):
    """
    Feedforward network predicting LST from land surface features.
    Kept shallow intentionally — 30K training samples don't need depth.
    """

    def __init__(self, n_features: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


# ── PINN Loss ────────────────────────────────────────────────────────────────
class PINNLoss(nn.Module):
    """
    Total loss = MSE(pred, true) + lambda_physics * mean(residual²)
    lambda_physics controls physics regularization strength.
    Start at 0.1, increase if overfitting persists.
    """

    def __init__(self, lambda_physics: float = 0.1):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.physics_head   = PhysicsHead()
        self.mse            = nn.MSELoss()

    def forward(
        self,
        lst_pred: torch.Tensor,
        lst_true: torch.Tensor,
        net_solar: torch.Tensor,
        air_temp_c: torch.Tensor,
        wind_speed: torch.Tensor,
        ndvi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        data_loss    = self.mse(lst_pred, lst_true)
        residual     = self.physics_head(lst_pred, net_solar, air_temp_c, wind_speed, ndvi)
        residual_scale = residual.detach().abs().mean() + 1e-6
        residual_norm = residual / residual_scale
        physics_loss = (residual_norm ** 2).mean()
        total_loss   = data_loss + self.lambda_physics * physics_loss

        return total_loss, data_loss, physics_loss


# ── Data Preparation ─────────────────────────────────────────────────────────
def prepare_data(df: pd.DataFrame):
    """
    Returns:
        X_arr          : numpy array of ML input features
        y_arr          : numpy array of LST target
        physics_arr    : numpy array of physics columns (same row order)
        feature_cols   : list of feature column names
    """
    df = df.copy()

    # Physics columns — extract before dropping
    missing_physics = [c for c in PHYSICS_COLS if c not in df.columns]
    if missing_physics:
        raise ValueError(f"Missing physics columns: {missing_physics}")
    physics_arr = df[PHYSICS_COLS].values.astype(np.float32)

    # Target
    y_arr = df[TARGET_COL].values.astype(np.float32)

    # ML features — drop everything that shouldn't be a feature
    drop = [c for c in DROP_COLS if c in df.columns]
    # Keep PHYSICS_COLS that are also valid features (e.g. NDVI)
    # but drop the ones that are context-only
    context_only = ["NET_SOLAR_RADIATION_W_M2", "AIR_TEMP_C", "WIND_SPEED_M_S"]
    drop += [c for c in context_only if c in df.columns]
    drop = list(set(drop))

    feature_df   = df.drop(columns=drop, errors="ignore")
    feature_cols = list(feature_df.columns)
    X_arr        = feature_df.values.astype(np.float32)

    return X_arr, y_arr, physics_arr, feature_cols


# ── Spatial Split (reuse same logic as XGBoost script) ───────────────────────
def spatial_train_val_test_split(
    df: pd.DataFrame,
    n_lat_bins: int = 5,
    n_lon_bins: int = 5,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    rng      = np.random.default_rng(seed)
    lat_bin  = pd.cut(df["latitude"],  bins=n_lat_bins, labels=False)
    lon_bin  = pd.cut(df["longitude"], bins=n_lon_bins, labels=False)
    df       = df.copy()
    df["_grid_cell"] = lat_bin.astype(str) + "_" + lon_bin.astype(str)

    unique_cells = np.array(df["_grid_cell"].unique().tolist())
    rng.shuffle(unique_cells)

    n_cells = len(unique_cells)
    n_test  = max(1, int(n_cells * test_frac))
    n_val   = max(1, int(n_cells * val_frac))

    test_cells  = set(unique_cells[:n_test])
    val_cells   = set(unique_cells[n_test:n_test + n_val])
    train_cells = set(unique_cells[n_test + n_val:])

    train_df = df[df["_grid_cell"].isin(train_cells)].drop(columns=["_grid_cell"])
    val_df   = df[df["_grid_cell"].isin(val_cells)].drop(columns=["_grid_cell"])
    test_df  = df[df["_grid_cell"].isin(test_cells)].drop(columns=["_grid_cell"])

    return train_df, val_df, test_df


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_pinn(
    csv_path: Path,
    model_out: Path,
    scaler_out: Path,
    lambda_physics: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 256,
    max_epochs: int = 200,
    patience: int = 20,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load & split ──
    df = pd.read_csv(csv_path).dropna(subset=[TARGET_COL])
    train_df, val_df, test_df = spatial_train_val_test_split(df, n_lat_bins=10, n_lon_bins=10, seed=seed)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # ── Prepare arrays ──
    X_train, y_train, phys_train, feature_cols = prepare_data(train_df)
    X_val,   y_val,   phys_val,   _            = prepare_data(val_df)
    X_test,  y_test,  phys_test,  _            = prepare_data(test_df)

    # ── Scale features ── (critical for NN, not needed for XGBoost)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── Tensors ──
    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32).to(device)

    X_tr, y_tr, p_tr = to_tensor(X_train), to_tensor(y_train), to_tensor(phys_train)
    X_v,  y_v,  p_v  = to_tensor(X_val),   to_tensor(y_val),   to_tensor(phys_val)
    X_te, y_te, p_te = to_tensor(X_test),  to_tensor(y_test),  to_tensor(phys_test)

    # ── DataLoader ──
    train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr, p_tr)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # ── Model & loss ──
    n_features = X_train.shape[1]
    model      = LSTNet(n_features).to(device)
    criterion  = PINNLoss(lambda_physics=lambda_physics).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, #verbose=True
    )

    # ── Training ──
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for X_batch, y_batch, p_batch in train_loader:
            optimizer.zero_grad()
            lst_pred = model(X_batch)
            total, data_l, phys_l = criterion(
                lst_pred, y_batch,
                net_solar  = p_batch[:, 0],
                air_temp_c = p_batch[:, 1],
                wind_speed = p_batch[:, 2],
                ndvi       = p_batch[:, 3],
            )
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            lst_val_pred = model(X_v)
            val_total, val_data, val_phys = criterion(
                lst_val_pred, y_v,
                net_solar  = p_v[:, 0],
                air_temp_c = p_v[:, 1],
                wind_speed = p_v[:, 2],
                ndvi       = p_v[:, 3],
            )

        scheduler.step(val_total)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"val_total={val_total:.4f} "
                f"val_data={val_data:.4f} "
                f"val_phys={val_phys:.4f}"
            )

        if val_total < best_val_loss:
            best_val_loss     = val_total.item()
            epochs_no_improve = 0
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── Restore best & evaluate ──
    model.load_state_dict(best_state)
    model.eval()

    def evaluate(X_t, y_t, p_t, split_name):
        with torch.no_grad():
            preds = model(X_t).cpu().numpy()
        true  = y_t.cpu().numpy()
        rmse  = mean_squared_error(true, preds) ** 0.5
        mae   = mean_absolute_error(true, preds)
        r2    = r2_score(true, preds)
        print(f"[{split_name}] RMSE={rmse:.3f} C  MAE={mae:.3f} C  R2={r2:.3f}")
        return preds

    print("\n── Final Evaluation ──")
    evaluate(X_tr, y_tr, p_tr, "train")
    evaluate(X_v,  y_v,  p_v,  "val")
    evaluate(X_te, y_te, p_te, "test")

    # ── Save ──
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "n_features":  n_features,
        "feature_cols": feature_cols,
        "lambda_physics": lambda_physics,
    }, model_out)
    joblib.dump(scaler, scaler_out)
    print(f"\nSaved model → {model_out}")
    print(f"Saved scaler → {scaler_out}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_pinn(
        csv_path    = Path("data/processed/kolkata_training_sample.csv"),
        model_out   = Path("outputs/pinn_model.pt"),
        scaler_out  = Path("outputs/pinn_scaler.joblib"),
        lambda_physics = 0.005,
    )