"""
evaluation.py
--------------

Model havuzu için 5-fold CV, train/test metrikleri ve tahmin vs. gerçek saçılım grafikleri.
Çıktılar:
- Konsola özet metrikler (CV ortalama, Train/Test R2–RMSE–MAE)
- Kaydedilen çoklu saçılım grafiği (ml_results.png)
- Excel'e iki sayfa: ML_Summary, BestModel_Folds
- Fonksiyon dönüşü: sonuç DataFrame'leri ve en iyi pipeline

"""

from typing import Dict, List, Tuple, Optional, Any
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import parallel_backend

from src.config import RANDOM_STATE, N_JOBS, OUT_DIR, OUT_DATA


def _rmse(y_true, y_pred) -> float:
    """Root-Mean-Squared-Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_and_plot(
    models: List[Tuple[str, Optional[Any]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = 5,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
    out_dir: str = OUT_DIR,
    out_data_path: str = OUT_DATA,
    df_meta: Optional[pd.DataFrame] = None,

) -> Dict[str, Any]:
    """
    Verilen (ad, pipeline) model listesi için CV + train/test değerlendirme ve görselleştirme yapar.
    None olan modeller "skipped" olarak geçilir.

    Dönüş:
        {
          "results_df": <tüm metrikler>,
          "results_sorted": <test R2'ye göre sıralı>,
          "best_name": <en iyi model adı>,
          "best_pipe": <en iyi pipeline>,
          "best_fold_df": <en iyi modelin fold metrikleri>,
          "out_fig": <kayıtlı figür yolu>
        }
    """
    # --- CV & scoring tanımı ---
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {
        "r2": "r2",
        "rmse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    # --- Grafik aralığı (tüm veri) ---
    y_all = pd.concat([y_train, y_test]).values
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    pad = 0.05 * (y_max - y_min) if (y_max > y_min) else 1.0
    lo, hi = y_min - pad, y_max + pad
    BLUE = "#1f77b4"
    RED = "#d62728"

    # 2x3 ızgara (6 panel). Model sayısı 6'dan azsa kalan paneller boş bırakılır.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    results: List[Dict[str, float]] = []
    fold_store: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Any]] = {}

    with parallel_backend("threading", n_jobs=n_jobs):
        for ax, (name, pipe) in zip(axes, models):
            if pipe is None:
                ax.axis("off")
                ax.set_title(name)
                continue

            # --- 5-fold CV ---
            cvres = cross_validate(
                pipe, X_train, y_train,
                cv=cv, scoring=scoring,
                n_jobs=n_jobs, return_train_score=False
            )
            fold_r2 = cvres["test_r2"]
            fold_rmse = np.sqrt(-cvres["test_rmse"])
            fold_mae = -cvres["test_mae"]

            cv_r2, cv_rmse, cv_mae = fold_r2.mean(), fold_rmse.mean(), fold_mae.mean()
            print(f"[CV] {name:14s} | R2={cv_r2:.3f} | RMSE={cv_rmse:.3f} | MAE={cv_mae:.3f}")

            # --- Train/Test fit & pred ---
            pipe.fit(X_train, y_train)
            yhat_tr = pipe.predict(X_train)
            yhat_te = pipe.predict(X_test)

            tr_r2, te_r2 = r2_score(y_train, yhat_tr), r2_score(y_test, yhat_te)
            tr_rmse, te_rmse = _rmse(y_train, yhat_tr), _rmse(y_test, yhat_te)
            tr_mae, te_mae = mean_absolute_error(y_train, yhat_tr), mean_absolute_error(y_test, yhat_te)

            results.append({
                "model": name,
                "cv_r2": cv_r2, "cv_rmse": cv_rmse, "cv_mae": cv_mae,
                "train_r2": tr_r2, "train_rmse": tr_rmse, "train_mae": tr_mae,
                "test_r2": te_r2, "test_rmse": te_rmse, "test_mae": te_mae,
            })
            fold_store[name] = (fold_r2, fold_rmse, fold_mae, pipe)

            # --- Saçılım grafikleri ---
            ax.scatter(
                y_train, yhat_tr, s=18, alpha=0.65, color=BLUE,
                label="Train" if name == models[0][0] else None, edgecolors="none"
            )
            ax.scatter(
                y_test, yhat_te, s=30, alpha=0.85, color=RED, marker="^",
                label="Test" if name == models[0][0] else None, edgecolors="none"
            )
            ax.plot([lo, hi], [lo, hi], "--", lw=1.0)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(name)
            ax.set_xlabel("Gerçek qe (mg/g)")
            ax.set_ylabel("Tahmin qe (mg/g)")
            tr_line = f"Train R2={tr_r2:.2f} | RMSE={tr_rmse:.1f} | MAE={tr_mae:.1f}"
            te_line = f"Test  R2={te_r2:.2f} | RMSE={te_rmse:.1f} | MAE={te_mae:.1f}"
            ax.text(
                0.02, 0.98,
                f"{tr_line}\n{te_line}\nCV R2={cv_r2:.2f}, RMSE={cv_rmse:.1f}, MAE={cv_mae:.1f}",
                transform=ax.transAxes, fontsize=8, family="monospace", va="top", ha="left"
            )

    axes[0].legend(loc="lower right")

    # --- Grafiği kaydet ---
    os.makedirs(out_dir, exist_ok=True)
    out_fig = os.path.join(out_dir, "ml_results.png")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    print(f"[OK] Grafik kaydedildi: {out_fig}")
    plt.show()

    # --- Özet & En iyi model ---
    res_df = pd.DataFrame(results)
    if len(res_df):
    # Sıralama: CV-RMSE (küçük), sonra CV-R2 (büyük), sonra Test-R2 (büyük)
        res_df_sorted = res_df.sort_values(
            ["cv_rmse", "cv_r2", "test_r2"],
            ascending=[True, False, False]
    )
        print("\n=== Özet (CV sonuçlarına göre sıralı) ===")
        print(res_df_sorted.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
       
        # En iyi modelin fold detayları
        best_name = res_df_sorted.iloc[0]["model"]
        fold_r2, fold_rmse, fold_mae, best_pipe = fold_store[best_name]

        # Fold validasyon örnek sayıları
        fold_sizes = [len(val_idx) for _, val_idx in KFold(
            n_splits=cv_splits, shuffle=True, random_state=random_state
        ).split(X_train, y_train)]

        best_fold_df = pd.DataFrame({
            "Fold": np.arange(1, len(fold_r2) + 1),
            "n_val": fold_sizes,
            "R2": fold_r2,
            "RMSE": fold_rmse,
            "MAE": fold_mae
        })

        print(f"\n===== En iyi model: {best_name} — {cv_splits}-Fold Detay =====")
        print(best_fold_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(
            "Mean±SD: "
            f"R2={fold_r2.mean():.3f}±{fold_r2.std():.3f} | "
            f"RMSE={fold_rmse.mean():.3f}±{fold_rmse.std():.3f} | "
            f"MAE={fold_mae.mean():.3f}±{fold_mae.std():.3f}"
        )

        # ---- OOF Ayrıntılı Değerlendirme (Target_Phar ile) ----
        df_meta_local = None
        if df_meta is not None:
            # evaluate_and_plot dışarıdan ham tablo alt kümesi verdiyse index uyumla
            df_meta_local = df_meta.loc[X_train.index, ["Target_Phar"]] if "Target_Phar" in df_meta.columns else None

        export_oof_with_pharma(
            best_pipe,
            X_train,
            y_train,
            out_data_path=out_data_path,
            cv=cv_splits,
            n_jobs=n_jobs,
            df_meta=df_meta_local
        )
        # --------------------------------------------------------


        from sklearn.model_selection import cross_val_predict

        def export_oof_with_pharma(best_pipe,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           out_data_path: str,
                           cv=5,
                           n_jobs=-1,
                           df_meta: pd.DataFrame | None = None):
            """
            OOF tahminlerini üretir ve Excel'e iki sayfa yazar:
            1) OOF_Detailed: satır bazında y_true, y_pred, err, abs_err, APE(%) (+ Target_Phar varsa)
            2) OOF_ByDrug: Target_Phar bazında n, MAE, RMSE, MAPE

            Notlar:
                - df_meta: X_train ile aynı index'e sahip ve en az 'Target_Phar' kolonu olan tablo.
                - y_true=0 için APE % NaN bırakılır (bölme hatasını önlemek için).
                - Excel dosyasında aynı sayfalar varsa 'replace' ile güncellenir.
            """

            # 1) OOF tahminleri
            oof_pred = cross_val_predict(best_pipe, X_train, y_train,
                                 cv=cv, n_jobs=n_jobs, method="predict")

            # 2) Satır bazında hata metrikleri
            oof = pd.DataFrame({"y_true": y_train.values,
                        "y_pred": oof_pred}, index=X_train.index)
            oof["err"] = oof["y_pred"] - oof["y_true"]
            oof["abs_err"] = oof["err"].abs()
            oof["sq_err"] = oof["err"] ** 2
            oof["APE_%"] = (oof["abs_err"] / oof["y_true"].replace(0, np.nan)) * 100  # y_true=0 -> NaN

            # 3) Target_Phar ekle (varsa)
            if df_meta is not None and "Target_Phar" in df_meta.columns:
                oof = oof.join(df_meta.loc[oof.index, ["Target_Phar"]])

            # 4) Ayrıntı sayfasını büyük hataya göre sırala
            oof.sort_values("abs_err", ascending=False, inplace=True)

            # 5) Farmasötik bazında özet tablo
            bydrug = None
            if "Target_Phar" in oof.columns:
                bydrug = (oof
                        .groupby("Target_Phar")
                        .agg(n=("y_true", "size"),
                            MAE=("abs_err", "mean"),
                            RMSE=("sq_err", lambda s: np.sqrt(s.mean())),
                            MAPE=("APE_%", "mean"))
                        .sort_values("MAE", ascending=False))

            # 6) Excel'e yaz
            with pd.ExcelWriter(out_data_path, mode="a", if_sheet_exists="replace") as xw:
                oof.to_excel(xw, sheet_name="OOF_Detailed", index=True)
                if bydrug is not None:
                    bydrug.to_excel(xw, sheet_name="OOF_ByDrug", index=True)

            print(f"[OK] OOF ayrıntıları ve farmasötik özetleri yazıldı → {out_data_path}")
# ==================== /OOF AYRINTILI DEĞERLENDİRME ======================


        # Metrikleri Excel'e yaz
        try:
            with pd.ExcelWriter(out_data_path, mode="a", if_sheet_exists="replace") as xw:
                res_df_sorted.to_excel(xw, sheet_name="ML_Summary", index=False)
                best_fold_df.to_excel(xw, sheet_name="BestModel_Folds", index=False)
            print(f"[OK] Metrikler Excel sayfalarına eklendi: {out_data_path}")
        except Exception as e:
            print(f"[Uyarı] Metrikleri Excel'e yazarken sorun: {e}")

        return {
            "results_df": res_df,
            "results_sorted": res_df_sorted,
            "best_name": best_name,
            "best_pipe": best_pipe,
            "best_fold_df": best_fold_df,
            "out_fig": out_fig,
        }

    # (Hiç sonuç yoksa)
    return {
        "results_df": res_df,
        "results_sorted": res_df,
        "best_name": None,
        "best_pipe": None,
        "best_fold_df": None,
        "out_fig": out_fig,
    }
