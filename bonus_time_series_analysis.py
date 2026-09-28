"""Mission 7 보너스: 시계열 분해와 간단 baseline 예측.

전처리된 CSV만 사용해 월별 평균 일일수요량을 추세·계절성·잔차로 분해하고,
마지막 30일을 대상으로 설명 가능한 baseline 예측을 비교한다.
복잡한 예측 모델이 아니라 학습용 기준선을 만드는 것이 목적이다.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


DAILY_COLUMNS = ["date", "year", "month", "daily_demand_gwh", "source_file"]
MONTHLY_COLUMNS = [
    "month_start",
    "year",
    "month",
    "monthly_demand_gwh",
    "day_count",
]


def parse_arguments() -> argparse.Namespace:
    """입력 폴더와 결과 폴더를 실행할 때 바꿀 수 있게 한다."""
    parser = argparse.ArgumentParser(
        description="보너스 시계열 분해와 baseline 예측 결과를 생성합니다."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="전처리 CSV가 있는 폴더 (기본값: data/processed)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="결과 저장 폴더 (기본값: results)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="그래프를 화면에 띄우지 않음. 자동 실행 환경에서 사용",
    )
    return parser.parse_args()


def configure_korean_font() -> str:
    """그래프 제목과 축 이름의 한글 깨짐을 줄이기 위해 한글 폰트를 설정한다."""
    preferred_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"그래프 한글 폰트: {font_name}")
            return font_name

    warnings.warn(
        "권장 한글 폰트를 찾지 못했습니다. 한글이 깨질 수 있습니다.",
        stacklevel=2,
    )
    plt.rcParams["axes.unicode_minus"] = False
    return "기본 sans-serif"


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전처리된 일별·월별 CSV만 읽고 컬럼 구조를 확인한다."""
    daily_path = data_dir / "daily_demand.csv"
    monthly_path = data_dir / "monthly_demand.csv"

    if not daily_path.exists():
        raise FileNotFoundError(f"일별 전처리 파일이 없습니다: {daily_path.resolve()}")
    if not monthly_path.exists():
        raise FileNotFoundError(f"월별 전처리 파일이 없습니다: {monthly_path.resolve()}")

    daily = pd.read_csv(daily_path, encoding="utf-8-sig", parse_dates=["date"])
    monthly = pd.read_csv(monthly_path, encoding="utf-8-sig", parse_dates=["month_start"])

    if list(daily.columns) != DAILY_COLUMNS:
        raise ValueError(f"daily_demand.csv 컬럼이 예상과 다릅니다: {list(daily.columns)}")
    if list(monthly.columns) != MONTHLY_COLUMNS:
        raise ValueError(f"monthly_demand.csv 컬럼이 예상과 다릅니다: {list(monthly.columns)}")

    daily = daily.sort_values("date").reset_index(drop=True)
    monthly = monthly.sort_values("month_start").reset_index(drop=True)

    print("\n보너스 분석 입력 데이터 확인")
    print("=" * 72)
    print(
        f"daily: {len(daily):,}행 | {daily['date'].min().date()} ~ "
        f"{daily['date'].max().date()}"
    )
    print(
        f"monthly: {len(monthly):,}행 | {monthly['month_start'].min().date()} ~ "
        f"{monthly['month_start'].max().date()}"
    )
    return daily, monthly


def validate_monthly_continuity(monthly: pd.DataFrame) -> None:
    """시계열 분해는 월이 빠지면 의미가 흔들리므로 연속 월인지 먼저 확인한다."""
    expected_months = pd.date_range(
        monthly["month_start"].min(),
        monthly["month_start"].max(),
        freq="MS",
    )
    actual_months = pd.DatetimeIndex(monthly["month_start"])
    missing_months = expected_months.difference(actual_months)
    duplicate_count = monthly["month_start"].duplicated().sum()

    if duplicate_count:
        raise ValueError(f"monthly_demand.csv에 중복 월이 있습니다: {duplicate_count:,}개")
    if len(missing_months):
        preview = ", ".join(month.strftime("%Y-%m") for month in missing_months[:20])
        raise ValueError(f"monthly_demand.csv에 누락 월이 있습니다: {preview}")

    print(
        f"월별 연속성 확인: {expected_months[0].strftime('%Y-%m')} ~ "
        f"{expected_months[-1].strftime('%Y-%m')}, {len(expected_months):,}개월 연속"
    )


def validate_daily_continuity(daily: pd.DataFrame) -> None:
    """마지막 30일 테스트가 실제 연속 날짜인지 확인한다."""
    expected_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    missing_dates = expected_dates.difference(pd.DatetimeIndex(daily["date"]))
    duplicate_count = daily["date"].duplicated().sum()

    if duplicate_count:
        raise ValueError(f"daily_demand.csv에 중복 날짜가 있습니다: {duplicate_count:,}개")
    if len(missing_dates):
        preview = ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates[:20])
        raise ValueError(f"daily_demand.csv에 누락 날짜가 있습니다: {preview}")


def save_and_show_figure(figure: plt.Figure, path: Path, show_graphs: bool) -> None:
    """PNG로 저장하고, 기본 실행에서는 화면에도 보여준다."""
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    if show_graphs:
        plt.show()
    plt.close(figure)


def run_time_series_decomposition(
    monthly: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    show_graphs: bool,
) -> tuple[Path, Path, pd.DataFrame]:
    """월별 평균 일일수요량을 추세, 계절성, 잔차로 분해한다."""
    validate_monthly_continuity(monthly)

    data = monthly.copy()
    # 월마다 일수가 다르므로 월 총수요량 대신 월별 평균 일일수요량을 사용한다.
    data["average_daily_demand_gwh"] = data["monthly_demand_gwh"] / data["day_count"]
    series = data.set_index("month_start")["average_daily_demand_gwh"].asfreq("MS")

    decomposition = seasonal_decompose(series, model="additive", period=12)
    summary = pd.DataFrame(
        {
            "month_start": series.index,
            "observed": decomposition.observed.to_numpy(),
            "trend": decomposition.trend.to_numpy(),
            "seasonal": decomposition.seasonal.to_numpy(),
            "residual": decomposition.resid.to_numpy(),
        }
    )

    table_path = tables_dir / "bonus_decomposition_summary.csv"
    summary.to_csv(table_path, index=False, encoding="utf-8-sig")

    figure, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    panels = [
        ("원본 월별 평균 일일수요량", "observed", "#2F6690"),
        ("추세 Trend", "trend", "#5B8E7D"),
        ("계절성 Seasonal", "seasonal", "#C9A66B"),
        ("잔차 Residual", "residual", "#5B6770"),
    ]
    for axis, (title, column, color) in zip(axes, panels):
        axis.plot(summary["month_start"], summary[column], color=color, linewidth=1.8)
        axis.set_title(title)
        axis.set_ylabel("GWh/일")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("월")
    figure.suptitle("월별 평균 일일 전력수요량 시계열 분해", y=1.02, fontsize=14)
    figure_path = figures_dir / "bonus_time_series_decomposition.png"
    save_and_show_figure(figure, figure_path, show_graphs)

    trend_nan_count = int(summary["trend"].isna().sum())
    residual_nan_count = int(summary["residual"].isna().sum())
    seasonal_by_month = (
        summary.assign(month=summary["month_start"].dt.month)
        .groupby("month", as_index=False)["seasonal"]
        .mean()
        .sort_values("seasonal", ascending=False)
    )
    high_months = seasonal_by_month.head(3)
    low_months = seasonal_by_month.tail(3).sort_values("seasonal")

    print("\n보너스 1: 시계열 분해")
    print("=" * 72)
    print(
        f"분해 데이터 기간: {series.index.min().date()} ~ {series.index.max().date()} | "
        f"월 수: {series.size:,} | 계절 주기: 12개월"
    )
    print(f"trend NaN 수: {trend_nan_count:,} | residual NaN 수: {residual_nan_count:,}")
    print("계절성 성분이 큰 달:")
    print(high_months.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print("계절성 성분이 작은 달:")
    print(low_months.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print("※ 계절성 성분은 반복 패턴을 분리한 값이며 원인을 단정하지 않습니다.")

    return table_path, figure_path, summary


def calculate_previous_year_date(date_value: pd.Timestamp) -> tuple[pd.Timestamp, str]:
    """전년 같은 날짜를 찾되, 2월 29일은 전년 2월 28일로 처리한다."""
    try:
        return date_value.replace(year=date_value.year - 1), "exact"
    except ValueError:
        return date_value.replace(year=date_value.year - 1, day=28), "feb29_to_feb28"


def evaluate_forecast(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """MAE, MAPE, RMSE를 계산한다."""
    error = actual - forecast
    absolute_error = error.abs()
    absolute_percentage_error = absolute_error / actual * 100
    rmse = float(np.sqrt(np.mean(np.square(error))))

    return {
        "MAE": float(absolute_error.mean()),
        "MAPE": float(absolute_percentage_error.mean()),
        "RMSE": rmse,
    }


def run_baseline_forecast(
    daily: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    show_graphs: bool,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    """마지막 30일을 테스트로 두고 두 가지 baseline 예측을 비교한다."""
    validate_daily_continuity(daily)

    if len(daily) < 395:
        raise ValueError("전년 같은 날짜 baseline을 계산하기에 일별 데이터가 너무 적습니다.")

    train = daily.iloc[:-30].copy()
    test = daily.iloc[-30:].copy()
    recent_window = train.tail(30)
    recent_mean = recent_window["daily_demand_gwh"].mean()

    # Baseline A: 테스트 직전 30일 평균을 모든 테스트 날짜의 예측값으로 사용한다.
    result = test[["date", "daily_demand_gwh"]].rename(
        columns={"daily_demand_gwh": "actual_daily_demand_gwh"}
    )
    result["forecast_recent_30day_mean"] = recent_mean

    # Baseline B: 전년 같은 날짜 값을 예측값으로 사용한다.
    demand_by_date = daily.set_index("date")["daily_demand_gwh"]
    previous_year_dates = []
    match_notes = []
    previous_year_forecasts = []
    for date_value in result["date"]:
        previous_date, note = calculate_previous_year_date(date_value)
        previous_year_dates.append(previous_date)
        match_notes.append(note)
        previous_year_forecasts.append(demand_by_date.get(previous_date, np.nan))

    result["previous_year_reference_date"] = previous_year_dates
    result["previous_year_match_note"] = match_notes
    result["forecast_previous_year_same_day"] = previous_year_forecasts

    missing_previous_year = result["forecast_previous_year_same_day"].isna().sum()
    if missing_previous_year:
        print(
            f"전년 같은 날짜 예측에서 매칭되지 않은 날짜 {missing_previous_year:,}개는 "
            "평가에서 제외합니다."
        )

    for baseline in ["recent_30day_mean", "previous_year_same_day"]:
        forecast_column = f"forecast_{baseline}"
        result[f"error_{baseline}"] = result["actual_daily_demand_gwh"] - result[forecast_column]
        result[f"absolute_error_{baseline}"] = result[f"error_{baseline}"].abs()
        result[f"absolute_percentage_error_{baseline}"] = (
            result[f"absolute_error_{baseline}"] / result["actual_daily_demand_gwh"] * 100
        )

    table_path = tables_dir / "bonus_baseline_forecast_summary.csv"
    result.to_csv(table_path, index=False, encoding="utf-8-sig")

    metric_rows = []
    metric_labels = {
        "recent_30day_mean": "최근 30일 평균",
        "previous_year_same_day": "전년 같은 날짜",
    }
    for baseline, label in metric_labels.items():
        forecast_column = f"forecast_{baseline}"
        valid_rows = result.dropna(subset=[forecast_column])
        metrics = evaluate_forecast(
            valid_rows["actual_daily_demand_gwh"],
            valid_rows[forecast_column],
        )
        metric_rows.append(
            {
                "baseline": label,
                "mae": metrics["MAE"],
                "mape": metrics["MAPE"],
                "rmse": metrics["RMSE"],
                "evaluated_day_count": len(valid_rows),
            }
        )
    metrics_df = pd.DataFrame(metric_rows)

    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(
        result["date"],
        result["actual_daily_demand_gwh"],
        marker="o",
        linewidth=2,
        color="#2F6690",
        label="실제값",
    )
    axis.plot(
        result["date"],
        result["forecast_recent_30day_mean"],
        linestyle="--",
        linewidth=2,
        color="#C9A66B",
        label="최근 30일 평균 baseline",
    )
    axis.plot(
        result["date"],
        result["forecast_previous_year_same_day"],
        linestyle="--",
        linewidth=2,
        color="#5B8E7D",
        label="전년 같은 날짜 baseline",
    )
    axis.set_title("마지막 30일 전력수요량 baseline 예측 비교")
    axis.set_xlabel("날짜")
    axis.set_ylabel("일별 총수요량 (GWh)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure_path = figures_dir / "bonus_baseline_forecast.png"
    save_and_show_figure(figure, figure_path, show_graphs)

    best_row = metrics_df.sort_values("mae").iloc[0]
    print("\n보너스 2: Baseline 예측")
    print("=" * 72)
    print(
        f"테스트 기간: {result['date'].min().date()} ~ {result['date'].max().date()} "
        f"({len(result):,}일)"
    )
    print(f"최근 30일 평균 기준 구간: {recent_window['date'].min().date()} ~ {recent_window['date'].max().date()}")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(
        f"테스트 기간에서는 '{best_row['baseline']}' baseline의 MAE가 더 낮았습니다. "
        "단, 30일 테스트 결과이므로 일반화에는 한계가 있습니다."
    )
    print(
        "※ 이 baseline은 기온, 요일, 공휴일, 산업활동 변수를 사용하지 않으므로 "
        "실제 예측 모델로 해석하기에는 제한이 있습니다."
    )

    return table_path, figure_path, result, metrics_df


def verify_outputs(
    output_paths: list[Path],
    report_path: Path,
    readme_path: Path,
) -> None:
    """생성 파일과 문서 수정 여부를 마지막에 확인한다."""
    print("\n보너스 분석 검증")
    print("=" * 72)
    for path in output_paths:
        status = "생성됨" if path.exists() and path.stat().st_size > 0 else "확인 필요"
        print(f"- {status}: {path.resolve()}")
    print(f"- REPORT.md 존재: {report_path.exists()}")
    print(f"- README.md 존재: {readme_path.exists()}")
    print("- 기존 필수 분석 결과 파일명은 bonus_ 접두사를 사용하지 않아 덮어쓰지 않습니다.")


def main() -> None:
    args = parse_arguments()
    tables_dir = args.results_dir / "tables"
    figures_dir = args.results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    configure_korean_font()
    daily, monthly = load_data(args.data_dir)
    show_graphs = not args.no_show

    decomposition_table, decomposition_figure, _ = run_time_series_decomposition(
        monthly, tables_dir, figures_dir, show_graphs
    )
    forecast_table, forecast_figure, _, metrics_df = run_baseline_forecast(
        daily, tables_dir, figures_dir, show_graphs
    )

    verify_outputs(
        [decomposition_table, decomposition_figure, forecast_table, forecast_figure],
        Path("REPORT.md"),
        Path("README.md"),
    )

    print("\n최종 출력 요약")
    print("=" * 72)
    print("생성된 그래프 파일:")
    print(f"- {decomposition_figure.resolve()}")
    print(f"- {forecast_figure.resolve()}")
    print("생성된 표 파일:")
    print(f"- {decomposition_table.resolve()}")
    print(f"- {forecast_table.resolve()}")
    print("예측 성능 요약:")
    print(metrics_df.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print("보너스 분석의 핵심 한계:")
    print("- 시계열 분해는 월별 평균 일일수요량만 사용하며 외부 원인 변수를 포함하지 않습니다.")
    print("- baseline 예측은 기온, 요일, 공휴일, 산업활동 정보를 사용하지 않습니다.")
    print("- 테스트 기간이 마지막 30일뿐이므로 성능을 일반화하기 어렵습니다.")


if __name__ == "__main__":
    main()
