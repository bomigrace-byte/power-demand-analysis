"""전처리된 전국 전력수요량을 이용한 Mission 7 2차 분석.

Q2 장기 추세, Q3 계절성, Q5 전일 대비 급격한 변화 후보,
Q8 코로나 전·영향 기간·이후 구간 비교를 계산하고 표와 그래프로 저장한다.
이 코드는 관찰 가능한 수치를 제공하며 원인이나 최종 인사이트를 결론 내리지 않는다.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = {
    "daily_demand.csv": ["date", "year", "month", "daily_demand_gwh", "source_file"],
    "monthly_demand.csv": [
        "month_start",
        "year",
        "month",
        "monthly_demand_gwh",
        "day_count",
    ],
    "yearly_demand.csv": ["year", "yearly_demand_gwh", "day_count"],
}


def parse_arguments() -> argparse.Namespace:
    """입력 폴더, 결과 폴더, 그래프 표시 방식을 실행 시 지정한다."""
    parser = argparse.ArgumentParser(
        description="전처리된 전력수요량으로 Q2·Q3·Q5·Q8 분석표와 그래프를 만듭니다."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="전처리 CSV 3개가 있는 폴더 (기본값: data/processed)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="분석 결과 저장 폴더 (기본값: results)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="화면에 그래프를 띄우지 않음. 자동 실행이나 서버 환경에서 사용",
    )
    return parser.parse_args()


def configure_korean_font() -> str:
    """운영체제에 설치된 한글 폰트를 찾아 그래프의 한글 깨짐을 줄인다."""
    preferred_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"그래프 한글 폰트: {font_name}")
            return font_name

    warnings.warn(
        "권장 한글 폰트를 찾지 못했습니다. 한글이 깨지면 Malgun Gothic 또는 "
        "Noto Sans CJK KR을 설치하세요.",
        stacklevel=2,
    )
    plt.rcParams["axes.unicode_minus"] = False
    return "기본 sans-serif"


def load_processed_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """전처리 결과 3개만 읽고 실제 구조와 기간을 확인한다."""
    frames: dict[str, pd.DataFrame] = {}

    for file_name, expected_columns in EXPECTED_COLUMNS.items():
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"전처리 결과 파일이 없습니다: {path.resolve()}")

        date_columns = [column for column in ("date", "month_start") if column in expected_columns]
        frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=date_columns)
        if list(frame.columns) != expected_columns:
            raise ValueError(
                f"{file_name} 컬럼이 예상 구조와 다릅니다.\n"
                f"예상: {expected_columns}\n실제: {list(frame.columns)}"
            )
        frames[file_name] = frame

    daily = frames["daily_demand.csv"].sort_values("date").reset_index(drop=True)
    monthly = frames["monthly_demand.csv"].sort_values("month_start").reset_index(drop=True)
    yearly = frames["yearly_demand.csv"].sort_values("year").reset_index(drop=True)

    # 변화율은 반드시 연속된 전날과 비교해야 하므로 날짜 누락이나 중복을 먼저 막는다.
    if daily["date"].duplicated().any():
        raise ValueError("daily_demand.csv에 중복 날짜가 있어 전일 대비 계산을 중단합니다.")
    expected_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    missing_dates = expected_dates.difference(daily["date"])
    if len(missing_dates):
        raise ValueError(
            "daily_demand.csv에 결측 날짜가 있어 전일 대비 계산을 중단합니다: "
            + ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates[:20])
        )

    print("\n전처리 데이터 확인")
    print("=" * 72)
    print(
        f"daily: {len(daily):,}행 | {daily['date'].min().date()} ~ "
        f"{daily['date'].max().date()} | 컬럼: {list(daily.columns)}"
    )
    print(
        f"monthly: {len(monthly):,}행 | {monthly['month_start'].min().date()} ~ "
        f"{monthly['month_start'].max().date()} | 컬럼: {list(monthly.columns)}"
    )
    print(
        f"yearly: {len(yearly):,}행 | {yearly['year'].min()} ~ "
        f"{yearly['year'].max()} | 컬럼: {list(yearly.columns)}"
    )
    return daily, monthly, yearly


def save_and_show_figure(figure: plt.Figure, path: Path, show_graphs: bool) -> None:
    """보고서용 PNG를 저장하고, 기본 실행에서는 화면에도 표시한다."""
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    if show_graphs:
        plt.show()
    plt.close(figure)


def analyze_yearly_trend(
    yearly: pd.DataFrame, tables_dir: Path, figures_dir: Path, show_graphs: bool
) -> Path:
    """Q2: 연간 총수요량과 전년 대비 변화량·변화율을 계산한다."""
    summary = yearly.copy()
    summary["year_over_year_change_gwh"] = summary["yearly_demand_gwh"].diff()
    summary["year_over_year_change_pct"] = summary["yearly_demand_gwh"].pct_change() * 100

    table_path = tables_dir / "q2_yearly_trend_summary.csv"
    summary.to_csv(table_path, index=False, encoding="utf-8-sig")

    print("\nQ2 연도별 장기 추세 결과표")
    print("=" * 72)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("※ day_count가 365일인지 366일인지 함께 확인할 수 있습니다.")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.plot(
        summary["year"],
        summary["yearly_demand_gwh"],
        marker="o",
        linewidth=2,
        color="#2F6690",
    )
    axis.set_title("2015~2025년 연간 총 전력수요량")
    axis.set_xlabel("연도")
    axis.set_ylabel("연간 총수요량 (GWh)")
    axis.set_xticks(summary["year"])
    axis.grid(axis="y", alpha=0.3)
    figure_path = figures_dir / "q2_yearly_demand_trend.png"
    save_and_show_figure(figure, figure_path, show_graphs)
    return table_path


def analyze_seasonality(
    monthly: pd.DataFrame, tables_dir: Path, figures_dir: Path, show_graphs: bool
) -> tuple[Path, Path]:
    """Q3: 달력 월별·계절별 수요 수준을 총수요와 일평균 기준으로 비교한다."""
    data = monthly.copy()

    # 월 길이가 달라 총수요만으로 비교하면 차이가 섞이므로 월별 일평균도 함께 계산한다.
    data["average_daily_demand_gwh"] = data["monthly_demand_gwh"] / data["day_count"]
    season_map = {
        1: "겨울",
        2: "겨울",
        3: "봄",
        4: "봄",
        5: "봄",
        6: "여름",
        7: "여름",
        8: "여름",
        9: "가을",
        10: "가을",
        11: "가을",
        12: "겨울",
    }
    data["season"] = data["month"].map(season_map)

    monthly_summary = (
        data.groupby("month", as_index=False)
        .agg(
            observed_years=("year", "nunique"),
            mean_monthly_total_demand_gwh=("monthly_demand_gwh", "mean"),
            mean_daily_demand_gwh=("average_daily_demand_gwh", "mean"),
        )
        .sort_values("month")
    )

    # 계절별 일평균은 해당 계절의 총수요를 실제 포함 일수로 나눈 가중 평균이다.
    seasonal_summary = (
        data.groupby("season", as_index=False)
        .agg(
            month_observation_count=("month", "size"),
            total_day_count=("day_count", "sum"),
            mean_monthly_total_demand_gwh=("monthly_demand_gwh", "mean"),
            total_demand_gwh=("monthly_demand_gwh", "sum"),
        )
    )
    seasonal_summary["weighted_mean_daily_demand_gwh"] = (
        seasonal_summary["total_demand_gwh"] / seasonal_summary["total_day_count"]
    )
    season_order = pd.CategoricalDtype(["봄", "여름", "가을", "겨울"], ordered=True)
    seasonal_summary["season"] = seasonal_summary["season"].astype(season_order)
    seasonal_summary = seasonal_summary.sort_values("season").reset_index(drop=True)

    monthly_path = tables_dir / "q3_monthly_seasonality_summary.csv"
    seasonal_path = tables_dir / "q3_seasonal_summary.csv"
    monthly_summary.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    seasonal_summary.to_csv(seasonal_path, index=False, encoding="utf-8-sig")

    print("\nQ3 월별 계절성 결과표")
    print("=" * 72)
    print(monthly_summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\nQ3 계절별 결과표")
    print(seasonal_summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("※ 월 총수요 평균과 월별 일평균 수요는 의미가 다르므로 별도 컬럼으로 표시했습니다.")

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].bar(
        monthly_summary["month"],
        monthly_summary["mean_monthly_total_demand_gwh"],
        color="#5B8E7D",
    )
    axes[0].set_title("2015~2025년 달력 월별 평균 총 전력수요량")
    axes[0].set_ylabel("월 평균 총수요량 (GWh)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        monthly_summary["month"],
        monthly_summary["mean_daily_demand_gwh"],
        color="#A7C4A0",
    )
    axes[1].set_title("2015~2025년 달력 월별 평균 일일 전력수요량")
    axes[1].set_xlabel("월")
    axes[1].set_ylabel("평균 일일수요량 (GWh/일)")
    axes[1].set_xticks(range(1, 13))
    axes[1].grid(axis="y", alpha=0.25)
    save_and_show_figure(
        figure, figures_dir / "q3_monthly_seasonality.png", show_graphs
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(
        seasonal_summary["season"].astype(str),
        seasonal_summary["mean_monthly_total_demand_gwh"],
        color="#5B8E7D",
    )
    axes[0].set_title("계절별 월 평균 총수요량")
    axes[0].set_xlabel("계절")
    axes[0].set_ylabel("월 평균 총수요량 (GWh)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        seasonal_summary["season"].astype(str),
        seasonal_summary["weighted_mean_daily_demand_gwh"],
        color="#A7C4A0",
    )
    axes[1].set_title("계절별 가중 평균 일일수요량")
    axes[1].set_xlabel("계절")
    axes[1].set_ylabel("평균 일일수요량 (GWh/일)")
    axes[1].grid(axis="y", alpha=0.25)
    save_and_show_figure(figure, figures_dir / "q3_seasonal_demand.png", show_graphs)
    return monthly_path, seasonal_path


def analyze_daily_change(
    daily: pd.DataFrame, tables_dir: Path, figures_dir: Path, show_graphs: bool
) -> tuple[Path, Path, Path]:
    """Q5: 전일 대비 변화율 양쪽 꼬리와 상·하위 10개 후보 날짜를 제시한다."""
    data = daily.sort_values("date").copy()
    data["previous_date"] = data["date"].shift(1)
    data["previous_daily_demand_gwh"] = data["daily_demand_gwh"].shift(1)
    data["change_gwh"] = data["daily_demand_gwh"].diff()
    data["change_pct"] = data["daily_demand_gwh"].pct_change(fill_method=None) * 100
    valid = data.dropna(subset=["change_pct"]).copy()

    output_columns = [
        "date",
        "previous_date",
        "previous_daily_demand_gwh",
        "daily_demand_gwh",
        "change_gwh",
        "change_pct",
    ]
    top_increase = valid.nlargest(10, "change_pct")[output_columns]
    top_decrease = valid.nsmallest(10, "change_pct")[output_columns]

    # 임의 임계값 대신 관측 변화율 분포의 하위 1%와 상위 1% 경계를 사용한다.
    lower_threshold = valid["change_pct"].quantile(0.01)
    upper_threshold = valid["change_pct"].quantile(0.99)
    candidates = valid.loc[
        (valid["change_pct"] <= lower_threshold)
        | (valid["change_pct"] >= upper_threshold),
        output_columns,
    ].copy()
    candidates["candidate_type"] = np.where(
        candidates["change_pct"] >= upper_threshold, "상위 1% 증가 후보", "하위 1% 감소 후보"
    )
    candidates["applied_threshold_pct"] = np.where(
        candidates["change_pct"] >= upper_threshold, upper_threshold, lower_threshold
    )
    candidates = candidates.sort_values("date")

    increase_path = tables_dir / "q5_top_increase_days.csv"
    decrease_path = tables_dir / "q5_top_decrease_days.csv"
    candidates_path = tables_dir / "q5_change_rate_candidates.csv"
    top_increase.to_csv(increase_path, index=False, encoding="utf-8-sig")
    top_decrease.to_csv(decrease_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")

    percentiles = valid["change_pct"].quantile([0.01, 0.05, 0.95, 0.99])
    distribution = pd.Series(
        {
            "평균": valid["change_pct"].mean(),
            "표준편차": valid["change_pct"].std(),
            "최소값": valid["change_pct"].min(),
            "최대값": valid["change_pct"].max(),
            "1% 분위수": percentiles.loc[0.01],
            "5% 분위수": percentiles.loc[0.05],
            "95% 분위수": percentiles.loc[0.95],
            "99% 분위수": percentiles.loc[0.99],
        },
        name="전일 대비 변화율(%)",
    )

    print("\nQ5 전일 대비 변화율 분포 기초 통계")
    print("=" * 72)
    print(distribution.to_string(float_format=lambda value: f"{value:,.3f}"))
    print("\n변화율 증가 상위 10개 날짜")
    print(top_increase.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print("\n변화율 감소 상위 10개 날짜")
    print(top_decrease.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(
        f"\n상위/하위 1% 후보: {len(candidates):,}일 | "
        f"하한 {lower_threshold:.3f}% | 상한 {upper_threshold:.3f}%"
    )
    print("※ 후보 날짜는 데이터 오류나 특정 외부 요인의 결과라고 단정하지 않습니다.")

    figure, axis = plt.subplots(figsize=(13, 5.5))
    axis.plot(valid["date"], valid["change_pct"], linewidth=0.7, color="#5B6770")
    axis.axhline(upper_threshold, color="#C44E52", linestyle="--", linewidth=1.2, label="99% 분위수")
    axis.axhline(lower_threshold, color="#4C72B0", linestyle="--", linewidth=1.2, label="1% 분위수")
    axis.scatter(
        candidates["date"],
        candidates["change_pct"],
        s=13,
        color="#D55E00",
        alpha=0.75,
        label="상·하위 1% 후보",
    )
    axis.set_title("2015~2025년 일별 전력수요량 전일 대비 변화율")
    axis.set_xlabel("날짜")
    axis.set_ylabel("전일 대비 변화율 (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save_and_show_figure(figure, figures_dir / "q5_daily_change_rate.png", show_graphs)
    return increase_path, decrease_path, candidates_path


def analyze_covid_periods(
    daily: pd.DataFrame, tables_dir: Path, figures_dir: Path, show_graphs: bool
) -> Path:
    """Q8: 임시로 정의한 코로나 전·영향 기간·이후 구간의 기술통계를 비교한다."""
    periods = [
        ("코로나 이전", pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31")),
        ("코로나 영향 기간", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
        ("코로나 이후", pd.Timestamp("2022-01-01"), pd.Timestamp("2025-12-31")),
    ]
    rows = []

    for period_name, start_date, end_date in periods:
        values = daily.loc[
            daily["date"].between(start_date, end_date), "daily_demand_gwh"
        ]
        rows.append(
            {
                "period": period_name,
                "start_date": start_date,
                "end_date": end_date,
                "day_count": values.size,
                "total_demand_gwh": values.sum(),
                "mean_daily_demand_gwh": values.mean(),
                "median_daily_demand_gwh": values.median(),
                "min_daily_demand_gwh": values.min(),
                "max_daily_demand_gwh": values.max(),
                "std_daily_demand_gwh": values.std(),
            }
        )

    summary = pd.DataFrame(rows)
    table_path = tables_dir / "q8_covid_period_summary.csv"
    summary.to_csv(table_path, index=False, encoding="utf-8-sig")

    print("\nQ8 코로나 전·영향 기간·이후 구간 결과표")
    print("=" * 72)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("※ 구간별 총수요량은 구간 길이가 달라 직접 비교할 때 주의해야 합니다.")
    print("※ 이 표는 구간 간 차이를 보여주며 코로나가 차이의 원인이라고 결론 내리지 않습니다.")

    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(
        summary["period"],
        summary["mean_daily_demand_gwh"],
        color=["#6C8EAD", "#C9A66B", "#6F9E75"],
    )
    axis.bar_label(bars, fmt="%.1f", padding=3)
    axis.set_title("코로나 전·영향 기간·이후 평균 일일 전력수요량")
    axis.set_xlabel("분석 구간")
    axis.set_ylabel("평균 일일수요량 (GWh/일)")
    axis.grid(axis="y", alpha=0.25)
    save_and_show_figure(
        figure, figures_dir / "q8_covid_period_comparison.png", show_graphs
    )
    return table_path


def print_generated_files(results_dir: Path) -> None:
    """실행 완료 후 생성된 표와 그림을 한눈에 확인한다."""
    print("\n생성된 파일")
    print("=" * 72)
    for path in sorted(results_dir.rglob("*")):
        if path.is_file():
            print(f"- {path.resolve()}")


def main() -> None:
    args = parse_arguments()
    tables_dir = args.results_dir / "tables"
    figures_dir = args.results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    configure_korean_font()
    daily, monthly, yearly = load_processed_data(args.data_dir)
    show_graphs = not args.no_show

    analyze_yearly_trend(yearly, tables_dir, figures_dir, show_graphs)
    analyze_seasonality(monthly, tables_dir, figures_dir, show_graphs)
    analyze_daily_change(daily, tables_dir, figures_dir, show_graphs)
    analyze_covid_periods(daily, tables_dir, figures_dir, show_graphs)
    print_generated_files(args.results_dir)


if __name__ == "__main__":
    main()
