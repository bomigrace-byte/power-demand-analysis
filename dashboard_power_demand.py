"""대한민국 전체 전력수요량 분석 결과를 보여주는 Streamlit 대시보드.

이 파일은 이미 생성된 전처리·분석 결과 CSV와 PNG를 읽어서 보여준다.
원본 raw CSV를 다시 읽지 않으며, 기존 분석 결과를 삭제하거나 덮어쓰지 않는다.
"""

from __future__ import annotations
import html
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# 로컬 실습 환경에서 Streamlit이 사용자 홈 폴더에 통계 설정을 쓰지 못해도
# 앱 실행이 흔들리지 않도록 사용 통계 수집을 끈다.
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import streamlit as st


DATA_DIR = Path("data/processed")
FIGURES_DIR = Path("results/figures")
TABLES_DIR = Path("results/tables")

MENU_OPTIONS = [
    "전체 개요",
    "일별 추세 탐색",
    "월별·계절성 분석",
    "급격한 변화 후보",
    "코로나 구간 비교",
    "보너스: 시계열 분해",
    "보너스: Baseline 예측",
]


def apply_custom_style() -> None:
    """제출용 시연 화면이 차분하게 보이도록 sage green 계열 CSS를 적용한다."""
    st.markdown(
        """
        <style>
            :root {
                --page-bg: #F7F8F3;
                --card-bg: #F1F4EC;
                --card-border: #D6DECB;
                --primary-sage: #6F826A;
                --text-main: #1F2A24;
                --text-muted: #6B7568;
            }

            .stApp {
                background-color: var(--page-bg);
                color: var(--text-main);
            }

            [data-testid="stAppViewContainer"] {
                background-color: var(--page-bg);
            }

            header[data-testid="stHeader"] {
                background-color: var(--page-bg);
                height: 3rem;
            }

            [data-testid="stToolbar"] {
                background-color: transparent;
            }

            div[data-testid="stDecoration"] {
                background-image: none;
                height: 0rem;
            }

            .block-container {
                padding-top: 2.5rem;
                padding-bottom: 2rem;
                padding-left: 2rem;
                padding-right: 2rem;
                max-width: 1180px;
            }

            section[data-testid="stSidebar"] {
                background-color: #EEF2E8;
                border-right: 1px solid var(--card-border);
            }

            h1 {
                color: var(--text-main);
                font-size: 2rem !important;
                line-height: 1.25 !important;
                margin-top: 0.5rem !important;
                margin-bottom: 1rem !important;
            }

            h2 {
                color: var(--primary-sage);
                font-size: 1.35rem !important;
                line-height: 1.35 !important;
                margin-top: 1.25rem !important;
                margin-bottom: 0.45rem !important;
            }

            h3 {
                color: var(--text-main);
                font-size: 1.08rem !important;
                margin-top: 1rem !important;
            }

            p, li, div, span {
                color: var(--text-main);
            }

            [data-testid="stCaptionContainer"] p,
            .stCaptionContainer {
                color: var(--text-muted) !important;
            }

            .metric-card {
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-left: 5px solid var(--primary-sage);
                border-radius: 14px;
                padding: 0.85rem 0.95rem;
                min-height: 104px;
                box-shadow: 0 1px 5px rgba(31, 42, 36, 0.04);
                margin-bottom: 0.75rem;
            }

            .metric-label {
                color: var(--text-muted);
                font-size: 0.86rem;
                font-weight: 650;
                letter-spacing: -0.01em;
                margin-bottom: 0.45rem;
            }

            .metric-value {
                color: var(--text-main);
                font-size: clamp(1.02rem, 1.45vw, 1.34rem);
                font-weight: 760;
                line-height: 1.25;
                letter-spacing: -0.025em;
                white-space: nowrap;
            }

            .metric-note {
                color: var(--text-muted);
                font-size: 0.78rem;
                margin-top: 0.45rem;
                white-space: nowrap;
            }

            div[data-testid="stDataFrame"],
            div[data-testid="stImage"] {
                background: rgba(255, 255, 255, 0.38);
                border-radius: 12px;
            }

            .stAlert {
                border-radius: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, note=None) -> None:
    """st.metric 대신 글자 크기가 균일한 커스텀 지표 카드를 표시한다."""
    label_text = html.escape(str(label))
    value_text = html.escape(str(value))
    note_html = ""
    if note is not None:
        note_html = f'<div class="metric-note">{html.escape(str(note))}</div>'

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label_text}</div>
            <div class="metric-value">{value_text}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_csv(path_text: str) -> pd.DataFrame | None:
    """CSV 파일을 읽고 날짜 컬럼은 날짜형으로 바꾼다.

    대시보드는 결과를 보여주는 역할이므로 파일이 없을 때 앱 전체를 멈추지 않고
    해당 화면에서 안내 메시지를 보여주기 위해 None을 반환한다.
    """
    path = Path(path_text)
    if not path.exists():
        return None

    frame = pd.read_csv(path, encoding="utf-8-sig")
    for column in ["date", "month_start", "start_date", "end_date", "previous_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def show_missing_file(path: Path, purpose: str) -> None:
    """필요한 파일이 없을 때 사용자가 어떤 코드를 먼저 실행해야 하는지 안내한다."""
    st.warning(
        f"{purpose} 파일이 없습니다: `{path}`\n\n"
        "먼저 전처리/분석 코드를 실행해 결과 파일을 생성해 주세요."
    )


def show_image(path: Path, caption: str) -> None:
    """PNG 결과 파일을 화면에 보여준다."""
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        show_missing_file(path, caption)


def format_number(value: float | int | None, digits: int = 2) -> str:
    """Streamlit metric에 보기 좋은 숫자 문자열을 표시한다."""
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}"


def get_daily_date_bounds(daily: pd.DataFrame | None) -> tuple[date, date]:
    """사이드바 기간 선택의 기본 시작일과 종료일을 만든다."""
    if daily is None or daily.empty or "date" not in daily.columns:
        today = pd.Timestamp.today().date()
        return today, today
    return daily["date"].min().date(), daily["date"].max().date()


def render_sidebar(daily: pd.DataFrame | None) -> tuple[str, date, date, str, int]:
    """사이드바에서 메뉴, 기간, 급격한 변화 후보 조건을 선택한다."""
    st.sidebar.header("분석 메뉴와 필터")
    menu = st.sidebar.selectbox("분석 메뉴 선택", MENU_OPTIONS)

    min_date, max_date = get_daily_date_bounds(daily)
    st.sidebar.subheader("기간 선택")
    start_date = st.sidebar.date_input("시작일", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("종료일", value=max_date, min_value=min_date, max_value=max_date)

    change_view = "증가 상위"
    display_count = 10
    if menu == "급격한 변화 후보":
        st.sidebar.subheader("변화율 후보 조건")
        change_view = st.sidebar.radio(
            "보기 방식",
            ["증가 상위", "감소 하위", "상위/하위 1% 후보 전체"],
        )
        display_count = st.sidebar.selectbox("표시 개수", [5, 10, 20], index=1)

    return menu, start_date, end_date, change_view, display_count


def show_overview(daily: pd.DataFrame | None, monthly: pd.DataFrame | None, yearly: pd.DataFrame | None) -> None:
    """전체 데이터 범위와 핵심 요약 지표를 보여준다."""
    st.header("전체 개요")
    st.write("연도별 총수요량의 장기 흐름을 확인하는 화면입니다.")

    if daily is None:
        show_missing_file(DATA_DIR / "daily_demand.csv", "일별 데이터")
        return

    max_row = daily.loc[daily["daily_demand_gwh"].idxmax()]
    min_row = daily.loc[daily["daily_demand_gwh"].idxmin()]
    monthly_count = 0 if monthly is None else len(monthly)
    yearly_count = 0 if yearly is None else len(yearly)

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        render_metric_card("전체 데이터 기간", f"{daily['date'].min().date()} ~ {daily['date'].max().date()}")
    with col2:
        render_metric_card("일별 데이터 수", f"{len(daily):,}일")
    with col3:
        render_metric_card("월별 / 연별 데이터 수", f"{monthly_count:,}개월 / {yearly_count:,}년")

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        render_metric_card("전체 평균 일일수요량", f"{format_number(daily['daily_demand_gwh'].mean())} GWh")
    with col2:
        render_metric_card("최댓값", f"{format_number(max_row['daily_demand_gwh'])} GWh", note=str(max_row["date"].date()))
    with col3:
        render_metric_card("최솟값", f"{format_number(min_row['daily_demand_gwh'])} GWh", note=str(min_row["date"].date()))

    show_image(FIGURES_DIR / "q2_yearly_demand_trend.png", "연도별 총 전력수요량 추세")


def show_daily_trend(daily: pd.DataFrame | None, start_date: date, end_date: date) -> None:
    """선택한 기간의 일별 전력수요량을 탐색한다."""
    st.header("일별 추세 탐색")
    st.write("기간 필터를 바꾸면 해당 구간의 일별 수요 변화를 확인할 수 있습니다.")

    if daily is None:
        show_missing_file(DATA_DIR / "daily_demand.csv", "일별 데이터")
        return
    if start_date > end_date:
        st.warning("시작일이 종료일보다 늦습니다. 사이드바에서 기간을 다시 선택해 주세요.")
        return

    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)
    filtered = daily[daily["date"].between(start_timestamp, end_timestamp)].copy()

    if filtered.empty:
        st.warning("선택한 기간에 해당하는 데이터가 없습니다.")
        return

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        render_metric_card("선택 기간 평균", f"{format_number(filtered['daily_demand_gwh'].mean())} GWh")
    with col2:
        render_metric_card("선택 기간 최댓값", f"{format_number(filtered['daily_demand_gwh'].max())} GWh")
    with col3:
        render_metric_card("선택 기간 최솟값", f"{format_number(filtered['daily_demand_gwh'].min())} GWh")

    # 날짜 컬럼을 명시적으로 x축으로 지정하기 위해 date 컬럼을 그대로 포함한 표를 만든다.
    # set_index 방식보다 Streamlit 차트에서 x축이 날짜임을 더 분명하게 전달할 수 있다.
    chart_data = filtered[["date", "daily_demand_gwh"]].copy()
    st.line_chart(
        chart_data,
        x="date",
        y="daily_demand_gwh",
        width="stretch",
    )
    st.caption("단위: GWh. x축은 선택한 기간의 날짜, y축은 일별 총수요량입니다.")

    st.subheader("선택 기간 데이터 일부")
    st.dataframe(filtered.head(50), width="stretch")


def show_seasonality() -> None:
    """월별·계절별 패턴 그래프와 요약표를 보여준다."""
    st.header("월별·계절성 분석")
    st.write("월별·계절별 반복 패턴을 확인하는 화면입니다. 원인 해석에는 기온 등 외부 데이터가 추가로 필요합니다.")

    show_image(FIGURES_DIR / "q3_monthly_seasonality.png", "월별 평균 전력수요량")
    show_image(FIGURES_DIR / "q3_seasonal_demand.png", "계절별 평균 전력수요량")

    monthly_summary = load_csv(str(TABLES_DIR / "q3_monthly_seasonality_summary.csv"))
    seasonal_summary = load_csv(str(TABLES_DIR / "q3_seasonal_summary.csv"))

    st.subheader("월별 요약표")
    if monthly_summary is None:
        show_missing_file(TABLES_DIR / "q3_monthly_seasonality_summary.csv", "월별 계절성 요약표")
    else:
        st.dataframe(monthly_summary, width="stretch")

    st.subheader("계절별 요약표")
    if seasonal_summary is None:
        show_missing_file(TABLES_DIR / "q3_seasonal_summary.csv", "계절별 요약표")
    else:
        st.dataframe(seasonal_summary, width="stretch")


def show_change_candidates(change_view: str, display_count: int) -> None:
    """전일 대비 변화율이 큰 후보 날짜를 선택 조건에 맞게 보여준다."""
    st.header("급격한 변화 후보")
    st.write("전일 대비 변화율이 큰 날짜 후보를 확인하는 화면입니다. 명절·연휴 등 원인 해석은 추가 데이터가 필요합니다.")

    show_image(FIGURES_DIR / "q5_daily_change_rate.png", "전일 대비 변화율 시계열")

    table_map = {
        "증가 상위": TABLES_DIR / "q5_top_increase_days.csv",
        "감소 하위": TABLES_DIR / "q5_top_decrease_days.csv",
        "상위/하위 1% 후보 전체": TABLES_DIR / "q5_change_rate_candidates.csv",
    }
    selected_path = table_map[change_view]
    selected_table = load_csv(str(selected_path))

    st.subheader(f"{change_view} 결과")
    if selected_table is None:
        show_missing_file(selected_path, "급격한 변화 후보 표")
    else:
        st.dataframe(selected_table.head(display_count), width="stretch")

    st.info("이 후보는 이상치 또는 원인을 확정한 것이 아니라 추가 검토가 필요한 후보입니다.")


def show_covid_comparison() -> None:
    """코로나 전·영향 기간·이후 구간의 기술통계를 보여준다."""
    st.header("코로나 구간 비교")
    st.write("코로나 이전·영향 기간·이후 구간의 평균 일일수요량을 비교하는 화면입니다. 인과관계가 아니라 구간별 기술통계입니다.")

    show_image(FIGURES_DIR / "q8_covid_period_comparison.png", "코로나 구간별 평균 일일 전력수요량")

    summary = load_csv(str(TABLES_DIR / "q8_covid_period_summary.csv"))
    if summary is None:
        show_missing_file(TABLES_DIR / "q8_covid_period_summary.csv", "코로나 구간 비교표")
    else:
        st.dataframe(summary, width="stretch")
    st.info("이 비교는 인과관계 분석이 아니라 구간별 기술통계입니다.")


def show_decomposition() -> None:
    """보너스 시계열 분해 결과를 보여준다."""
    st.header("보너스: 시계열 분해")
    st.write("시계열 분해는 전력수요를 추세, 계절성, 잔차로 나누어 보는 학습용 분석입니다.")

    show_image(FIGURES_DIR / "bonus_time_series_decomposition.png", "시계열 분해 결과")

    st.subheader("분해 성분의 의미")
    st.markdown(
        """
        - **Observed**: 실제 월별 평균 일일수요량
        - **Trend**: 장기 흐름
        - **Seasonal**: 반복되는 월별 패턴
        - **Residual**: 추세와 계절성으로 설명되지 않는 변동
        """
    )

    summary = load_csv(str(TABLES_DIR / "bonus_decomposition_summary.csv"))
    if summary is None:
        show_missing_file(TABLES_DIR / "bonus_decomposition_summary.csv", "시계열 분해 결과표")
    else:
        st.subheader("시계열 분해 결과표 일부")
        st.dataframe(summary.head(30), width="stretch")

    st.info("계절성의 원인을 냉난방으로 단정하지 않습니다. 기온, 공휴일, 산업활동 등 외부 데이터가 없는 한계가 있습니다.")


def calculate_forecast_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    """baseline별 MAE, MAPE, RMSE를 대시보드에서 다시 계산한다."""
    baselines = {
        "최근 30일 평균 baseline": "forecast_recent_30day_mean",
        "전년 같은 날짜 baseline": "forecast_previous_year_same_day",
    }
    rows = []
    for baseline_name, forecast_column in baselines.items():
        if forecast_column not in summary.columns:
            continue
        valid = summary.dropna(subset=[forecast_column]).copy()
        error = valid["actual_daily_demand_gwh"] - valid[forecast_column]
        absolute_error = error.abs()
        absolute_percentage_error = absolute_error / valid["actual_daily_demand_gwh"] * 100
        rows.append(
            {
                "baseline": baseline_name,
                "MAE": absolute_error.mean(),
                "MAPE": absolute_percentage_error.mean(),
                "RMSE": np.sqrt((error**2).mean()),
                "평가 일수": len(valid),
            }
        )
    return pd.DataFrame(rows)


def show_forecast() -> None:
    """보너스 baseline 예측 결과와 평가 지표를 보여준다."""
    st.header("보너스: Baseline 예측")
    st.write("이 예측은 실제 미래 예측 모델이 아니라, 단순 기준선과 실제값을 비교해 예측의 가정과 한계를 이해하기 위한 분석입니다.")

    show_image(FIGURES_DIR / "bonus_baseline_forecast.png", "Baseline 예측 비교")

    st.subheader("Baseline 설명")
    st.markdown(
        """
        - **최근 30일 평균 baseline**: 테스트 직전 30일 평균을 테스트 기간 전체 예측값으로 사용
        - **전년 같은 날짜 baseline**: 테스트 날짜별 전년 같은 날짜 값을 예측값으로 사용
        - **테스트 구간**: 마지막 30일
        """
    )

    summary = load_csv(str(TABLES_DIR / "bonus_baseline_forecast_summary.csv"))
    if summary is None:
        show_missing_file(TABLES_DIR / "bonus_baseline_forecast_summary.csv", "Baseline 예측 결과표")
        return

    metrics = calculate_forecast_metrics(summary)
    if not metrics.empty:
        st.subheader("예측 성능 요약")
        st.dataframe(metrics, width="stretch")
        best_row = metrics.sort_values("MAE").iloc[0]
        st.info(
            f"마지막 30일 테스트 구간에서는 `{best_row['baseline']}`이 "
            "더 낮은 MAE를 보였습니다. 이 결과는 해당 테스트 기간에 한정됩니다."
        )

    st.subheader("Baseline 예측 결과표")
    st.dataframe(summary, width="stretch")
    st.info("기온, 요일, 공휴일, 산업활동 변수를 사용하지 않았기 때문에 일반화에는 한계가 있습니다.")


def main() -> None:
    """Streamlit 대시보드의 전체 흐름을 구성한다."""
    st.set_page_config(
        page_title="대한민국 전체 전력수요량 분석 대시보드",
        layout="wide",
    )
    apply_custom_style()

    daily = load_csv(str(DATA_DIR / "daily_demand.csv"))
    monthly = load_csv(str(DATA_DIR / "monthly_demand.csv"))
    yearly = load_csv(str(DATA_DIR / "yearly_demand.csv"))

    menu, start_date, end_date, change_view, display_count = render_sidebar(daily)

    st.title("대한민국 전체 전력수요량 분석 대시보드")
    st.write(
        "2015~2025년 한국전력거래소 시간별 전국 전력수요량을 일별·월별·연별로 "
        "집계해 탐색하는 대시보드입니다."
    )
    st.caption("분석 결과를 기간과 조건을 바꿔가며 탐색하기 위한 학습용 대시보드입니다.")

    if start_date > end_date:
        st.warning("사이드바에서 선택한 시작일이 종료일보다 늦습니다.")

    if menu == "전체 개요":
        show_overview(daily, monthly, yearly)
    elif menu == "일별 추세 탐색":
        show_daily_trend(daily, start_date, end_date)
    elif menu == "월별·계절성 분석":
        show_seasonality()
    elif menu == "급격한 변화 후보":
        show_change_candidates(change_view, display_count)
    elif menu == "코로나 구간 비교":
        show_covid_comparison()
    elif menu == "보너스: 시계열 분해":
        show_decomposition()
    elif menu == "보너스: Baseline 예측":
        show_forecast()


if __name__ == "__main__":
    main()
