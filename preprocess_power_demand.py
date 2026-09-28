"""한국전력거래소 시간별 전국 전력수요량 전처리 및 집계.

시각화와 인사이트 분석은 수행하지 않는다. 원본 CSV는 읽기만 하고,
정제된 일별/월별/연별 결과를 별도 CSV로 저장한다.
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


START_DATE = pd.Timestamp("2015-01-01")
END_DATE = pd.Timestamp("2025-12-31")
DATE_COLUMN = "날짜"
HOUR_COLUMNS = [f"{hour}시" for hour in range(1, 25)]
REQUIRED_COLUMNS = [DATE_COLUMN, *HOUR_COLUMNS]
FILE_PATTERN = "한국전력거래소_시간별 전국 전력수요량_*.csv"


def parse_arguments() -> argparse.Namespace:
    """실행할 때 입력/출력 폴더를 바꿀 수 있게 한다."""
    parser = argparse.ArgumentParser(
        description="2015~2025년 시간별 전국 전력수요량을 일/월/연 단위로 집계합니다."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="원본 CSV 파일들이 들어 있는 폴더",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="결과 CSV 저장 폴더 (기본값: ./output)",
    )
    return parser.parse_args()


def snapshot_date_from_name(path: Path) -> pd.Timestamp:
    """파일명의 YYYYMMDD를 자료 공개 시점으로 사용한다."""
    match = re.search(r"_(\d{8})\.csv$", path.name)
    if not match:
        raise ValueError(f"파일명에서 기준일(YYYYMMDD)을 찾을 수 없습니다: {path.name}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def print_rows(title: str, frame: pd.DataFrame) -> None:
    """문제가 있는 행을 삭제하기 전에 사람이 확인할 수 있도록 출력한다."""
    if frame.empty:
        return
    print(f"\n[{title}] {len(frame):,}행")
    print(frame.to_string(index=False))


def read_and_audit_file(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """한 파일의 구조를 검증하고, 빈/불완전 행을 먼저 공개한 뒤 정리한다."""
    # 실제 파일 확인 결과가 CP949이므로 인코딩을 명시해 한글 컬럼 깨짐을 막는다.
    raw = pd.read_csv(path, encoding="cp949")
    raw.columns = raw.columns.astype(str).str.strip()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    extra_columns = [column for column in raw.columns if column not in REQUIRED_COLUMNS]
    if missing_columns or extra_columns:
        raise ValueError(
            f"{path.name} 컬럼 구조가 예상과 다릅니다. "
            f"누락={missing_columns}, 추가={extra_columns}"
        )

    print(f"\n파일: {path.name}")
    print(f"  원본 행 수: {len(raw):,}, 컬럼 수: {len(raw.columns):,}")

    # 완전히 빈 행은 날짜나 수요 관측을 뜻하지 않으므로 먼저 출력한 뒤 제외한다.
    all_blank_mask = raw[REQUIRED_COLUMNS].isna().all(axis=1)
    all_blank_rows = raw.loc[all_blank_mask, REQUIRED_COLUMNS].copy()
    print_rows("완전히 빈 행 - 관측값이 아니므로 제외", all_blank_rows)

    working = raw.loc[~all_blank_mask, REQUIRED_COLUMNS].copy()
    raw_hourly_missing = int(raw[HOUR_COLUMNS].isna().sum().sum())

    # 날짜 변환 실패를 NaT로 표시하면 잘못된 날짜를 조용히 통과시키지 않을 수 있다.
    original_dates = working[DATE_COLUMN].copy()
    working[DATE_COLUMN] = pd.to_datetime(working[DATE_COLUMN], errors="coerce")
    invalid_date_mask = working[DATE_COLUMN].isna()
    invalid_date_rows = working.loc[invalid_date_mask, REQUIRED_COLUMNS].copy()
    if not invalid_date_rows.empty:
        invalid_date_rows.insert(0, "원본_날짜", original_dates.loc[invalid_date_mask])
        print_rows("날짜 변환 실패 행 - 집계할 수 없어 제외", invalid_date_rows)

    # 시간별 값은 숫자로 변환한다. 숫자가 아닌 값은 결측값으로 바꾸고 반드시 보고한다.
    for column in HOUR_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    partial_mask = ~invalid_date_mask & working[HOUR_COLUMNS].isna().any(axis=1)
    partial_rows = working.loc[partial_mask, REQUIRED_COLUMNS].copy()
    print_rows(
        "시간별 값이 불완전한 행 - 24시간 일계가 왜곡되므로 제외",
        partial_rows,
    )

    # 일별 총수요는 24시간이 모두 있어야 유효하므로 날짜 오류/부분 결측 행을 제외한다.
    cleaned = working.loc[~invalid_date_mask & ~partial_mask].copy()
    cleaned["source_file"] = path.name
    cleaned["snapshot_date"] = snapshot_date_from_name(path)

    if cleaned.empty:
        print("  유효 행이 없습니다.")
    else:
        print(
            f"  유효 행 수: {len(cleaned):,}, "
            f"날짜 범위: {cleaned[DATE_COLUMN].min().date()} ~ "
            f"{cleaned[DATE_COLUMN].max().date()}"
        )

    audit = {
        "raw_rows": len(raw),
        "all_blank_rows": int(all_blank_mask.sum()),
        "invalid_date_rows": int(invalid_date_mask.sum()),
        "partial_rows": int(partial_mask.sum()),
        "raw_hourly_missing": raw_hourly_missing,
    }
    return cleaned, audit


def report_cross_file_overlaps(all_data: pd.DataFrame) -> None:
    """같은 날짜가 어느 파일에서 겹치는지 처리 전에 요약한다."""
    print("\n" + "=" * 72)
    print("파일 간 중복 날짜 점검 (최종 파일 선택 전)")
    print("=" * 72)

    target = all_data.loc[all_data[DATE_COLUMN].between(START_DATE, END_DATE)].copy()
    files = sorted(target["source_file"].unique())
    overlap_found = False

    for left_file, right_file in combinations(files, 2):
        left = target.loc[target["source_file"] == left_file, [DATE_COLUMN, *HOUR_COLUMNS]]
        right = target.loc[target["source_file"] == right_file, [DATE_COLUMN, *HOUR_COLUMNS]]
        overlap_dates = pd.Index(left[DATE_COLUMN]).intersection(right[DATE_COLUMN])
        if overlap_dates.empty:
            continue

        overlap_found = True
        left_values = left.set_index(DATE_COLUMN).loc[overlap_dates, HOUR_COLUMNS].sort_index()
        right_values = right.set_index(DATE_COLUMN).loc[overlap_dates, HOUR_COLUMNS].sort_index()
        different_days = int((left_values != right_values).any(axis=1).sum())
        print(
            f"- {left_file} <-> {right_file}: {len(overlap_dates):,}일 중복, "
            f"범위 {overlap_dates.min().date()} ~ {overlap_dates.max().date()}, "
            f"값이 다른 날짜 {different_days:,}일"
        )

    if not overlap_found:
        print("파일 간 중복 날짜가 없습니다.")
    else:
        print(
            "원인: 같은 누적 기간을 담은 중간 스냅샷과 연말 최종 파일이 함께 있어 "
            "날짜가 겹칩니다. 각 연도에서 가장 늦게 공개된 파일을 선택합니다."
        )


def select_latest_file_per_year(all_data: pd.DataFrame) -> pd.DataFrame:
    """연도별 최신 스냅샷만 선택해 중간 파일의 중복 반영을 막는다."""
    target = all_data.loc[all_data[DATE_COLUMN].between(START_DATE, END_DATE)].copy()
    target["record_year"] = target[DATE_COLUMN].dt.year

    # 같은 연도를 포함하는 파일 중 파일명 기준일이 가장 늦은 자료를 최종본으로 본다.
    candidates = (
        target.groupby(["record_year", "source_file"], as_index=False)
        .agg(
            snapshot_date=("snapshot_date", "max"),
            rows=(DATE_COLUMN, "size"),
            first_date=(DATE_COLUMN, "min"),
            last_date=(DATE_COLUMN, "max"),
        )
        .sort_values(["record_year", "snapshot_date", "source_file"])
    )
    chosen = candidates.groupby("record_year", as_index=False).tail(1).copy()

    print("\n연도별 선택 파일 (가장 늦은 스냅샷)")
    print(
        chosen[["record_year", "source_file", "rows", "first_date", "last_date"]]
        .to_string(index=False)
    )

    selected_parts = []
    for row in chosen.itertuples(index=False):
        selected_parts.append(
            target.loc[
                (target["record_year"] == row.record_year)
                & (target["source_file"] == row.source_file)
            ]
        )
    selected = pd.concat(selected_parts, ignore_index=True)
    return selected.drop(columns="record_year")


def resolve_selected_duplicates(selected: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """최종 파일 선택 후에도 남는 중복은 값 일치 여부를 확인하고 처리한다."""
    duplicate_mask = selected.duplicated(DATE_COLUMN, keep=False)
    duplicate_rows = selected.loc[
        duplicate_mask, [DATE_COLUMN, "source_file", *HOUR_COLUMNS]
    ].sort_values([DATE_COLUMN, "source_file"])

    duplicate_date_count = int(duplicate_rows[DATE_COLUMN].nunique())
    print_rows("최종 선택 자료의 중복 날짜", duplicate_rows)
    if duplicate_date_count == 0:
        return selected, 0

    # 중복 날짜의 24시간 값이 다르면 어느 값을 쓸 근거가 없으므로 즉시 중단한다.
    conflicts = []
    for date, group in duplicate_rows.groupby(DATE_COLUMN):
        if len(group[HOUR_COLUMNS].drop_duplicates()) > 1:
            conflicts.append(date)
    if conflicts:
        raise ValueError(
            "같은 날짜인데 시간별 값이 다른 중복이 있습니다. 임의로 선택하지 않습니다: "
            + ", ".join(date.strftime("%Y-%m-%d") for date in conflicts)
        )

    print("중복 행의 24시간 값이 완전히 같아 동일 관측의 반복으로 판단하고 1행만 유지합니다.")
    deduplicated = selected.drop_duplicates(DATE_COLUMN, keep="first")
    return deduplicated, duplicate_date_count


def build_aggregates(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """24시간 MWh를 일별 GWh로 바꾸고 월별·연별로 합산한다."""
    selected = selected.sort_values(DATE_COLUMN).reset_index(drop=True)

    # 1 MWh = 0.001 GWh이므로 24시간 합계를 1,000으로 나눈다.
    selected["daily_demand_gwh"] = selected[HOUR_COLUMNS].sum(axis=1, min_count=24) / 1000
    selected["year"] = selected[DATE_COLUMN].dt.year
    selected["month"] = selected[DATE_COLUMN].dt.month

    daily = selected[
        [DATE_COLUMN, "year", "month", "daily_demand_gwh", "source_file"]
    ].rename(columns={DATE_COLUMN: "date"})

    # 월별 합계와 일수는 계절성 분석과 월별 완전성 확인에 사용할 수 있다.
    monthly = (
        daily.groupby(["year", "month"], as_index=False)
        .agg(
            monthly_demand_gwh=("daily_demand_gwh", "sum"),
            day_count=("date", "size"),
        )
        .sort_values(["year", "month"])
    )
    monthly.insert(
        0,
        "month_start",
        pd.to_datetime(
            {"year": monthly["year"], "month": monthly["month"], "day": 1}
        ),
    )

    # 연별 합계는 장기 추세 비교에 바로 사용할 수 있는 구조로 만든다.
    yearly = (
        monthly.groupby("year", as_index=False)
        .agg(
            yearly_demand_gwh=("monthly_demand_gwh", "sum"),
            day_count=("day_count", "sum"),
        )
        .sort_values("year")
    )
    return daily, monthly, yearly


def validate_and_print(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    yearly: pd.DataFrame,
    duplicate_date_count: int,
    raw_hourly_missing: int,
) -> bool:
    """기간 완전성과 집계 일치 여부를 검증하고 통과/확인 필요를 나눠 출력한다."""
    expected_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    actual_dates = pd.DatetimeIndex(daily["date"])
    missing_dates = expected_dates.difference(actual_dates)
    unexpected_dates = actual_dates.difference(expected_dates)
    final_hourly_missing = int(daily["daily_demand_gwh"].isna().sum())

    # 독립적인 두 경로로 연간 합계를 계산해 월별 집계 누락 여부를 확인한다.
    annual_from_daily = daily.groupby("year")["daily_demand_gwh"].sum()
    annual_from_monthly = monthly.groupby("year")["monthly_demand_gwh"].sum()
    comparison = pd.concat(
        [
            annual_from_daily.rename("daily_direct_gwh"),
            annual_from_monthly.rename("monthly_resummed_gwh"),
        ],
        axis=1,
    ).reset_index()
    comparison["difference_gwh"] = (
        comparison["monthly_resummed_gwh"] - comparison["daily_direct_gwh"]
    )

    yearly_check = yearly.merge(comparison, on="year", how="outer")
    yearly_check["yearly_vs_daily_difference_gwh"] = (
        yearly_check["yearly_demand_gwh"] - yearly_check["daily_direct_gwh"]
    )

    aggregation_ok = bool(
        np.allclose(comparison["difference_gwh"], 0, rtol=0, atol=1e-9)
        and np.allclose(
            yearly_check["yearly_vs_daily_difference_gwh"], 0, rtol=0, atol=1e-9
        )
    )
    period_ok = bool(
        len(daily) == len(expected_dates)
        and not missing_dates.size
        and not unexpected_dates.size
        and daily["date"].min() == START_DATE
        and daily["date"].max() == END_DATE
    )
    quality_ok = duplicate_date_count == 0 and final_hourly_missing == 0

    print("\n" + "=" * 72)
    print("최종 검증 결과")
    print("=" * 72)
    print(f"최종 데이터 시작일/종료일: {daily['date'].min().date()} ~ {daily['date'].max().date()}")
    print(f"전체 일수: {len(daily):,}일 (예상 {len(expected_dates):,}일)")
    print(f"결측 날짜 수: {len(missing_dates):,}일")
    if missing_dates.size:
        print("결측 날짜:", ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates))
    print(f"범위 밖 날짜 수: {len(unexpected_dates):,}일")
    print(f"최종 선택 자료의 중복 날짜 수: {duplicate_date_count:,}일")
    print(f"원본 파일 전체의 시간별 결측값 수: {raw_hourly_missing:,}개")
    print(f"최종 일별 데이터의 시간별 결측 영향 행 수: {final_hourly_missing:,}일")
    print("\n연도별 데이터 건수")
    print(daily.groupby("year").size().rename("day_count").to_string())
    print(
        f"\n일별 총수요량 최소/최대: {daily['daily_demand_gwh'].min():,.3f} / "
        f"{daily['daily_demand_gwh'].max():,.3f} GWh"
    )
    print("\n월별 재합산 연간값과 일별 직접 합산 연간값의 차이")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:,.9f}"))

    passed = []
    needs_review = []
    (passed if period_ok else needs_review).append("2015-01-01~2025-12-31 날짜 연속성")
    (passed if aggregation_ok else needs_review).append("일별→월별→연별 합계 일치")
    (passed if quality_ok else needs_review).append("최종 중복 및 시간별 결측 없음")

    print("\n[검증 통과]")
    for item in passed:
        print(f"- {item}")
    print("\n[확인 필요]")
    if needs_review:
        for item in needs_review:
            print(f"- {item}")
    else:
        print("- 없음")
    return not needs_review


def main() -> None:
    args = parse_arguments()
    paths = sorted(args.input_dir.glob(FILE_PATTERN))
    if not paths:
        raise FileNotFoundError(
            f"{args.input_dir.resolve()}에서 '{FILE_PATTERN}' 파일을 찾지 못했습니다."
        )

    print(f"발견한 원본 파일: {len(paths):,}개")
    cleaned_parts = []
    audits = []
    for path in paths:
        cleaned, audit = read_and_audit_file(path)
        cleaned_parts.append(cleaned)
        audits.append(audit)

    all_data = pd.concat(cleaned_parts, ignore_index=True)
    raw_hourly_missing = sum(audit["raw_hourly_missing"] for audit in audits)
    print(f"\n모든 파일의 유효 행 통합 건수: {len(all_data):,}행")

    report_cross_file_overlaps(all_data)
    selected = select_latest_file_per_year(all_data)
    selected, duplicate_date_count = resolve_selected_duplicates(selected)
    daily, monthly, yearly = build_aggregates(selected)

    print("\n단계별 데이터 건수와 범위")
    print(
        f"- 일별: {len(daily):,}행, {daily['date'].min().date()} ~ {daily['date'].max().date()}"
    )
    print(
        f"- 월별: {len(monthly):,}행, {monthly['month_start'].min().date()} ~ "
        f"{monthly['month_start'].max().date()}"
    )
    print(f"- 연별: {len(yearly):,}행, {yearly['year'].min()} ~ {yearly['year'].max()}")

    # 결과는 원본과 분리하고, 한글이 엑셀에서도 잘 보이도록 UTF-8 BOM으로 저장한다.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_dir / "daily_demand.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(args.output_dir / "monthly_demand.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(args.output_dir / "yearly_demand.csv", index=False, encoding="utf-8-sig")
    print(f"\n결과 저장 폴더: {args.output_dir.resolve()}")

    all_ok = validate_and_print(
        daily,
        monthly,
        yearly,
        duplicate_date_count,
        raw_hourly_missing,
    )
    if not all_ok:
        raise SystemExit("검증에서 확인할 항목이 발견되었습니다. 위 출력 내용을 확인하세요.")


if __name__ == "__main__":
    main()
