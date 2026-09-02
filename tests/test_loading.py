"""The parser fixes six documented defects in the published CSVs. Each gets a test."""

import pandas as pd
import pytest

from elmanguard.features import normalise_columns
from elmanguard.loading import parse_timestamps, read_day_file


def test_normalise_columns_strips_the_leading_spaces():
    df = pd.DataFrame(columns=[" Source IP", "Flow ID", " Flow Bytes/s", " Fwd Header Length.1"])
    assert list(normalise_columns(df).columns) == [
        "source_ip", "flow_id", "flow_bytes_s", "fwd_header_length1",
    ]


def test_missing_meridiem_is_restored():
    """`7/7/2017 1:02` is 13:02 -- the capture never runs before 08:00."""
    ts = parse_timestamps(pd.Series(["7/7/2017 9:15", "7/7/2017 1:02"]), "Friday")
    assert ts.iloc[0].hour == 9
    assert ts.iloc[1].hour == 13


def test_day_month_order_resolved_by_the_expected_weekday():
    """`6/7/2017` is 6 July (Thursday), not 7 June (Wednesday)."""
    ts = parse_timestamps(pd.Series(["6/7/2017 9:00"] * 4), "Thursday")
    assert (ts.dt.day_name() == "Thursday").all()
    assert ts.iloc[0].month == 7 and ts.iloc[0].day == 6


def test_impossible_weekday_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError, match="Neither day/month reading"):
        parse_timestamps(pd.Series(["6/7/2017 9:00"] * 4), "Sunday")


def test_rows_with_no_timestamp_do_not_veto_the_reading():
    """One CICIDS2017 file is padded with 288k empty rows; they must be ignored."""
    s = pd.Series(["6/7/2017 9:00"] * 3 + [None] * 20)
    ts = parse_timestamps(s, "Thursday")
    assert ts.notna().sum() == 3


def test_read_day_file_repairs_labels_and_headers(tmp_path):
    header = "Flow ID, Source IP, Timestamp, Flow Duration, Label"
    rows = [
        "a-b-1-2-6,172.16.0.1,6/7/2017 9:00,100,BENIGN",
        header.replace(" ", ""),                       # a repeated header row
        "a-b-1-2-6,172.16.0.1,6/7/2017 9:01,200,Web Attack \x96 Brute Force",
    ]
    path = tmp_path / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
    path.write_bytes(("\n".join([header] + rows)).encode("latin-1"))

    df = read_day_file(path, verbose=False)
    assert len(df) == 2, "the repeated header row should be removed"
    assert set(df["label"]) == {"BENIGN", "Web Attack - Brute Force"}
    assert df["day"].unique().tolist() == ["Thursday"]
