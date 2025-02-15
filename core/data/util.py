from assertpy import assert_that
import csv
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
import math
import os
import pandas as pd
from time import mktime, time, sleep
from typing import Any, Generator

def get_markets_from_data(data_path: str) -> list[str]:
    market_set: set[str] = set()
    for file_name in os.listdir(data_path):
        _, market, _ = parse_data_name(file_name)
        market_set.add(market)
    return list(sorted(market_set))

def parse_data_name(data_name: str) -> tuple[str, str, str]:
    # data_name is no less than 3 group words like "upbit-btc-240" or "upbit-btc-240-chart"
    groups = data_name.split("-")
    assert_that(len(groups)).is_greater_than_or_equal_to(3)
    exchange = groups[0]
    market = groups[1]
    candle_unit = groups[2]
    return exchange, market, candle_unit

def func_relativedelta(candle_unit: str) -> relativedelta:
    if candle_unit.isnumeric():
        return lambda cnt : relativedelta(minutes=cnt*int(candle_unit))
    elif candle_unit == "day":
        return lambda cnt : relativedelta(days=cnt)
    elif candle_unit == "week":
        return lambda cnt : relativedelta(weeks=cnt)
    elif candle_unit == "month":
        return lambda cnt : relativedelta(months=cnt)
    else:
        raise Exception(candle_unit)

def range_datetimes(datetime_fr: datetime, datetime_to: datetime, candle_unit: str) -> Generator[tuple[int, datetime], Any, None]:
    i_240 = 0
    cur_datetime = datetime_fr
    while cur_datetime < datetime_to:
        yield i_240, cur_datetime
        i_240 += 1
        cur_datetime += func_relativedelta(candle_unit)(1)

class CsvWriter:

    def __init__(self, path: str, title_columns: list[str], append: bool):
        self.path = path
        self.title_columns = title_columns
        self.append = append

    def __enter__(self):
        if not self.append:
            self.file = open(self.path, "w", newline="", buffering=1)
            csv_writer = csv.writer(self.file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(self.title_columns)
        else:
            first_append = not os.path.exists(self.path)
            self.file = open(self.path, "a", newline="", buffering=1)
            csv_writer = csv.writer(self.file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
            if first_append:
                csv_writer.writerow(self.title_columns)
        return csv_writer

    def __exit__(self, type, value, traceback):
        self.file.close()

def to_simple_str(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, int):
        return f"{value:d}"
    elif isinstance(value, float):
        if math.isnan(value):
            return ""
        else:
            return f"{value:f}".rstrip("0").rstrip(".")
    else:
        return str(value)

def get_first_record_of(file_path: str):
    with open(file_path, "r") as file:
        file.readline() # Skip the head record
        first_record = file.readline()
    return first_record

def get_datetime_fr_of(file_path: str):
    return datetime.fromisoformat(get_first_record_of(file_path).split(",")[0]) # inclusive

def get_last_record_of(file_path: str):
    with open(file_path, "r") as file:
        file.readline() # Skip the head record
        while True:
            record = file.readline()
            if not record: break
            last_record = record
    return last_record

def get_datetime_to_of(file_path: str):
    data_name = os.path.splitext(os.path.basename(file_path))[0]
    _, _, candle_unit = parse_data_name(data_name)
    return datetime.fromisoformat(get_last_record_of(file_path).split(",")[0]) + func_relativedelta(candle_unit)(1) # exclusive

def check_datetime_fr_integrity(file_path: str, datetime_fr_expected_str: str):
    if datetime_fr_expected_str is None: return 0
    datetime_fr_expected = datetime.fromisoformat(datetime_fr_expected_str)

    # Fetch the actual datetime_fr
    datetime_fr_actual = get_datetime_fr_of(file_path)
    logging.info(f"Expected datetime_fr: {datetime_fr_expected.isoformat()} / Actual datetime_fr: {datetime_fr_actual.isoformat()}")

    # Compare the dates
    if datetime_fr_actual < datetime_fr_expected:
        return -1
    elif datetime_fr_actual == datetime_fr_expected:
        return 0
    else:
        return 1

def check_datetime_to_integrity(file_path: str, datetime_to_expected_str: str):
    if datetime_to_expected_str is None: return 0
    datetime_to_expected = datetime.fromisoformat(datetime_to_expected_str)

    # Fetch the actual datetime_to
    datetime_to_actual = get_datetime_to_of(file_path)
    logging.info(f"Expected datetime_to: {datetime_to_expected.isoformat()} / Actual datetime_to: {datetime_to_actual.isoformat()}")

    # Compare the dates
    if datetime_to_actual < datetime_to_expected:
        return -1
    elif datetime_to_actual == datetime_to_expected:
        return 0
    else:
        return 1

def force_select_datetime_range(file_path: str, datetime_select_fr_str: str=None, datetime_select_to_str: str=None):
    # This range selection method is described with "force"
    # to imply that it raises an exception if it lacks data in the given date range
    if not os.path.isfile(file_path): return

    cmp_datetime_fr = check_datetime_fr_integrity(file_path, datetime_select_fr_str)
    cmp_datetime_to = check_datetime_to_integrity(file_path, datetime_select_to_str)
    logging.info(f"cmp_datetime_fr: {cmp_datetime_fr} / cmp_datetime_to: {cmp_datetime_to}")

    # Select records from datetime_select_fr
    if 0 < cmp_datetime_fr:
        raise Exception(f"{file_path} is more than one interval ahead of the target date {datetime_select_fr_str}")
    elif cmp_datetime_fr < 0:
        # Read records from datetime_select_fr
        datetime_select_fr = datetime.fromisoformat(datetime_select_fr_str)
        records = []
        with open(file_path, "r") as file:
            i = 0
            while True:
                record = file.readline()
                if not record: break
                if i == 0:
                    records.append(record) # Append the head record as it is
                else:
                    date = datetime.fromisoformat(record.split(",")[0])
                    if datetime_select_fr <= date:
                        records.append(record)
                i += 1

        # Overwrite the records with proper new line characters
        with open(file_path, "w") as file:
            for record in records:
                file.write(record.rstrip("\n").rstrip("\r") + "\r\n")
        logging.info(f"Selected records from {datetime_select_fr.isoformat()} for {file_path}")

    # Select records to datetime_select_to
    if cmp_datetime_to < 0:
        raise Exception(f"{file_path} is more than one interval behind of the target date {datetime_select_to_str}")
    elif 0 < cmp_datetime_to:
        # Read records to datetime_select_to
        datetime_select_to = datetime.fromisoformat(datetime_select_to_str)
        records = []
        with open(file_path, "r") as file:
            i = 0
            while True:
                record = file.readline()
                if not record: break
                if i == 0:
                    records.append(record) # Append the head record as it is
                else:
                    date = datetime.fromisoformat(record.split(",")[0])
                    if datetime_select_to <= date: break
                    records.append(record)
                i += 1

        # Overwrite the records with proper new line characters
        with open(file_path, "w") as file:
            for record in records:
                file.write(record.rstrip("\n").rstrip("\r") + "\r\n")
        logging.info(f"Selected records to {datetime_select_to.isoformat()} for {file_path}")

def find_datetime_indexs(input_path: str, datetime_fr: datetime, datetime_to: datetime):
    i_fr = None
    i_to = None
    with open(input_path, "r") as file:
        file.readline() # Skip the head record
        i = 0
        while True:
            record = file.readline()
            if not record: break

            date = datetime.fromisoformat(record.split(",")[0])
            if datetime_fr and i_fr is None and datetime_fr <= date:
                i_fr = i
            if datetime_to and i_to is None and datetime_to <= date:
                i_to = i
            if i_fr and i_to:
                assert_that(i_fr).is_less_than(i_to)
                break
            i += 1
    return i_fr, i_to

def assert_datetime_integrity(data: pd.DataFrame, candle_unit: str):
    acc_date = datetime.fromisoformat(data.loc[0, "datetime"])
    for _, row in data.iterrows():
        cur_date = datetime.fromisoformat(row["datetime"])
        assert_that(acc_date).is_equal_to(cur_date)
        acc_date += func_relativedelta(candle_unit)(1)

# https://gist.github.com/jmoz/1f93b264650376131ed65875782df386
def RSI(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """See source https://github.com/peerchemist/finta
    and fix https://www.tradingview.com/wiki/Talk:Relative_Strength_Index_(RSI)
    Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
    RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30.
    Signals can also be generated by looking for divergences, failure swings and centerline crossovers.
    RSI can also be used to identify the general trend."""

    delta = ohlc["close"].diff()

    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)), name="RSI")

# https://github.com/peerchemist/finta/blob/master/finta/finta.py#L58
def SMA(ohlc: pd.DataFrame, period: int = 41, column: str = "close") -> pd.Series:
    """
    Simple moving average - rolling mean in pandas lingo. Also known as "MA".
    The simple moving average (SMA) is the most basic of the moving averages used for trading.
    """

    return pd.Series(
        ohlc[column].rolling(window=period).mean(),
        name="{0} period SMA".format(period),
    )

# https://github.com/peerchemist/finta/blob/master/finta/finta.py#L935
def BBANDS(
    ohlc: pd.DataFrame,
    period: int = 20,
    MA: pd.Series = None,
    column: str = "close",
    std_multiplier: float = 1.95, # For some unknown reason, approximately 1.95 best reconstructs BBANDS of the upbit tradingview
) -> pd.DataFrame:
    """
        Developed by John Bollinger, Bollinger Bands® are volatility bands placed above and below a moving average.
        Volatility is based on the standard deviation, which changes as volatility increases and decreases.
        The bands automatically widen when volatility increases and narrow when volatility decreases.

        This method allows input of some other form of moving average like EMA or KAMA around which BBAND will be formed.
        Pass desired moving average as <MA> argument. For example BBANDS(MA=TA.KAMA(20)).
        """

    std = ohlc[column].rolling(window=20).std()

    if not isinstance(MA, pd.core.series.Series):
        middle_band = pd.Series(SMA(ohlc, period), name="BB_MIDDLE")
    else:
        middle_band = pd.Series(MA, name="BB_MIDDLE")

    upper_bb = pd.Series(middle_band + (std_multiplier * std), name="BB_UPPER")
    lower_bb = pd.Series(middle_band - (std_multiplier * std), name="BB_LOWER")

    return pd.concat([upper_bb, middle_band, lower_bb], axis=1)
