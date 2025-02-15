from concurrent.futures import ThreadPoolExecutor

from bin.util import *
from core.data.const import *
from core.data.util import *

# TODO: Integrate with select
def async_pad(datetime_fr_str: str, datetime_to_str: str, data_file_path: str, candle_unit: str, append: bool=False):
    time_begin = time()

    # Fetch data into a dict
    df = pd.read_csv(data_file_path, header=0, dtype=str) # Read as str type to avoid to_simple_str() calls as in other data commands
    data_dict: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        cur_datetime_key = row[0]
        data_dict[cur_datetime_key] = row.tolist()

    # Pad empty rows for given datetime range
    padded_data: list[list[str]] = []
    datetime_fr = datetime.fromisoformat(datetime_fr_str)
    datetime_to = datetime.fromisoformat(datetime_to_str)
    for _, cur_datetime in range_datetimes(datetime_fr, datetime_to, candle_unit):
        cur_datetime_key = cur_datetime.isoformat()
        if cur_datetime_key in data_dict:
            padded_data.append(data_dict[cur_datetime_key])
        else:
            new_row = [cur_datetime_key, None, None, None, None, None]
            padded_data.append(new_row)
    padded_df = pd.DataFrame(padded_data, columns=COLUMNS_CHART_DATA, dtype=str)

    # Overwrite padded data
    if not append:
        padded_df.to_csv(data_file_path, index=False, mode="w", header=True, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    else:
        first_append = not os.path.exists(data_file_path)
        padded_df.to_csv(data_file_path, index=False, mode="a", header=first_append, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")

    logging.info(f"Padded {data_file_path} in {time()-time_begin:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime_fr", help="date from (UTC, isoformat), inclusive", type=str, required=True)
    parser.add_argument("--datetime_to", help="date to (UTC, isoformat), exclusive", type=str, required=True)
    parser.add_argument("--data_glob_pattern", type=str, required=True)

    parser.add_argument("--candle_unit", choices=CANDLE_UNITS, type=str, default="240")
    args = parser.parse_args()

    time_begin = time()

    # Pad in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        for data_file_path, _ in glob(args.data_glob_pattern):
            futures.append(executor.submit(async_pad, args.datetime_fr, args.datetime_to, data_file_path, args.candle_unit))
        for future in futures:
            future.result() # Explicitly call result() to raise Exception encountered in the async function

    logging.info(f"Padded all markets in {time()-time_begin:.2f} seconds")
