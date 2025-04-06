import pandas as pd


def count_repeated_rows(df, threshold, use_hash=True):
    repeated_rows = {}
    n_rows = len(df)
    prev_row = df.iloc[0]
    actual_values = {}
    if use_hash:
        for i, row in df.iterrows():
            key = tuple(round(val / threshold) * threshold for val in row)
            repeated_rows[key] = repeated_rows.get(key, 0) + 1
            if key in actual_values:
                actual_values[key].append(tuple(row))
            else:
                actual_values[key] = [tuple(row)]
    else:
        # Problematic - for testing only
        for i in range(n_rows-1):
            row1 = df.iloc[i]
            for j in range(i + 1, n_rows):
                row2 = df.iloc[j]
                key1 = tuple(row1)
                # key2 = tuple(row2)
                if (abs(row1 - row2) <= threshold).all():
                    if (abs(row1 - prev_row) > threshold).all():
                        repeated_rows[key1] = repeated_rows.get(key1, 0) + 1
                        prev_row = row1
                    else:
                        prv_key = tuple(prev_row)
                        repeated_rows[prv_key] = repeated_rows.get(prv_key, 0) + 1

                # repeated_rows[key2] = repeated_rows.get(key2, 0) + 1

    return repeated_rows, actual_values


if __name__ == "__main__":
    import numpy as np
