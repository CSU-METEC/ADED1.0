import pandas as pd


def bin2(DF, Category, nBins):
    try:
        # Get all of the value for xCategory
        values = DF[Category]
        start = values.min()
        end = values.max()
        Bins = pd.interval_range(start=start, end=end, periods=nBins)
        # Round off bins to 3 sig figs
        return Bins
    except Exception as e:
        print(f'Could not bin values due to exception: {e}')
        return None
