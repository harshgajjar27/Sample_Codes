import pandas as pd
import numpy as np
from itertools import cycle

data = {'usage_month': ['1/3/2023', '1/3/2023'],
        'billing_account_id': ['01F392-34E6A6-65C4D4', '01F392-34E6A6-65C4D4'],
        'project_number': ['439664739426', '439664739426'],
        'project.id': ['openai-380413', 'openai-380413'],
        'service.id': ['5490-F7B7-8DF6', '95FF-2EF5-5EA1'],
        'service.description': ['Cloud Logging', 'Cloud Storage'],
        'location.location': ['us', 'us'],
        'location.region': ['us-central1', 'us-central1'],
        'carbon_footprint_kgCO2e.scope1': [0.000178561, 0.003228433],
        'carbon_footprint_kgCO2e.scope2.location_based': [0.000240251, 0.004647504],
        'carbon_footprint_kgCO2e.scope2.market_based': [0.000419228, 0.007883978],
        'carbon_footprint_kgCO2e.scope3': [0.002542078, 0.002542078],
        'carbon_footprint_total_kgCO2e.after_offsets': [8, 8],
        'carbon_footprint_total_kgCO2e.market_based': [0, 0],
        'carbon_footprint_total_kgCO2e.location_based': [0, 0],
        'carbon_offsets_kgCO2e': [0.000178561, 0.000178561],
        'carbon_model_version': [8, 8]
        }

data_look = {
    'billing_account_id': ['01F392-34E6A6-65C4D4', '01F392-34E6A6-65C4D4'],
    'project_number': ['439664739426', '439664739426'],
    'project.id': ['openai-380413', 'openai-380413'],
    'service.id': ['5490-F7B7-8DF6', '95FF-2EF5-5EA1'],
    'service.description': ['Cloud Logging', 'Cloud Storage'],
    'location.location': ['us', 'us'],
    'location.region': ['us-central1', 'us-central1'],
    'carbon_footprint_kgCO2e.scope1': [0.000178561, 0.003228433],
    'carbon_footprint_kgCO2e.scope2.location_based': [0.000240251, 0.004647504],
    'carbon_footprint_kgCO2e.scope2.market_based': [0.000419228, 0.007883978],
    'carbon_footprint_kgCO2e.scope3': [0.002542078, 0.002542078],
    'carbon_footprint_total_kgCO2e.after_offsets': [8, 8],
    'carbon_footprint_total_kgCO2e.market_based': [0, 0],
    'carbon_footprint_total_kgCO2e.location_based': [0, 0],
    'carbon_offsets_kgCO2e': [0.000178561, 0.000178561],
    'carbon_model_version': [8, 8]
}

df = pd.DataFrame(data)
print(df)


def generate_fake_dataframe(size, cols, col_names=None, intervals=None, seed=None):
    categories_dict = {
        'billing_account_id': ['01F392-34E6A6-65C4D4'],
        'project_number': ['439664739426', '248712589857', '1094642354898', '30270049481', '248712589857',
                           '478232462941', '478232462738', '8684642354898'],
        'project.id': ['openai-380413', 'cloudgreener', 'harsh-389208', 'sushil-389208', 'varadha', 'ashish-389208',
                       'gyan-389208', 'rajini-389208'],
        'service.id': ['5490-F7B7-8DF6', '95FF-2EF5-5EA1', '8B5D-EF7D-EB12', '95FF-2EF5-5EA1', '6G7H-2EF5-7G01',
                       '95FF-4D2E-5EA1'],
        'service.description': ['Cloud Logging', 'Cloud Storage', 'Cloud Functions', 'Cloud Build', 'Cloud Run',
                                'Vortex AI'],
        'location.location': ['us', 'us-central1', 'eu-central1', 'asia-south1-a', 'melbourne-south1'],
        'location.region': ['us-central1', 'us-central1', 'eu-central1', 'asia', 'melbourne'],
        'carbon_model_version': ['8']
    }
    default_intervals = {"i": (0, 10), "f": (0, 0.01), "c": ("project.id", 7), "d": ("2023-03-01", "2023-03-31")}
    rng = np.random.default_rng(seed)

    first_c = default_intervals["c"][0]
    categories_names = cycle([first_c] + [c for c in categories_dict.keys() if c != first_c])
    default_intervals["c"] = (categories_names, default_intervals["c"][1])

    if isinstance(col_names, list):
        assert len(col_names) == len(
            cols), f"The fake DataFrame should have {len(cols)} columns but col_names is a list with {len(col_names)} elements"
    elif col_names is None:
        suffix = {"c": "cat", "i": "int", "f": "float", "d": "date"}
        col_names = [f"column_{str(i)}_{suffix.get(col)}" for i, col in enumerate(cols)]

    if isinstance(intervals, list):
        assert len(intervals) == len(
            cols), f"The fake DataFrame should have {len(cols)} columns but intervals is a list with {len(intervals)} elements"
    else:
        if isinstance(intervals, dict):
            assert len(
                set(intervals.keys()) - set(default_intervals.keys())) == 0, f"The intervals parameter has invalid keys"
            default_intervals.update(intervals)
        intervals = [default_intervals[col] for col in cols]
    df = pd.DataFrame()
    for col, col_name, interval in zip(cols, col_names, intervals):
        if interval is None:
            interval = default_intervals[col]
        assert (len(interval) == 2 and isinstance(interval, tuple)) or isinstance(interval,
                                                                                  list), f"This interval {interval} is neither a tuple of two elements nor a list of strings."
        if col in ("i", "f", "d"):
            start, end = interval
        if col == "i":
            df[col_name] = rng.integers(start, end, size)
        elif col == "f":
            df[col_name] = rng.uniform(start, end, size)
        elif col == "c":
            if isinstance(interval, list):
                categories = np.array(interval)
            else:
                cat_family, length = interval
                if isinstance(cat_family, cycle):
                    cat_family = next(cat_family)
                assert cat_family in categories_dict.keys(), f"There are no samples for category '{cat_family}'. Consider passing a list of samples or use one of the available categories: {categories_dict.keys()}"
                categories = rng.choice(categories_dict[cat_family], length, replace=False, shuffle=True)
            df[col_name] = rng.choice(categories, size, shuffle=True)
        elif col == "d":
            df[col_name] = rng.choice(pd.date_range(start, end), size)
    return df


df2 = generate_fake_dataframe(size=25000,
                              cols='dcccccccffffffffc',
                              col_names=['usage_month', 'billing_account_id', 'project_number', 'project.id',
                                         'service.id', 'service.description', 'location.location',
                                         'location.region', 'carbon_footprint_kgCO2e.scope1',
                                         'carbon_footprint_kgCO2e.scope2.location_based',
                                         'carbon_footprint_kgCO2e.scope2.market_based',
                                         'carbon_footprint_kgCO2e.scope3',
                                         'carbon_footprint_total_kgCO2e.after_offsets',
                                         'carbon_footprint_total_kgCO2e.market_based',
                                         'carbon_footprint_total_kgCO2e.location_based',
                                         'carbon_offsets_kgCO2e',
                                         'carbon_model_version'],
                              intervals=[None, ('billing_account_id', 1), ('project_number', 8), ('project.id', 8),
                                         ('service.id', 6), ('service.description', 6), ('location.location', 5),
                                         ('location.region', 5), (0.0, 0.003), (0.0, 0.003), (0.0, 0.003), (0.0, 0.003),
                                         None, None, None, None, ('carbon_model_version', 1)])
print(df2)
df2.to_csv('/Users/harshgajjar/Downloads/file2.csv', header=True, index=False)
