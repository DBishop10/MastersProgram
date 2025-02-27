import pandas as pd
from math import radians, sin, cos, sqrt, asin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

class Pipeline:
    """
    A class used to represent an ETL pipeline for transaction data.

    ...

    Attributes
    ----------
    None

    Methods
    -------
    extract(filename):
        Extracts data from a CSV file.
        
    transform(data):
        Transforms the data by cleaning and preparing it for modeling.
        
    load(data, filename):
        Loads the transformed data into a new CSV file.
    """
    
    def __init__(self):
        self.encoders = {}
    
    def extract(self, filename):
        """
        Extracts data from a CSV file.

        Parameters
        ----------
        filename : str
            The path to the CSV file to extract data from.

        Returns
        -------
        data : DataFrame
            The extracted data.
        """
        data = pd.read_csv(filename)
        return data
    
    def transform(self, data):
        """
        Transforms the data by cleaning and preparing it for modeling.

        Parameters
        ----------
        data : DataFrame
            The raw data to transform.

        Returns
        -------
        transformed_data : DataFrame
            The data after transformation.
        """
        # Convert dates and times
        # First, ensure both columns are in string format to concatenate
        df = data.copy(deep=True)

        df['trans_date'] = df['trans_date'].astype(str)
        df['trans_time'] = df['trans_time'].astype(str)

        df['trans_datetime'] = df['trans_date'] + ' ' + df['trans_time']

        df['trans_datetime'] = pd.to_datetime(df['trans_datetime'])
        df['trans_date'] = pd.to_datetime(df['trans_date'])
        df['hour'] = df['trans_datetime'].dt.hour

        # Encoding 'day_of_week' as numbers 1-7
        days_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
        df['day_of_week'] = df['trans_date'].dt.day_name().map(days_map)

        # Encoding 'month' as numbers 1-12
        df['month'] = df['trans_date'].dt.month

        # Encoding 'weekday_vs_weekend' as binary values 0 for Weekday and 1 for Weekend
        df['weekday_vs_weekend'] = df['trans_datetime'].dt.dayofweek.apply(lambda x: 0 if x < 5 else 1)

        categorical_columns = ['hour']
        df = pd.get_dummies(df, columns=categorical_columns)

        cal = calendar()
        holidays = cal.holidays(start=df['trans_datetime'].min(), end=df['trans_datetime'].max())
        df['is_public_holiday'] = df['trans_datetime'].dt.normalize().isin(holidays)

        daily_amounts = df.groupby('trans_date')['amt'].sum().rename('total_amount_per_day')

        first_payday = pd.Timestamp('2014-12-19')
        paydays = pd.date_range(start=first_payday, end=df['trans_datetime'].max(), freq='2W-FRI')
        df['is_payday'] = df['trans_datetime'].dt.date.astype('datetime64[ns]').isin(paydays.date)

        # Aggregating total and fraud transactions per day
        daily_totals = df.groupby('trans_date').size().rename('total_daily_transactions')
        daily_fraud_totals = df.groupby('trans_date')['is_fraud'].sum().rename('total_daily_fraud_transactions')

        hourly_columns = ['hour_' + str(i) for i in range(24)]
        daily_hourly_totals = df.groupby('trans_date')[hourly_columns].sum()

        # Combining aggregated features
        daily_data = pd.DataFrame(daily_totals).join(daily_fraud_totals)
        daily_data = daily_data.merge(daily_hourly_totals, on='trans_date', how='left')
        daily_features = df.groupby('trans_date')[['day_of_week', 'month', 'weekday_vs_weekend']].first()

        # Adding non-aggregated features
        daily_data['is_public_holiday'] = df.groupby('trans_date')['is_public_holiday'].max()
        daily_data['is_payday'] = df.groupby('trans_date')['is_payday'].max()
        daily_data = daily_data.join(daily_features)
        daily_data = daily_data.join(daily_amounts, on='trans_date')
        
        # Resetting index to ensure 'trans_date' is a column for the merge operation
        daily_data.reset_index(inplace=True)

        # Assuming the 'hourly_columns' aggregation has already been performed
        # Merge hourly totals and other features as needed

        daily_data.set_index('trans_date', inplace=True)

        return daily_data
    
    def load(self, data, filename):
        """
        Loads the transformed data into a new CSV file.

        Parameters
        ----------
        data : DataFrame
            The transformed data to load.
        filename : str
            The path to the CSV file to load data into.

        Returns
        -------
        None
        """
        data.to_csv(filename, index=False)
        
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).

    Parameters:
    lon1 (float): Longitude of the first point.
    lat1 (float): Latitude of the first point.
    lon2 (float): Longitude of the second point.
    lat2 (float): Latitude of the second point.

    Returns:
    float: Distance between the two points in kilometers.
    """

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    return c * r

def standardize(series):
    """
    Standardize a pandas series using sklearn's StandardScaler.

    Parameters
    ----------
    series : pandas.Series
        Series to standardize.

    Returns
    -------
    standardized_series : pandas.Series
        The standardized series with mean of 0 and standard deviation of 1.
    """
    scaler = StandardScaler()
    # StandardScaler expects a 2D array, so we use series.values.reshape(-1, 1) to reshape the series
    standardized_series = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return pd.Series(standardized_series, index=series.index)