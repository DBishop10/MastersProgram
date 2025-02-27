import pandas as pd
from math import radians, sin, cos, sqrt, asin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class ETL_Pipeline:
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
        data['trans_date'] = pd.to_datetime(data['trans_date_trans_time'])
        data['hour'] = data['trans_date'].dt.hour
        data['day_of_week'] = data['trans_date'].dt.day_name()
        data['month'] = data['trans_date'].dt.month_name()
        
        # Calculate age of the customer
        data['dob'] = pd.to_datetime(data['dob'])
        data['age'] = data['trans_date'].dt.year - data['dob'].dt.year

        # Bin ages into categories
        bins = [0, 18, 30, 40, 50, 60, 70, 80, 90, 100]
        labels = ['<18', '18-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90+']
        data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)

        # Encode categorical variables
        data['sex'] = data['sex'].map({'F': 0, 'M': 1})  # Example for gender
        # For other categorical variables consider using pd.get_dummies() or sklearn's OneHotEncoder

        # Calculate distance using haversine formula
        # Assuming we have a haversine function defined elsewhere
        data['distance'] = data.apply(
            lambda row: haversine(row['long'], row['lat'], row['merch_long'], row['merch_lat']),
            axis=1
        )

        # Normalize/standardize transaction amount
        # Assuming a function standardize() that standardizes the series
        data['amt_standardized'] = standardize(data['amt'])

        # Create a feature for customer's activity level
        transaction_counts = data['cc_num'].value_counts().to_dict()
        data['activity_level'] = data['cc_num'].map(transaction_counts)
        
        # List of categorical variables to encode
        categorical_variables = ['category', 'state', 'job', 'day_of_week', 'month', 'age_group']

        # Iterate over each categorical variable to encode
        for var in categorical_variables:
            if var in data.columns:
                # Initialize label encoder and fit to the data
                label_encoder = LabelEncoder()
                data[f'{var}_encoded'] = label_encoder.fit_transform(data[var])
                
                # Save the label encoder for inverse_transform operations if needed
                self.encoders[var] = label_encoder
                
                # Create and save mapping DataFrame
                mapping_df = pd.DataFrame({
                    'encoded': range(len(label_encoder.classes_)),
                    var: label_encoder.classes_
                })
                            
        # Drop original columns not needed for modeling
        data.drop(['Unnamed: 0', 'trans_date_trans_time', 'category', 'cc_num', 'merchant', 'first', 'last', 'long', 'lat', 'merch_long', 'merch_lat',
                   'street', 'city', 'dob', 'trans_num', 'unix_time', 'state', 'job', 'day_of_week', 'month', 'trans_date', 'age_group'], axis=1, inplace=True)

        transformed_data = data 
        return transformed_data
    
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