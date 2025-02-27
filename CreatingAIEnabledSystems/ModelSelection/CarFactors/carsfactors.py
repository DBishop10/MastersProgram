# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class carsfactors:
    def __init__(self):
        self.modelLearn = False
        self.stats = 0

    def model_learn(self):
        # Importing the dataset into a pandas dataframe
        df = pd.read_csv("cars.csv")

        # note your selected features to address the concern.  Select "useful" columns.  You do not need many.
        useful_columns = ['price_usd', 'odometer_value', 'year_produced', 'transmission', 'body_type', 'color', 'duration_listed']
        
        #Remove Unwanted Columns
        columns_to_drop = [col for col in df.columns if col not in useful_columns]
        df.drop(columns = columns_to_drop, inplace=True)

        # Seperate X and y (features and label)  The last feature "duration_listed" is the label (y)
        # Seperate X vs Y
        X = df.drop('duration_listed', axis=1)
        y = df['duration_listed']
        
    
        # Do the ordinal Encoder for car type to reflect that some cars are bigger than others.  
        # This is the order 'universal','hatchback', 'cabriolet','coupe','sedan','liftback', 'suv', 'minivan', 'van','pickup', 'minibus','limousine'
        # make sure this is the entire set by using unique()
        # create a seperate dataframe for the ordinal number - so you must strip it out and save the column
        # make sure to save the OrdinalEncoder for future encoding due to inference
        
        body_type = df['body_type'].unique()
        ordinal_encoder = OrdinalEncoder(categories=[body_type])
        df['body_type_ordinal'] = ordinal_encoder.fit_transform(df[['body_type']])
        df_ordinal = pd.DataFrame(df['body_type_ordinal'], columns=['body_type_ordinal'])
        category_mapping_df = pd.DataFrame({
            'body_type': body_type,
            'encoded_value': range(len(body_type))
        })
        
        category_mapping_df.to_csv('body_type_ordinal_mapping.csv', index=False)
        df.drop("body_type_ordinal",  axis='columns', inplace=True)

        # Do onehotencoder the selected features - again you need to make a new dataframe with just the encoding of the transmission
        # save the OneHotEncoder to use for future encoding of transmission due to inference
 
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        transmission_encoded = onehot_encoder.fit_transform(df[['transmission']])
        transmission_categories = onehot_encoder.categories_[0]
        transmission_encoded_df = pd.DataFrame(transmission_encoded, columns=transmission_categories)
        pd.DataFrame(transmission_categories, columns=['transmission_type']).to_csv('transmission_types.csv', index=False)
        
        # Do onehotencoder for Color
        # Save the OneHotEncoder to use for future encoding of color for inference

        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        color_encoded = onehot_encoder.fit_transform(df[['color']])
        color_categories = onehot_encoder.categories_[0]
        color_encoded_df = pd.DataFrame(color_encoded, columns=color_categories)
        pd.DataFrame(color_categories, columns=['color_type']).to_csv('color_types.csv', index=False)
        
        # combine all three together endocdings into 1 data frame (need 2 steps with "concatenate")
        # add the ordinal and transmission then add color
        
        encoded_df = pd.concat([df_ordinal, transmission_encoded_df, color_encoded_df], axis=1)
        
        # then dd to original data set
        
        df_final = pd.concat([df, encoded_df], axis=1)
        
        #delete the columns that are substituted by ordinal and onehot - delete the text columns for color, transmission, and car type 

        df_final.drop(['body_type', 'transmission', 'color'], axis=1, inplace=True)
        
        X = df_final.drop('duration_listed', axis=1)
        y = df_final['duration_listed']
        
        # Splitting the dataset into the Training set and Test set 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)        
        
        # Feature Scaling - required due to different orders of magnitude across the features
        # make sure to save the scaler for future use in inference
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        scaler_params = pd.DataFrame({
            'mean': self.scaler.mean_,
            'scale': self.scaler.scale_
        })

        # Save the parameters to a CSV file
        scaler_params.to_csv('scaler_parameters.csv', index=False)
                
        # Select useful model to deal with regression (it is not categorical for the number of days can vary quite a bit)
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)
        
        self.stats = self.model.score(X_train, y_train)
        self.modelLearn = True

    # this demonstrates how you have to convert these values using the encoders and scalers above (if you choose these columns - you are free to choose any you like)
    def model_infer(self, transmission, color, odometer, year, bodytype, price):
        if(self.modelLearn == False):
            self.model_learn()

        scaler_parameters = pd.read_csv('scaler_parameters.csv')
        mean_values = scaler_parameters['mean'].values
        scale_values = scaler_parameters['scale'].values

            
        #convert the body type into a numpy array that holds the correct encoding
        mapping_df = pd.read_csv('body_type_ordinal_mapping.csv')
        body_type_mapping = pd.Series(mapping_df.encoded_value.values, index=mapping_df.body_type).to_dict()
        carTypeTest = np.array([[body_type_mapping[bodytype]]])

        #convert the transmission into a numpy array with the correct encoding
        mapping_df = pd.read_csv('transmission_types.csv')
        transmission_type_mapping = pd.Series(mapping_df.index, index=mapping_df.transmission_type).to_dict()
        one_hot_transmission = np.zeros((1, len(transmission_type_mapping)))
        transmission_index = transmission_type_mapping[transmission]
        one_hot_transmission[0, transmission_index] = 1
        carHotTransmissionTest = one_hot_transmission

        #conver the color into a numpy array with the correct encoding
        mapping_df = pd.read_csv('color_types.csv')
        color_type_mapping = pd.Series(mapping_df.index, index=mapping_df.color_type).to_dict()
        one_hot_color = np.zeros((1, len(color_type_mapping)))
        color_index = color_type_mapping[color]
        one_hot_color[0, color_index] = 1
        carHotColorTest = one_hot_color
        
        #add the three above
        total = np.concatenate((carTypeTest,carHotTransmissionTest), 1)
        total = np.concatenate((total,carHotColorTest), 1)
        
        # build a complete test array and then predict
        othercolumns = np.array([[odometer ,year, price]])
        totaltotal = np.concatenate((othercolumns, total),1)
                
        #must scale
        attempt = self.scaler.transform(totaltotal)
        
        #determine prediction
        y_pred = self.model.predict(attempt)
        return str(y_pred)
        
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)
