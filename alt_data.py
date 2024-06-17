# Importing modules
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Alternative_Features class
class Alternative_Features:
    
    """
    This class processes and returns alternative feature data.

    @METHODS:
    - __init__: Class constructor
    - on_init: Method to handle logical steps
    - set_columns_to_drop: Method to set value of columns to drop
    - get_features: Method to get DataFrame of features
    - set_alt_features: Method to set the alternative features values
    - set_scaler: Method to set the value of the scaler
    - transform_features: Method to scale features
    - set_scaled_features: Method to set scaled feautre data
    - get_feature_data: Method to return feature data
    """


    # Class constructor
    def __init__(self, X, X_train, X_test):

        """
            Initialized variables and calls step logic method
        """

        # Class variables
        self.original_columns_names = ['SPECIAL_CONSIDERATIONS', 'STATUS', 'USE_CASE']
        self.columns_to_drop = []
        self.X = X
        self.X_train = X_train
        self.X_test = X_test
        self.X_alt_scaler = None
        self.X_train_alt = None
        self.X_test_alt = None
        self.X_train_scaled_alt = None
        self.X_test_scaled_alt = None
        self.on_init()

        # Method to handle step logic
    def on_init(self):

        """
            Handles step logic
        """

        # Sets columns to drop
        self.set_columns_to_drop()
        # Sets alternative features
        self.set_alt_features()
        # Sets scaler
        self.set_scaler()
        # Sets scaled features
        self.set_scaled_features()

        # Method to set columns to drop
    def set_columns_to_drop(self):

        """
            Sets columns to drop
        """

        # Loop over original columns names
        for i in self.original_columns_names:
            # Get feature names
            for feature_name in self.X.columns[pd.Series(self.X.columns).str.startswith(i)]:
                # Add feature name to columns to drop
                self.columns_to_drop.append(feature_name)

       # Method to get features
    def get_features(self,features):

        """
           Gets features
        """
        # Return features
        return features.copy().drop(self.columns_to_drop, axis=1)

        # Method to set alternative features
    def set_alt_features(self):

        """
            Sets alternative features
        """

        # Set alternative training features
        self.X_train_alt = self.get_features(self.X_train)
        # Set alternative test features
        self.X_test_alt = self.get_features(self.X_test)

        # Method to set scaler value
    def set_scaler(self):

        """
            Sets scaler
        """

        # Set value for scaler
        self.X_alt_scaler = StandardScaler().fit(self.X_train_alt)

        # Method to return and transform feature
    def transform_features(self, features):

        """
            Transforms feature
        """

        # Return/transform feature
        return self.X_alt_scaler.transform(features)

        # Method to set scaled features
    def set_scaled_features(self):

        """
            Sets scaled feature data
        """

        # Set scaled training features
        self.X_train_scaled_alt = self.transform_features(self.X_train_alt)
        # Set scaled test features
        self.X_test_scaled_alt = self.transform_features(self.X_test_alt)

        # Method to return feature data
    def get_feature_data(self):

        """
            Returns feature data
        """

        # Returning feature data
        return self.X_train_scaled_alt, self.X_test_scaled_alt, self.X_test_alt