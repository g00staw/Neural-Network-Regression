import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'AB_NYC_2019.csv')
df = pd.read_csv(_data_path)

df = df[(df['availability_365'] > 0) & (df['price'] > 0)]

price_cap = df['price'].quantile(0.99)
df = df[df['price'] <= price_cap]

drop_columns = ['id', 'host_id', 'name', 'host_name', 'last_review', 'neighbourhood', 'price']

X = df.drop(columns=drop_columns)

y = np.log1p(df['price'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

num_cols = ['minimum_nights', 'number_of_reviews', 'reviews_per_month',
            'calculated_host_listings_count', 'availability_365', 'latitude', 'longitude']
cat_cols = ['neighbourhood_group', 'room_type']

# RobustScaler zamiast StandardScaler — odporny na outliery (używa mediany i IQR)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', RobustScaler())
])

cat_pipeline = Pipeline([
    ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)