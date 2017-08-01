RAW_TRAINING_SET_FILE = 'datasets/train.csv'

RAW_TEST_SET_FILE = 'datasets/test.csv'

TRAINING_SET_FILE = 'datasets/processed_train.csv'

TEST_SET_FILE = 'datasets/processed_test.csv'

CV_TRAINING_SET_FILE = 'datasets/split_train_light.csv'

CV_TEST_SET_FILE = 'datasets/split_test_light.csv'

# CV_TRAINING_SET_FILE = 'datasets/split_train.csv'

# CV_TEST_SET_FILE = 'datasets/split_test.csv'

PREDICTION_SET_FILE = 'datasets/sample_submission.csv'

COLUMNS = [
    'passenger_count',
    'vendor_id',
    'store_and_fwd_flag',
    'pickup_hour',
    'day_week',
    'pickup_hour_weekofyear',
    'pickup_pca0',
    'pickup_pca1',
    'dropoff_pca0',
    'dropoff_pca1',
    'manhattan_dist',
    'haversine_dist',
    'trip_to_JFK',
    'trip_from_JFK',
    'trip_from_LGA',
    'trip_to_LGA',
    'work',
    'trip_duration',
    ]

CATEGORICALS = [
        'vendor_id',
        'store_and_fwd_flag',
        'pickup_hour',
        'day_week',
        'pickup_hour_weekofyear',
        'trip_to_JFK',
        'trip_from_JFK',
        'trip_from_LGA',
        'trip_to_LGA',
        'work'
    ]

FEATURES = [
    'vendor_id',
    'store_and_fwd_flag',
    'pickup_hour',
    'day_week',
    'pickup_hour_weekofyear',
    'pickup_pca0',
    'pickup_pca1',
    'dropoff_pca0',
    'dropoff_pca1',
    'manhattan_dist',
    'haversine_dist',
    'trip_to_JFK',
    'trip_from_JFK',
    'trip_from_LGA',
    'trip_to_LGA',
    'work',
    ]

LABEL = 'trip_duration'

MODELS_DIR = 'tensorflow_part/models'

SUBMISSIONS_DIR = 'submissions'
