TRAINING_SET_FUSION = 'datasets/fusion/trainingNN1.csv'

CV_SET_FUSION = 'datasets/fusion/trainingNN2.csv'

TEST_SET_FUSION = 'datasets/fusion/trainingNN3.csv'

TEST_SET_FILE_FUSION = 'datasets/fusion/processed_testNEW.csv'

RAW_TRAINING_SET_FILE = 'datasets/train.csv'

RAW_TEST_SET_FILE = 'datasets/test.csv'

TRAINING_SET_FILE = 'datasets/processed_train.csv'

TEST_SET_FILE = 'datasets/processed_test.csv'

CV_TRAINING_SET_FILE = 'datasets/split_train.csv'

CV_TEST_SET_FILE = 'datasets/split_test.csv'

PREDICTION_SET_FILE = 'datasets/sample_submission.csv'

COLUMNS = [
    'passenger_count',
    'vendor_id',
    'store_and_fwd_flag',
    'weekofyear',
    'coords_days_x',
    'coords_days_y',
    'coords_hours_x',
    'coords_hours_y',
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
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
    'log_trip_duration',
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

LABEL = 'log_trip_duration'

MODELS_DIR = 'tensorflow_part/models'

SUBMISSIONS_DIR = 'submissions'
