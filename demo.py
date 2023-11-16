import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pyarrow.parquet as pq 

train_series = pd.read_parquet('./data/train_series.parquet', engine='pyarrow')
train_events = pd.read_csv('./data/train_events.csv')
test_series = pd.read_parquet('./data/test_series.parquet', engine='pyarrow')


train_combined = pd.merge(train_series, train_events, on=['series_id', 'step'], how='left')
train_combined['event'] = train_combined['event'].fillna('none')  # 이벤트가 없는 경우 'none'으로 표시

features = ['anglez', 'enmo']
X = train_combined[features]
y = train_combined['event']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

test_features = test_series[features]
test_predictions = model.predict(test_features)

submission = pd.DataFrame({
    'series_id': test_series['series_id'],
    'step': test_series['step'],
    'event': test_predictions,
    # 'score': 모델에서 제공하는 예측 확률 또는 신뢰도 점수
})

submission.to_csv('submission.csv', index=False)

