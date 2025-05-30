import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import warnings
from sklearn.exceptions import ConvergenceWarning

# Bỏ qua cảnh báo ConvergenceWarning nếu cần thiết
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Bước 1: Đọc dữ liệu từ file CSV
df = pd.read_csv("mail_data.csv")
print("Dữ liệu ban đầu:")
print(df.head())

# Bước 2: Xử lý dữ liệu bị thiếu trong DataFrame bằng cách thay thế giá trị bằng chuỗi rỗng
data = df.where((pd.notnull(df)), " ")
print("\nThông tin dữ liệu:")
data.info()
print("\nKích thước dữ liệu:", data.shape)

# Bước 3: Xử lý trước dữ liệu
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1
data['Message'] = data['Message'].apply(lambda x: re.sub(r'<[^>]+>', '', x))
data['Message'] = data['Message'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

X = data['Message']
Y = data['Category']
print("\nMessage (X):")
print(X.head())
print("\nCategory (Y):")
print(Y.head())

# Bước 4: Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
print("\nKích thước dữ liệu:")
print("X:", X.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Y:", Y.shape)
print("Y_train:", Y_train.shape)
print("Y_test:", Y_test.shape)

# Bước 5: Chuyển đổi văn bản thành các đặc trưng số bằng TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Tối ưu hóa siêu tham số và huấn luyện mô hình Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
model = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train_features, Y_train)

# Lưu lại mô hình với siêu tham số tối ưu
best_model = grid_search.best_estimator_
print("\nSiêu tham số tối ưu:", grid_search.best_params_)

# Bước 7: Đánh giá mô hình với siêu tham số tối ưu
prediction_on_training_data = best_model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print("\nĐộ chính xác trên tập huấn luyện:", accuracy_on_training_data)

prediction_on_test_data = best_model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print("Độ chính xác trên tập kiểm tra với siêu tham số tối ưu:", accuracy_on_test_data)

print("\nĐánh giá trên tập kiểm tra:")
print(classification_report(Y_test, prediction_on_test_data, target_names=['Spam', 'Ham']))

# Bước 8: Triển khai mô hình để phân loại email mới
input_email = ["Ur cash-balance is currently 500 pounds - to maximize ur cash-in now send CASH to 86688 only 150p/msg. CC: 08708800282 HG/Suite342/2Lands Row/W1J6HL"]
input_data_features = feature_extraction.transform(input_email)
prediction = best_model.predict(input_data_features)

print("\nDự đoán cho email mới:")
if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")
