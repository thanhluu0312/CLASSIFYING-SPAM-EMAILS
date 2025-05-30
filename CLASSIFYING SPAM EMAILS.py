import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Đường dẫn đến file dữ liệu
file_path = r"spambase.csv"  # Thay thế bằng đường dẫn thực tế đến file spambase.csv

# Đặt tên cột cho DataFrame (57 cột đặc trưng + 1 cột nhãn)
column_names = [f'word_{i}' for i in range(1, 58)] + ['label']

# Đọc file CSV
df = pd.read_csv(file_path, header=None, names=column_names)

# Kiểm tra thông tin của DataFrame
print("Dữ liệu đầu tiên:")
print(df.head())
print("\nSố lượng dữ liệu:", df.shape)

# Chia dữ liệu thành đầu vào (X) và đầu ra (y)
X = df.drop('label', axis=1)
y = df['label']

# Chia dữ liệu thành tập huấn luyện và kiểm tra (70% huấn luyện, 30% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình Logistic Regression
model = LogisticRegression(max_iter=2000, solver='saga')
model.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_scaled)

# Tính toán các chỉ số đánh giá cho Logistic Regression
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# In kết quả đánh giá
print("\nĐánh giá mô hình Logistic Regression:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Ma trận nhầm lẫn cho Logistic Regression
cm_logistic = confusion_matrix(y_test, y_pred)
print("\nMa trận nhầm lẫn (Logistic Regression):")
print(cm_logistic)

# Vẽ ma trận nhầm lẫn cho Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(cm_logistic, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Not Spam', 'Spam'], 
            yticklabels=['Not Spam', 'Spam'])
plt.title('Ma trận nhầm lẫn (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('logistic_confusion_matrix.png')
plt.close()

# Phân cụm dữ liệu với KMeans
kmeans = KMeans(n_clusters=2, random_state=42)  # 2 cụm: spam và không spam
kmeans.fit(X_train_scaled)

# Dự đoán cụm trên tập kiểm tra
kmeans_labels = kmeans.predict(X_test_scaled)

# Đánh giá phân cụm
# Tính Adjusted Rand Index để so sánh với nhãn thực tế
ari_score = adjusted_rand_score(y_test, kmeans_labels)
print("\nĐánh giá phân cụm KMeans:")
print(f"Adjusted Rand Index: {ari_score:.2f}")

# Tính độ chính xác của KMeans (ánh xạ nhãn tối ưu)
kmeans_accuracy = max(accuracy_score(y_test, kmeans_labels), 
                     accuracy_score(y_test, 1 - kmeans_labels))  # Đảo nhãn nếu cần
print(f"Accuracy (KMeans): {kmeans_accuracy:.2f}")

# Ma trận nhầm lẫn cho KMeans
cm_kmeans = confusion_matrix(y_test, kmeans_labels)
print("\nMa trận nhầm lẫn (KMeans):")
print(cm_kmeans)

# Vẽ ma trận nhầm lẫn cho KMeans
plt.figure(figsize=(8, 6))
sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Spam', 'Spam'], 
            yticklabels=['Not Spam', 'Spam'])
plt.title('Ma trận nhầm lẫn (KMeans)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('kmeans_confusion_matrix.png')
plt.close()

# Biểu đồ so sánh độ chính xác
models = ['Logistic Regression', 'KMeans']
accuracies = [accuracy, kmeans_accuracy]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['#4CAF50', '#2196F3'])
plt.title('So sánh độ chính xác: Logistic Regression vs KMeans')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for bar, v in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.2f}', ha='center')
plt.savefig('accuracy_comparison.png')
plt.close()

# Biểu đồ 1: Histogram phân bố nhãn thực tế theo cụm
# Tạo DataFrame chứa nhãn cụm và nhãn thực tế
cluster_df = pd.DataFrame({'Cluster': kmeans_labels, 'Actual Label': y_test})

# Vẽ histogram
plt.figure(figsize=(8, 6))
cluster_df[cluster_df['Actual Label'] == 0].groupby('Cluster').size().plot(kind='bar', 
    color='teal', alpha=0.7, position=0, width=0.4, label='Actual Label 0')
cluster_df[cluster_df['Actual Label'] == 1].groupby('Cluster').size().plot(kind='bar', 
    color='coral', alpha=0.7, position=1, width=0.4, label='Actual Label 1')
plt.title('Phân bố nhãn thực tế trong từng cụm')
plt.xlabel('Cụm')
plt.ylabel('Số lượng')
plt.legend()
plt.savefig('actual_labels_distribution.png')
plt.close()

# Biểu đồ 2: Phân cụm KMeans (Giảm chiều xuống 2D)
# Giảm chiều dữ liệu xuống 2D bằng PCA
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

# Tạo DataFrame chứa dữ liệu 2D và nhãn cụm
pca_df = pd.DataFrame(X_test_pca, columns=['Thành phần chính 1', 'Thành phần chính 2'])
pca_df['Cụm'] = kmeans_labels

# Vẽ scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_df['Thành phần chính 1'], pca_df['Thành phần chính 2'], 
                      c=pca_df['Cụm'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cụm')
plt.title('Phân cụm KMeans (Giảm chiều xuống 2D)')
plt.xlabel('Thành phần chính 1')
plt.ylabel('Thành phần chính 2')
plt.savefig('kmeans_pca_2d.png')
plt.close()

# Phân loại email mới (ví dụ)
new_email_features = [0] * 57  # 57 đặc trưng, tất cả bằng 0
new_email_df = pd.DataFrame([new_email_features], columns=X.columns)

# Chuẩn hóa dữ liệu cho email mới
new_email_scaled = scaler.transform(new_email_df)

# Dự đoán với Logistic Regression
prediction = model.predict(new_email_scaled)
print("\nDự đoán email mới (Logistic Regression):", "Spam" if prediction[0] == 1 else "Not Spam")

# Dự đoán với KMeans
kmeans_prediction = kmeans.predict(new_email_scaled)
print("Dự đoán email mới (KMeans):", "Spam" if kmeans_prediction[0] == 1 else "Not Spam")

# Lưu mô hình Logistic Regression
joblib.dump(model, 'logistic_regression_model.pkl')
print("\nMô hình Logistic Regression đã được lưu vào 'logistic_regression_model.pkl'")

# Lưu mô hình KMeans
joblib.dump(kmeans, 'kmeans_model.pkl')
print("Mô hình KMeans đã được lưu vào 'kmeans_model.pkl'")

# Lưu StandardScaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler đã được lưu vào 'scaler.pkl'")