# Nhận diện và phân loại biển báo giao thông 

- Mô hình đã có từ môn Học máy thống kê  (KNN, SVM, CNN) với độ chính xác cao hơn.
- Mô hình Random Forest, XGBoot vừa được đào tạo thêm trong dự án này.
    
File [traffic_sign_clf.py] mô hình CNN được tùy chỉnh phục vụ mục tiêu đề ra trong đồ án môn học này. Lý do chọn CNN là vì dựa trên kết quả so sánh giữa các mô hình thì CNN cho độ chính xác cao nhất, là mô hình phân loại tốt nhất.

## Giới thiệu
- Mục tiêu của dự án này là đào tạo một mô hình phát hiện và phân loại biển báo giao thông của Đức
- Đào tạo các mô hình phân loại: KNN, SVM, RF, CNN
- Bài toán phân loại 1 ảnh nhiều lớp
- Input: Một ảnh hoặc chứa biển báo giao thông.
- Output: Nhận dạng và phân loại biển báo giao thông đó.

## Dataset 
Tập dữ liệu được sử dụng để đào tạo mô hình phân loại biển báo giao thông là [Bien bao gia thong (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) 
  - Tập dữ liệu công khai tại Kaggle, được cập nhật bởi cộng đồng những người làm việc trong lĩnh vực ML, AI mỗi ngày và là một trong những thư viện tập dữ liệu trực tuyến lớn nhất.
  - Ngoài ra, GTSRB là một thử thách phân loại nhiều lớp, được tổ chức tại International Joint Conference on Neural Networks (IJCNN) 2011.
  - Tập dữ liệu gồm có hơn 50.000 hình ảnh, gồm 43 lớp 

![43 Classes Meta](https://user-images.githubusercontent.com/85627308/167721365-159d000f-5664-46b3-a048-019d69366696.png)

## Công cụ và Framework hỗ trợ
- [Tensorflow](https://www.tensorflow.org/)
- [Sk-learn](https://scikit-learn.org/)
- [Keras](https://keras.io/)
- [Matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)

## Phương pháp tiếp cận
- Load dữ liệu
- Khám phá, phân tích tập dữ liệu
- Trực quan hóa dữ liệu
- Tiền xử lí (Resizing, Grayscaling, Histogram equalization,...)
- Create model
- Train model
- Fit
- Prediction
- Đánh giá mô hình
DEMO PYTHON streamlit nhadienbienbao.py
  Nhận diện ảnh và đọc ảnh
- Nhóm biển báo
- Tên biển báo
- Mô tả chi tiết
- Mức xử phạt
- Độ chính xác

