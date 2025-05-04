import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

# Load model
model = load_model('bienbaogiaothong.h5')

# Class labels and descriptions
classNames = {
    0: 'Giới hạn tốc độ (20km/h)',
    1: 'Giới hạn tốc độ (30km/h)',
    2: 'Giới hạn tốc độ (50km/h)',
    3: 'Giới hạn tốc độ (60km/h)',
    4: 'Giới hạn tốc độ (70km/h)',
    5: 'Giới hạn tốc độ (80km/h)',
    6: 'Hết giới hạn tốc độ (80km/h)',
    7: 'Giới hạn tốc độ (100km/h)',
    8: 'Giới hạn tốc độ (120km/h)',
    9: 'Cấm vượt',
    10: 'Cấm xe trên 3.5 tấn vượt',
    11: 'Được ưu tiên tại ngã tư tiếp theo',
    12: 'Đường ưu tiên',
    13: 'Nhường đường',
    14: 'Dừng lại',
    15: 'Cấm xe',
    16: 'Cấm xe trên 3.5 tấn',
    17: 'Cấm vào',
    18: 'Chú ý chung',
    19: 'Khúc cua nguy hiểm bên trái',
    20: 'Khúc cua nguy hiểm bên phải',
    21: 'Liên tiếp nhiều khúc cua',
    22: 'Đường gồ ghề',
    23: 'Đường trơn trượt',
    24: 'Đường hẹp bên phải',
    25: 'Đang thi công đường',
    26: 'Tín hiệu giao thông',
    27: 'Người đi bộ',
    28: 'Trẻ em qua đường',
    29: 'Xe đạp qua đường',
    30: 'Cẩn thận băng/tuyết',
    31: 'Động vật hoang dã băng qua',
    32: 'Hết tất cả giới hạn tốc độ và cấm vượt',
    33: 'Rẽ phải phía trước',
    34: 'Rẽ trái phía trước',
    35: 'Chỉ được đi thẳng',
    36: 'Đi thẳng hoặc rẽ phải',
    37: 'Đi thẳng hoặc rẽ trái',
    38: 'Đi về bên phải',
    39: 'Đi về bên trái',
    40: 'Bắt buộc đi theo vòng xuyến',
    41: 'Hết cấm vượt',
    42: 'Hết cấm vượt với xe trên 3.5 tấn'
}

descriptions = {
    0: "Biển báo giới hạn tốc độ tối đa là 20km/h. Người điều khiển phương tiện không được vượt quá tốc độ này.",
    1: "Biển báo giới hạn tốc độ tối đa là 30km/h. Thường xuất hiện trong khu vực đông dân cư, trường học.",
    2: "Biển báo giới hạn tốc độ tối đa là 50km/h. Thường thấy trên các tuyến đường nội thành.",
    3: "Biển báo giới hạn tốc độ tối đa là 60km/h. Áp dụng cho các đoạn đường đủ an toàn cho tốc độ trung bình.",
    4: "Biển báo giới hạn tốc độ tối đa là 70km/h. Thường xuất hiện ở khu vực ngoại ô, ít phương tiện qua lại.",
    5: "Biển báo giới hạn tốc độ tối đa là 80km/h. Phù hợp cho các tuyến đường quốc lộ, liên tỉnh.",
    6: "Biển báo hết hiệu lực giới hạn tốc độ 80km/h. Phương tiện được phép chạy tốc độ cao hơn nếu điều kiện cho phép.",
    7: "Biển báo giới hạn tốc độ tối đa là 100km/h. Áp dụng cho các tuyến cao tốc hoặc đường lớn.",
    8: "Biển báo giới hạn tốc độ tối đa là 120km/h. Thường chỉ áp dụng tại một số tuyến cao tốc đặc biệt.",
    9: "Biển báo cấm vượt. Cấm các phương tiện vượt nhau trong đoạn đường này.",
    10: "Biển báo cấm xe có trọng lượng lớn hơn 3.5 tấn vượt xe khác.",
    11: "Biển báo xe được ưu tiên đi qua ngã tư kế tiếp.",
    12: "Biển báo đường ưu tiên. Phương tiện đi trên đường này có quyền ưu tiên.",
    13: "Biển báo nhường đường. Phương tiện phải nhường cho xe khác đi trước.",
    14: "Biển báo dừng lại. Phương tiện phải dừng hẳn trước khi tiếp tục di chuyển.",
    15: "Biển báo cấm xe cơ giới đi vào.",
    16: "Biển báo cấm xe có trọng tải trên 3.5 tấn đi vào.",
    17: "Biển báo cấm tất cả các loại phương tiện đi vào.",
    18: "Biển cảnh báo nguy hiểm chung. Người lái xe cần thận trọng.",
    19: "Biển báo khúc cua nguy hiểm phía bên trái.",
    20: "Biển báo khúc cua nguy hiểm phía bên phải.",
    21: "Biển báo nhiều khúc cua liên tiếp. Thường xuất hiện trên đèo, dốc.",
    22: "Biển báo đường gồ ghề, mặt đường không bằng phẳng.",
    23: "Biển báo đường trơn trượt. Cần giảm tốc và lái cẩn thận.",
    24: "Biển báo đường bị hẹp về phía bên phải.",
    25: "Biển báo khu vực đang thi công, sửa chữa đường.",
    26: "Biển báo có tín hiệu giao thông (đèn đỏ, xanh, vàng) phía trước.",
    27: "Biển báo khu vực người đi bộ băng qua đường.",
    28: "Biển báo khu vực có trẻ em băng qua đường. Thường gần trường học.",
    29: "Biển báo khu vực xe đạp băng qua.",
    30: "Biển cảnh báo khu vực có thể có băng hoặc tuyết. Cần lái xe cẩn thận.",
    31: "Biển cảnh báo có động vật hoang dã băng qua đường.",
    32: "Biển báo hết tất cả các giới hạn về tốc độ và cấm vượt trước đó.",
    33: "Biển báo bắt buộc rẽ phải phía trước.",
    34: "Biển báo bắt buộc rẽ trái phía trước.",
    35: "Biển báo chỉ được phép đi thẳng.",
    36: "Biển báo được phép đi thẳng hoặc rẽ phải.",
    37: "Biển báo được phép đi thẳng hoặc rẽ trái.",
    38: "Biển báo yêu cầu phương tiện chỉ được đi về bên phải.",
    39: "Biển báo yêu cầu phương tiện chỉ được đi về bên trái.",
    40: "Biển báo bắt buộc đi theo vòng xuyến phía trước.",
    41: "Biển báo hết hiệu lực cấm vượt.",
    42: "Biển báo hết hiệu lực cấm vượt với xe trên 3.5 tấn."
}

penalties = {
    0: "Phạt tiền từ 800.000đ đến 1.000.000đ nếu vượt quá tốc độ tối đa từ 5 đến dưới 10 km/h.",
    1: "Phạt từ 800.000đ đến 1.000.000đ nếu vượt tốc độ cho phép.",
    2: "Phạt từ 800.000đ đến 1.000.000đ nếu vượt tốc độ cho phép.",
    3: "Phạt từ 800.000đ đến 1.000.000đ nếu vượt tốc độ cho phép.",
    4: "Phạt từ 800.000đ đến 1.000.000đ nếu vượt tốc độ cho phép.",
    5: "Phạt từ 800.000đ đến 1.000.000đ nếu vượt tốc độ cho phép.",
    6: "Không áp dụng xử phạt (biển báo đã hết hiệu lực).",
    7: "Phạt từ 3.000.000đ đến 5.000.000đ nếu vượt quá tốc độ trên 20 km/h.",
    8: "Phạt từ 4.000.000đ đến 6.000.000đ nếu vượt quá tốc độ quy định.",
    9: "Phạt từ 4.000.000đ đến 5.000.000đ nếu vượt trong khu vực cấm vượt.",
    10: "Phạt đến 6.000.000đ nếu xe tải vượt sai quy định.",
    11: "Không tuân theo biển: Cảnh cáo hoặc phạt đến 500.000đ.",
    12: "Không tuân theo biển báo đường ưu tiên: Phạt từ 200.000đ đến 400.000đ.",
    13: "Không nhường đường: Phạt từ 400.000đ đến 600.000đ.",
    14: "Không dừng đúng quy định: Phạt từ 2.000.000đ đến 3.000.000đ.",
    15: "Đi vào đường cấm: Phạt từ 3.000.000đ đến 5.000.000đ.",
    16: "Đi vào đường cấm đối với xe tải: Phạt từ 4.000.000đ đến 6.000.000đ.",
    17: "Cấm vào, vẫn đi: Phạt từ 3.000.000đ đến 5.000.000đ.",
    18: "Không giảm tốc trong khu vực nguy hiểm: Cảnh cáo hoặc phạt đến 500.000đ.",
    19: "Không giảm tốc độ tại khúc cua: Phạt từ 200.000đ đến 400.000đ.",
    20: "Không giảm tốc độ tại khúc cua: Phạt từ 200.000đ đến 400.000đ.",
    21: "Không đảm bảo an toàn tại đoạn đường cong: Phạt từ 400.000đ đến 600.000đ.",
    22: "Đi nhanh tại đoạn đường gồ ghề: Phạt từ 400.000đ đến 600.000đ.",
    23: "Không giảm tốc độ trên đường trơn trượt: Phạt từ 400.000đ đến 600.000đ.",
    24: "Không nhường đường tại đoạn đường hẹp: Phạt đến 600.000đ.",
    25: "Không tuân thủ biển thi công: Phạt từ 500.000đ đến 1.000.000đ.",
    26: "Không chấp hành tín hiệu đèn: Phạt từ 4.000.000đ đến 6.000.000đ.",
    27: "Không nhường đường cho người đi bộ: Phạt từ 800.000đ đến 1.000.000đ.",
    28: "Không giảm tốc khi có trẻ em qua đường: Phạt từ 1.000.000đ đến 2.000.000đ.",
    29: "Không giảm tốc cho xe đạp qua đường: Phạt từ 400.000đ đến 600.000đ.",
    30: "Không đảm bảo an toàn khi trời lạnh/băng tuyết: Cảnh cáo.",
    31: "Không giảm tốc khi có động vật hoang dã: Phạt từ 200.000đ.",
    32: "Không áp dụng xử phạt (biển báo hết hiệu lực).",
    33: "Không rẽ theo biển báo: Phạt từ 800.000đ đến 1.000.000đ.",
    34: "Không rẽ theo biển báo: Phạt từ 800.000đ đến 1.000.000đ.",
    35: "Không đi đúng hướng: Phạt từ 800.000đ đến 1.000.000đ.",
    36: "Không đi đúng làn, đúng hướng: Phạt từ 800.000đ đến 1.000.000đ.",
    37: "Không đi đúng làn, đúng hướng: Phạt từ 800.000đ đến 1.000.000đ.",
    38: "Không đi đúng hướng bắt buộc: Phạt từ 800.000đ đến 1.000.000đ.",
    39: "Không đi đúng hướng bắt buộc: Phạt từ 800.000đ đến 1.000.000đ.",
    40: "Không tuân thủ vòng xuyến: Phạt từ 400.000đ đến 600.000đ.",
    41: "Không áp dụng xử phạt (biển hết hiệu lực).",
    42: "Không áp dụng xử phạt (biển hết hiệu lực)."
}

# Nhóm biển báo
def get_label_group(label):
    prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
    mandatory = [33, 34, 35, 36, 37, 38, 39, 40]
    danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

    if label in prohibitory:
        return "🚫 Biển báo cấm"
    elif label in mandatory:
        return "⚠️ Biển báo bắt buộc"
    elif label in danger:
        return "⚡ Biển báo nguy hiểm"
    else:
        return "ℹ️ Biển báo khác"

# Tiền xử lý ảnh
def preprocessing(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# Giao diện Streamlit
st.set_page_config(page_title="Nhận dạng biển báo giao thông", page_icon="🚦")
st.title("🚦 Nhận dạng biển báo giao thông")

uploaded_file = st.file_uploader("📷 Tải ảnh biển báo giao thông", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Ảnh gốc", use_container_width=True)

    # Resize và xử lý ảnh
    resized_img = original_image.resize((32, 32))
    img_processed = preprocessing(resized_img).reshape(1, 32, 32, 1)

    # Dự đoán
    prediction = model.predict(img_processed)[0]
    index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    label_name = classNames.get(index, "Không xác định")
    label_group = get_label_group(index)
    label_desc = descriptions.get(index, "🚧 Chưa có mô tả cho biển báo này.")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Nhóm biển báo:")
        st.success(label_group)

        st.subheader("📘 Tên biển báo:")
        st.info(label_name)

        st.subheader("ℹ️ Mô tả chi tiết:")
        st.write(label_desc)

        st.subheader("💸 Mức xử phạt:")
        penalty = penalties.get(index, "Không có thông tin xử phạt cho biển báo này.")
        st.warning(penalty)

    with col2:
        st.subheader("📊 Mức độ tự tin:")
        st.metric("Độ chính xác", f"{confidence*100:.2f}%")

        # Biểu đồ xác suất
        top_preds = np.argsort(prediction)[::-1][:5]
        fig, ax = plt.subplots()
        ax.barh([classNames[i] for i in top_preds], prediction[top_preds])
        ax.invert_yaxis()
        ax.set_xlabel("Xác suất")
        st.pyplot(fig)
