import cv2
import os
import time
from ultralytics import YOLO
import requests

# Thông tin Telegram Bot
BOT_TOKEN = ''  # Thay bằng token của bạn
CHAT_ID = ''       # Thay bằng chat ID của bạn
TELEGRAM_API_URL = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'

# Load your custom YOLOv8 model
model = YOLO('D:/aiot/models/best (2).pt')  # Đường dẫn đến model của bạn

# Create the 'detected_frames' directory if it doesn't exist
output_folder = 'usingphone'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_count = 0
last_save_time = 0
last_notify_time = 0  # Thời điểm gửi thông báo cuối cùng

def send_telegram_notification(image_path, class_name):
    """Gửi ảnh và thông báo qua Telegram"""
    with open(image_path, 'rb') as photo:
        message = f"Phát hiện: {class_name}"
        files = {'photo': photo}
        data = {
            'chat_id': CHAT_ID,
            'caption': message
        }
        response = requests.post(TELEGRAM_API_URL, files=files, data=data)
        if response.status_code == 200:
            print(f"Đã gửi thông báo Telegram: {message}")
        else:
            print(f"Lỗi gửi thông báo Telegram: {response.text}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Perform object detection
    results = model(frame)

    # Kiểm tra nếu có phát hiện đối tượng
    if len(results) > 0:
        result = results[0]
        if len(result.boxes) > 0:
            # Lấy danh sách tên lớp của các đối tượng được phát hiện
            class_ids = result.boxes.cls.cpu().numpy()
            class_names = [result.names[int(cls_id)] for cls_id in class_ids]
            annotated_frame = result.plot()  # Vẽ bounding boxes lên frame

            # Kiểm tra nếu có nhãn "Using_phone"
            if "Using_phone" in class_names:
                current_time = time.time()
                # Lưu ảnh nếu đã qua 1 giây
                if current_time - last_save_time >= 1:
                    frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
                    cv2.imwrite(frame_filename, annotated_frame)
                    print(f"Đã lưu ảnh: {frame_filename}")
                    last_save_time = current_time
                    frame_count += 1

                    # Gửi thông báo Telegram nếu đã qua 5 giây
                    if current_time - last_notify_time >= 5:
                        send_telegram_notification(frame_filename, "Using_phone")
                        last_notify_time = current_time
        else:
            annotated_frame = frame
    else:
        annotated_frame = frame

    # Hiển thị frame
    cv2.imshow('Live Stream Object Detection', annotated_frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()