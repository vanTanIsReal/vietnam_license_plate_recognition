from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import csv
from datetime import datetime
import threading

# === Cấu hình Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Thiết lập ứng dụng Flask ===
app = Flask(__name__)

# === Hằng số và Cấu hình ===
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
CHECK_FOLDER = 'static/check/processed'
MODEL_DIR = 'models/trocr'
MIN_PLATE_WIDTH = 50
MIN_PLATE_HEIGHT = 20
ASPECT_RATIO_THRESHOLD = 1.3
ONE_LINE_RATIO_THRESHOLD = 2.5  # Ngưỡng tỷ số w/h để xác định biển số một dòng
CSV_FILE = 'plate_recognition.csv'
CSV_WRITE_INTERVAL = 3  # giây

# Tạo các thư mục cần thiết
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, CHECK_FOLDER, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

# Khởi tạo CSV
if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Thời gian', 'Văn bản biển số', 'Độ tin cậy', 'Tọa độ'])

# === Tải mô hình ===
# Tải mô hình YOLO
try:
    model = YOLO('best.pt')
    logger.info("Mô hình YOLO được tải thành công")
except Exception as e:
    logger.error(f"Lỗi khi tải mô hình YOLO: {str(e)}")
    raise

# Tải mô hình TrOCR với bộ nhớ đệm
@torch.no_grad()
def load_trocr_model():
    model_name = "microsoft/trocr-base-printed"
    try:
        processor_path = os.path.join(MODEL_DIR, "processor")
        model_path = os.path.join(MODEL_DIR, "model")
        
        if os.path.exists(processor_path) and os.path.exists(model_path):
            processor = TrOCRProcessor.from_pretrained(processor_path)
            trocr_model = VisionEncoderDecoderModel.from_pretrained(model_path)
        else:
            processor = TrOCRProcessor.from_pretrained(model_name)
            trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
            processor.save_pretrained(processor_path)
            trocr_model.save_pretrained(model_path)
        
        # Đặt mô hình ở chế độ đánh giá và chuyển sang GPU nếu có
        trocr_model.eval()
        if torch.cuda.is_available():
            trocr_model = trocr_model.to("cuda")
            logger.info("Mô hình TrOCR được tải trên GPU")
        else:
            logger.info("Mô hình TrOCR được tải trên CPU")
        return processor, trocr_model
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình TrOCR: {str(e)}")
        raise

processor, trocr_model = load_trocr_model()

# === Biến toàn cục để điều khiển camera ===
camera = None
camera_running = False
camera_lock = threading.Lock()
last_csv_write = 0

# === Hàm trợ giúp ===
def deskew_image(image):
    """Xoay ảnh để chỉnh góc nghiêng của biển số"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        
        angle = 0
        if lines is not None:
            angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for x1, y1, x2, y2 in lines[:, 0]]
            angles = [a for a in angles if abs(a) < 30]  # Lọc các góc quá lớn
            angle = np.median(angles) if angles else 0  # Sử dụng trung vị để tăng độ ổn định
        
        if abs(angle) > 2:  # Chỉ xoay nếu góc đáng kể
            (h, w) = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        return image
    except Exception as e:
        logger.warning(f"Lỗi khi chỉnh nghiêng ảnh: {str(e)}")
        return image

# Nhận dạng OCR hàng loạt với dự đoán mô hình sử dụng GPU nếu có
@torch.no_grad()
def ocr_with_trocr_batch(images, max_batch_size=8):
    """Nhận dạng văn bản từ một lô ảnh"""
    try:
        results = []
        # Xử lý theo các lô nhỏ hơn để tránh vấn đề bộ nhớ
        for i in range(0, len(images), max_batch_size):
            batch = images[i:i+max_batch_size]
            pil_images = [Image.fromarray(img).convert("RGB") for img in batch]
            
            # Xử lý ảnh
            inputs = processor(pil_images, return_tensors="pt")
            pixel_values = inputs.pixel_values
            
            # Chuyển sang GPU nếu có
            if torch.cuda.is_available():
                pixel_values = pixel_values.to("cuda")
            
            # Tạo văn bản
            generated_ids = trocr_model.generate(
                pixel_values, 
                max_length=20,
                num_beams=3,
                early_stopping=True
            )
            
            # Giải mã
            batch_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            results.extend([text.strip().upper() for text in batch_texts])
            
        return results
    except Exception as e:
        logger.error(f"Lỗi OCR hàng loạt: {str(e)}")
        return ["Không nhận dạng" for _ in images]

def split_plate_two_lines(plate_img):
    """Tách biển số hai dòng thành hai ảnh riêng biệt"""
    height = plate_img.shape[0]
    mid = height // 2
    
    # Tìm điểm chia tốt hơn bằng phép chiếu ngang
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img
        
    # Tìm khoảng cách giữa các dòng bằng phép chiếu ngang
    projection = np.sum(gray, axis=1)
    for i in range(mid-10, mid+10):
        if i > 0 and i < height-1:
            if projection[i] < 0.7 * max(projection):
                mid = i
                break
                
    return plate_img[:mid, :], plate_img[mid:, :]

def process_two_line_plate(plate_img, filename, x1, y1):
    """Xử lý biển số hai dòng"""
    line1_img, line2_img = split_plate_two_lines(plate_img)
    base_name = f"{filename}_{x1}_{y1}"
    
    # Lưu ảnh gốc để gỡ lỗi
    line1_path = os.path.join(CHECK_FOLDER, f"{base_name}_line1.jpg")
    line2_path = os.path.join(CHECK_FOLDER, f"{base_name}_line2.jpg")
    cv2.imwrite(line1_path, line1_img)
    cv2.imwrite(line2_path, line2_img)
    
    # Nhận dạng văn bản từng dòng
    texts = ocr_with_trocr_batch([line1_img, line2_img])
    combined_text = f"{texts[0]} {texts[1]}".strip()
    
    return {
        'text': combined_text,
        'processed_images': [
            f'/static/check/processed/{os.path.basename(line1_path)}',
            f'/static/check/processed/{os.path.basename(line2_path)}'
        ]
    }

def process_single_line_plate(plate_img, filename, x1, y1):
    """Xử lý biển số một dòng"""
    save_path = os.path.join(CHECK_FOLDER, f"{filename}_{x1}_{y1}.jpg")
    cv2.imwrite(save_path, plate_img)
    text = ocr_with_trocr_batch([plate_img])[0]
    
    return {
        'text': text,
        'processed_images': [f'/static/check/processed/{os.path.basename(save_path)}']
    }

def process_single_plate(plate_img, aspect_ratio, filename, x1, y1, x2, y2, conf):
    """Xử lý một biển số duy nhất"""
    try:
        # Chỉnh nghiêng nếu cần
        if aspect_ratio > ASPECT_RATIO_THRESHOLD:
            plate_img = deskew_image(plate_img)
            
        # Xác định biển số một dòng hay hai dòng dựa trên tỷ số w/h
        h, w = plate_img.shape[:2]
        aspect_ratio = w / h  # Tỷ số chiều rộng/chiều cao
        is_two_line = aspect_ratio <= ONE_LINE_RATIO_THRESHOLD  # w/h <= 2.5 là biển số hai dòng
        logger.info(f"Tỷ số w/h: {aspect_ratio:.2f}, Hai dòng: {is_two_line}, Tọa độ: [{x1}, {y1}, {x2}, {y2}]")
        
        # Xử lý tương ứng
        if is_two_line:
            # Tách và xử lý biển số hai dòng
            result = process_two_line_plate(plate_img, filename, x1, y1)
        else:
            # Xử lý biển số một dòng
            result = process_single_line_plate(plate_img, filename, x1, y1)
            
        # Xác thực kết quả
        if not result['text'] or len(result['text']) < 4:
            result['text'] = "Không nhận dạng"
            
        return {
            'text': result['text'],
            'confidence': float(conf),
            'coordinates': [x1, y1, x2, y2],
            'processed_images': result['processed_images']
        }
    except Exception as e:
        logger.error(f"Lỗi khi xử lý biển số tại [{x1}, {y1}, {x2}, {y2}]: {str(e)}")
        return {
            'text': "Không nhận dạng",
            'confidence': float(conf),
            'coordinates': [x1, y1, x2, y2],
            'processed_images': []
        }

def write_to_csv(plates):
    """Ghi kết quả nhận dạng vào tệp CSV"""
    if not plates:
        return
        
    try:
        with open(CSV_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            for plate in plates:
                writer.writerow([
                    timestamp, 
                    plate['text'], 
                    plate['confidence'], 
                    plate['coordinates']
                ])
    except Exception as e:
        logger.error(f"Lỗi ghi CSV: {str(e)}")

# === Hàm xử lý video ===
def extract_plates_from_frame(frame):
    """Trích xuất biển số từ một khung hình video"""
    results = model(frame, conf=0.35)
    plates_data = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Lọc theo kích thước
            if box_width < MIN_PLATE_WIDTH or box_height < MIN_PLATE_HEIGHT:
                continue
                
            aspect_ratio = box_width / box_height
            plate_img = frame[y1:y2, x1:x2]
            plates_data.append((plate_img, aspect_ratio, 'camera', x1, y1, x2, y2, box.conf))
    
    return plates_data

def process_plates_batch(plates_data):
    """Xử lý một lô biển số song song"""
    if not plates_data:
        return []
        
    with ThreadPoolExecutor(max_workers=min(len(plates_data), 4)) as executor:
        plates = list(executor.map(
            lambda x: process_single_plate(*x),
            [(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]) for data in plates_data]
        ))
    
    return plates

def draw_results_on_frame(frame, plates):
    """Vẽ hộp giới hạn và kết quả nhận dạng trên khung hình"""
    for plate in plates:
        x1, y1, x2, y2 = plate['coordinates']
        text = plate['text']
        conf = plate['confidence']
        
        # Vẽ hộp
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tạo nền cho văn bản
        text_size, _ = cv2.getTextSize(f"{text} ({conf:.2f})", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
        
        # Vẽ văn bản
        cv2.putText(frame, f"{text} ({conf:.2f})", 
                  (x1, y1 - 5), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def generate_frames():
    """Tạo các khung hình video với phát hiện biển số"""
    global camera, camera_running, last_csv_write
    
    if camera is None or not camera.isOpened():
        logger.error("Camera không khả dụng hoặc chưa được khởi tạo")
        return
    
    camera_running = True
    last_csv_write = time.time()
    frame_skip = 0  # Xử lý mỗi khung hình thứ N
    
    while camera_running:
        with camera_lock:
            success, frame = camera.read()
            
        if not success:
            logger.error("Không thể đọc khung hình từ camera")
            time.sleep(0.1)
            continue
        
        # Chỉ xử lý mỗi khung hình thứ N để giảm tải CPU
        frame_skip = (frame_skip + 1) % 2
        if frame_skip == 0:
            # Trích xuất biển số
            plates_data = extract_plates_from_frame(frame)
            
            # Xử lý biển số
            plates = process_plates_batch(plates_data)
            
            # Vẽ kết quả trên khung hình
            frame = draw_results_on_frame(frame, plates)
            
            # Ghi vào CSV định kỳ
            current_time = time.time()
            if plates and current_time - last_csv_write >= CSV_WRITE_INTERVAL:
                write_to_csv(plates)
                last_csv_write = current_time
        
        # Chuyển đổi khung hình sang JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# === Các tuyến Flask ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có tệp được tải lên'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Không có tệp được chọn'}), 400
    
    try:
        # Lưu và xử lý ảnh được tải lên
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        
        # Đọc ảnh và xác thực
        img = cv2.imread(upload_path)
        if img is None:
            return jsonify({'error': 'Tệp ảnh không hợp lệ'}), 400
        
        # Phát hiện biển số
        results = model(img)
        result_img = results[0].plot()
        result_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(result_path, result_img)
        
        # Trích xuất vùng biển số
        plates_data = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Lọc theo kích thước
                if box_width < MIN_PLATE_WIDTH or box_height < MIN_PLATE_HEIGHT:
                    continue
                    
                aspect_ratio = box_width / box_height
                plate_img = img[y1:y2, x1:x2]
                plates_data.append((plate_img, aspect_ratio, filename, x1, y1, x2, y2, box.conf))
        
        # Xử lý biển số song song
        plates = process_plates_batch(plates_data)
        
        # Lưu kết quả vào CSV
        if plates:
            write_to_csv(plates)
        
        return jsonify({
            'status': 'thành công',
            'original_image': f'/static/uploads/{filename}',
            'result_image': f'/static/results/{filename}',
            'plates': plates,
            'plate_count': len(plates)
        })
    
    except Exception as e:
        logger.error(f"Lỗi xử lý: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_camera')
def start_camera():
    global camera, camera_running
    
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({'error': 'Không thể mở camera'}), 500
                
            # Thiết lập thuộc tính camera để cải thiện hiệu suất
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 15)
            
        camera_running = True
        
    return jsonify({'status': 'Camera đã khởi động'})

@app.route('/stop_camera')
def stop_camera():
    global camera, camera_running
    
    with camera_lock:
        camera_running = False
        if camera is None:
            camera.release()
            camera = None
            
    return jsonify({'status': 'Camera đã dừng'})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)