from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import os
import io
from ultralytics import YOLO
from flask_cors import CORS

# 創建 Flask 應用
app = Flask(__name__)

# 啟用 CORS 支援，允許所有來源訪問
CORS(app, resources={r"/*": {"origins": "*"}})

# 設定文件上傳的資料夾
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
RESULTS_FOLDER = os.path.join(os.getcwd(), 'static', 'results')

# 創建資料夾，如果不存在的話
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# 載入 YOLO 模型
model = YOLO('../models/v1.pt')

# 設置模型為評估模式
model.eval()

@app.route('/')
def home():
    return "PCB Defect Detection with YOLOv8!"

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # 儲存原始圖片到本地
        original_image_path = os.path.join(UPLOAD_FOLDER, 'original_image.jpg')
        file.save(original_image_path)

        # 使用 YOLO 進行檢測
        img = Image.open(original_image_path)
        results = model(img)
        output_image = results[0].plot()  # 渲染結果
        output_image = Image.fromarray(output_image)
        
        # 儲存檢測結果圖片
        output_image_path = os.path.join(RESULTS_FOLDER, 'result_image.jpg')
        output_image.save(output_image_path)

        # 返回圖片 URL
        image_url = request.url_root.rstrip('/') + '/static/results/result_image.jpg'
        return jsonify({'image_url': image_url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/results/<filename>')
def download_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
