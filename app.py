from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
import os
from datetime import datetime
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# مدل YOLOv11n رو لود کن
model = YOLO("yolo11n.pt")  # مطمئن شو تو محیطت YOLOv11 رو نصب کردی

@app.route('/', methods=['GET', 'POST'])
def index():
    output_image = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            # ذخیره عکس آپلود شده
            filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.jpg'
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            # پردازش YOLO
            results = model(path)
            results[0].save(output_dir=app.config['UPLOAD_FOLDER'], exist_ok=True)

            # YOLO خروجی رو تو همون فولدر ذخیره میکنه
            output_image = filename

    return render_template('index.html', output_image=output_image)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
