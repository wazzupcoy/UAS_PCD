from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        file = request.files['image']
        operator = request.form['operator']

        if file:
            before_path = os.path.join(UPLOAD_FOLDER, 'before.png')
            after_path  = os.path.join(UPLOAD_FOLDER, 'after.png')
            file.save(before_path)

            # === BACA CITRA (MODUL) ===
            img = cv2.imread(before_path)

            # === GRAYSCALE ===
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # === THRESHOLD (KHUSUS REGION FILLING) ===
            ret, thresh = cv2.threshold(gray, 70, 255, 0)

            # === KERNEL SESUAI MODUL ===
            kernel = np.ones((5,5), np.uint8)

            # === OPERATOR ===
            if operator == 'dilation':
                processed = cv2.dilate(img, kernel, iterations=1)

            elif operator == 'erosion':
                processed = cv2.erode(img, kernel, iterations=1)

            elif operator == 'opening':
                processed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            elif operator == 'closing':
                processed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            elif operator == 'filling':
                im_floodfill = thresh.copy()
                h, w = thresh.shape
                mask = np.zeros((h+2, w+2), np.uint8)

                cv2.floodFill(im_floodfill, mask, (0,0), 255)
                im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                processed = thresh | im_floodfill_inv

            # === SIMPAN HASIL ===
            cv2.imwrite(after_path, processed)

            # === ANALISIS OBJEK (KHUSUS BINARY) ===
            if operator == 'filling':
                contours, _ = cv2.findContours(
                    processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50]
            else:
                areas = []

            result = {
                'operator': operator,
                'jumlah': len(areas),
                'max': int(max(areas)) if areas else 0,
                'min': int(min(areas)) if areas else 0
            }

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
