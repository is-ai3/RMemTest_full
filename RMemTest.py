import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # ★追加：MobileNetV2専用の前処理

import numpy as np


classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]  # ★変更：0〜9 → 5種類の花の名前
image_size = 224  # ★変更：28 → 224（MobileNetV2は224x224が必須）

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

# app.secret_key = "your_secret_key_here"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model_full.keras')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # ★変更：color_mode='grayscale' を削除（花はカラー画像）
            img = image.load_img(filepath, target_size=(image_size, image_size))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)       # ★変更：np.array([img]) → np.expand_dims（同じ意味だが明示的）
            data = preprocess_input(img)             # ★追加：MobileNetV2専用の前処理（/255.0の代わり）

            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)



#if __name__ == "__main__":
#    app.run()


#変更箇所は4箇所です。
#classes を花の名前5つに変更しました。順番はColabで学習した際のアルファベット順と必ず合わせてください。
#image_size を28→224に変更しました。MobileNetV2の必須サイズです。
#color_mode='grayscale' を削除しました。花はカラー画像なのでグレースケール変換は不要です。
#preprocess_input() を追加しました。MobileNetV2専用の前処理で、これを忘れると精度が大きく落ちます。