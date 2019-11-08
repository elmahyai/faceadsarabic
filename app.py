from flask import Flask, render_template, jsonify, request
from PIL import Image
import base64
import io
# facerec.py
#import cv2

from face_recognize import predictor
global p
p = predictor()

app = Flask(__name__, static_folder='static')

@app.route('/')
def index(name=None):
    return render_template('index.html',name=name)


@app.route('/snap_a_signal', methods=["POST", "GET"])
def process_signal():
    pixels = request.get_json()['data']
    selected_model = request.get_json()['selected_model']
    results, accuracy=im2info(pixels, selected_model)
    if results:
        return jsonify(results=results,
                       accuracy=accuracy)


def im2info(pixels, modelname):
    image_data = base64.b64decode(pixels.split(",")[1])
    image = Image.open(io.BytesIO(image_data))
    background = image.convert('RGB')
    if modelname == 'gender':
        return p.predict_gender(background)
    elif modelname == 'look':
        return  p.predict_look(background)
    elif modelname == 'chubby':
        return p.predict_chubby(background)
    elif modelname == 'glass':
        return p.predict_glass(background)
    elif modelname == 'Receding_Hairline':
        return p.predict_Receding_Hairline(background)
    elif modelname == 'Bags_Under_Eyes':
        return p.predict_Bags_Under_Eyes(background)
    elif modelname == 'Bald':
        return p.predict_Bald(background)
    elif modelname == 'Young':
        return p.predict_Young(background)
    elif modelname == 'Pale_Skin':
        return p.predict_Pale_Skin(background)

    
        
        
    
    
if __name__ == '__main__':
    app.run()
    app.debug = True