from flask import Flask, request, jsonify
import subprocess
import base64
from PIL import Image
import io

app = Flask(__name__)

@app.route('/submit', methods=['POST'])
def submit_points():
    data = request.json
    points = data['points']
    # 将点信息传递给C++程序
    result = subprocess.run(['./process_points', str(points)], capture_output=True)
    
    # 假设C++程序输出结果图像路径
    result_image_path = result.stdout.decode().strip()
    
    # 读取结果图像并编码为base64
    with open(result_image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    
    return jsonify({'result_image': encoded_image})

if __name__ == '__main__':
    app.run(port=5000)