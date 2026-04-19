from flask import Flask, render_template, request
import os
from detector import analyze_document

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

ALLOWED = {'png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        file = request.files.get('document')
        if not file or file.filename == '':
            error = "Please select a file!"
        elif not allowed_file(file.filename):
            error = "Only JPG, PNG, PDF files allowed!"
        else:
            path = os.path.join('uploads', file.filename)
            file.save(path)
            result = analyze_document(path)
    return render_template('index.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)