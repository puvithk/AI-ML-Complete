from flask import Flask
app = Flask(__name__)

@app.route('/hello_world' , methods=['GET'])
def hello_world():
    return "Hello world"

if __name__ =='__main__':

    app.run()