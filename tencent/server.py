# -*- coding:utf-8 -*-
#https://blog.csdn.net/qq_26669719/article/details/80817819
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/asr', methods=['POST'])
def register():
    # print(request.headers)
    # print(request.form)
    print(request.form['requestId'])
    print(request.form.get('text'))
    # print(request.form.getlist('name'))
    # print(request.form.get('nickname', default='little apple'))
    #do something else
    #
    #
    return 'welcome'

if __name__ == '__main__':
    app.run(port=8081,debug=True)