from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def page1():
    return render_template('page1.html')

@app.route('/book')
def page2():
    return render_template('page2.html')

@app.route('/chat')
def page3():
    return render_template('page3.html')

@app.route('/goto/<target>')
def goto(target):
    if target == 'book':
        return redirect(url_for('page2'))
    elif target == 'chat':
        return redirect(url_for('page3'))
    else:
        return redirect(url_for('page1'))

if __name__ == '__main__':
    app.run(debug=True)
