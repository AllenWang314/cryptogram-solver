from flask import Blueprint, render_template, abort, request
from backend.algorithms.encode import encode, clean_text
from backend.algorithms.decode import decode_web_version

index_bp = Blueprint('index_bp', __name__, template_folder='templates')

@index_bp.route('/assignment')
def assignment():
    return render_template("assignment.html")

@index_bp.route('/help')
def help():
    return render_template("help.html")

@index_bp.route('/report')
def report():
    return render_template("report.html")

@index_bp.route('/')
def solver():
    return render_template("solver.html", 
        input_plaintext = "what is the point of worrying oneself too much about what one could or could not have done to control the course ones life took. surely it is enough that the likes of you and i at least try to make our small contribution count for something true and worthy. and if some of us are prepared to sacrifice much in life in order to pursue such aspirations surely that in itself whatever the outcome cause for pride and contentment. written by kazuo ishiguro in the remains of the day")

@index_bp.route('/encrypt')
def encrypt():
    plaintext = request.args.get('input_plaintext')
    cleaned_plaintext = clean_text(plaintext)
    encrypted_text = encode(cleaned_plaintext)
    return render_template("solver.html", 
        input_plaintext = cleaned_plaintext, 
        generated_ciphertext = encrypted_text)

@index_bp.route('/decrypt')
def decrypt():
    ciphertext = request.args.get('input_ciphertext')
    print(ciphertext)
    plaintext = decode_web_version(ciphertext)
    print(plaintext)
    return render_template("solver.html", 
        input_ciphertext = ciphertext, 
        generated_plaintext = plaintext)

