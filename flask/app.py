from flask import Flask, jsonify, request, render_template
# from flask_jsonify import jsonify
import json, requests, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
# from ingredient_parser import ingredient_parser
import final
from final import rec_sys




app = Flask(__name__)

@app.route("/hello", methods=["GET"])
    # form= RegistrationForm()
def hello():
    return render_template('hello.html', title='HI')



@app.route('/recipe', methods=["GET"])
def recipe():
    ingredients = request.args.get('ingredients')   
    recipe = final.rec_sys(ingredients)
    
    response = {}
    count = 0
    for index, row in recipe.iterrows():
        response[count] = {
            'recipe': str(row['recipe']),
            'score': str(row['score']),
            'ingredients': str(row['ingredients']),
            'url': str(row['url'])
        }
        count += 1
    return jsonify(response)
   
if __name__== "__main__":
    app.debug=True
    app.run()



# http://127.0.0.1:5000/recipe?ingredients=pasta

# use ipconfig getifaddr en0 in terminal (ipconfig if you are on windows, ip a if on linux) 
# to find intenal (LAN) IP address. Then on any devide on network you can use server.
    

#     return render_template('about.html')

# @app.route("/register", methods= ['GET', 'POST'])
# def register():
#     form= RegistrationForm()
#     return render_template('register.html', title='Register', form=form)

# @app.route("/login")
# def login():
#     form=LoginForm()
#     return render_template('login.html', title= 'Login', form=form)



# if __name__=='__main__':
#     app.run(debug=True)
    