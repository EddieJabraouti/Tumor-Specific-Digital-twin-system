import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time 
import requests
import json
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


if __name__== "__main__": 
    print("Starting application")
    app.run(host="0.0.0.0", port=8000, debug=True)