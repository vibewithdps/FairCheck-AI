import firebase_admin
import streamlit as st
from firebase_admin import credentials, db

def initialize_firebase():
    if not firebase_admin._apps:
        # Streamlit automatically handles the PEM \n issues when reading from secrets
        key_dict = dict(st.secrets["firebase"])
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://faircheck-ai-default-rtdb.firebaseio.com/'
        })