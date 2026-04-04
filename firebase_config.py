import firebase_admin
from firebase_admin import credentials, db

def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://faircheck-ai-default-rtdb.firebaseio.com/'
        })

def push_audit_data(result):
    ref = db.reference('audit_history')
    ref.push(result)

def get_audit_history():
    ref = db.reference('audit_history')
    return ref.get()