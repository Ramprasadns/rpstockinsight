
# RPStockInsight (Streamlit Cloud Ready)

This is a ready-to-deploy Streamlit app for RPStockInsight.

## How to deploy on Streamlit Cloud (no coding required)

1. Create a GitHub repo and upload the contents of this project.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click 'New app', select your repo, branch and `app.py` as the file to run.
4. Deploy — Streamlit Cloud will install dependencies from `requirements.txt` and give you a public URL.

## Local testing

In the project folder run:
```
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- User accounts are stored in `users.json` (hashed passwords).
- This setup uses simple linear regression for predictions (fast, lightweight).
- For production, replace file storage with a proper database and secure auth.
