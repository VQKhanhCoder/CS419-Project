import os
import subprocess

if __name__ == "__main__":
    web_path = os.path.join(os.path.dirname(__file__), "web", "web.py")
    subprocess.run(["streamlit", "run", web_path])
