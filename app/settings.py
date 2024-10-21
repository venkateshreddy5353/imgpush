import os

IMAGES_DIR = "/images/"
CACHE_DIR = "/cache/"
DEBUG = False
OUTPUT_TYPE = None
MAX_UPLOADS_PER_DAY = 1000
MAX_UPLOADS_PER_HOUR = 100
MAX_UPLOADS_PER_MINUTE = 20
ALLOWED_ORIGINS = ["*"]
NAME_STRATEGY = "randomstr"
MAX_TMP_FILE_AGE = 5 * 60
RESIZE_TIMEOUT = 5
NUDE_FILTER_MAX_THRESHOLD = None
ALLOW_VIDEO = True
USERNAME="admin"
PASSWORD="Changeme@123"
ALLOWED_FILETYPES=["jpeg","jpg","png","gif","doc","docx","pdf","txt","csv","ppt","pptx","xls","xlsx","zip","rar","js","html","css","xml","json","mp4","mp3","avi","mov"]

VALID_SIZES = []

MAX_SIZE_MB = 16

for variable in [item for item in globals() if not item.startswith("__")]:
    NULL = "NULL"
    env_var = os.getenv(variable, NULL).strip()
    if env_var is not NULL:
        try:
            env_var = eval(env_var)
        except Exception:
            pass
    globals()[variable] = env_var if env_var is not NULL else globals()[variable]
