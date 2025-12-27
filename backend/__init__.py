import logging
import sys
from backend.config import LOG_LEVEL

# Định nghĩa version của Backend
__version__ = "1.0.0"
__author__ = "ASR Thesis Team"

# Thiết lập cấu hình Logging cho toàn bộ hệ thống
# Log sẽ có dạng: 2023-10-25 10:00:00 [INFO] Message...
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Tạo logger gốc cho package
logger = logging.getLogger("backend")
logger.info(f"Backend initialized (Version {__version__})")
