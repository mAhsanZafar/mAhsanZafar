import chardet
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        logging.info(f"Detected encoding for {file_path}: {encoding}")
        return encoding
    except Exception as e:
        logging.error(f"Error detecting encoding for {file_path}: {e}")
        return None



file_path = 'E:\\BAI\\save_directory\\model.csv'
encoding = detect_encoding(file_path)
if encoding:
    print(f"Detected encoding: {encoding} of {file_path}")
else:
    print("Failed to detect encoding.")
