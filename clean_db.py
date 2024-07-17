import os
from datetime import datetime, timedelta

DB = 'access.txt'
EXPIRATION_HOURS = 4

def clean_old_credentials():
    if not os.path.exists(DB):
        return

    current_time = datetime.now()
    valid_credentials = []

    with open(DB, 'r') as f:
        lines = f.read().splitlines()

    for line in lines:
        credential, timestamp = line.split()
        creation_time = datetime.fromisoformat(timestamp)
        if current_time - creation_time < timedelta(hours=EXPIRATION_HOURS):
            valid_credentials.append(line)

    with open(DB, 'w') as f:
        for valid_credential in valid_credentials:
            f.write(valid_credential + '\n')

if __name__ == "__main__":
    clean_old_credentials()
