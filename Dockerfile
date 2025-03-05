FROM supervisely/base-py-sdk:6.73.322

COPY dev_requirements.txt .
COPY Performance_Test.tar /tmp

RUN pip install --no-cache-dir -r dev_requirements.txt