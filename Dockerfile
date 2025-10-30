FROM astrocrpublic.azurecr.io/runtime:3.1-2
#Copy requirements and install additional packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

