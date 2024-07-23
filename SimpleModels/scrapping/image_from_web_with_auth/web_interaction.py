import json
import requests
import base64
from requests.auth import HTTPBasicAuth


# auth = HTTPBasicAuth('891pdrekc@mozmail.com', 'Cq7&Np3Npts*47E')


# response = session.get('http://' + hostname + '/rest/applications')


def make_post(word):
    # 1. Setting parameters for the POST request
    url = 'https://api.fusionbrain.ai/web/api/v1/text2image/run?model_id=1'
    # "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
    headers ={
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-GB,en;q=0.5",
            "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJUeUFOUUV0TkFsZ0pjaWNfcE01ZTBwV3pWbXUyNG1zV0dLa2h1OXZpbzFFIn0.eyJleHAiOjE2OTk4ODkwNTYsImlhdCI6MTY5OTg4NTQ1NiwiYXV0aF90aW1lIjoxNjk5ODg1NDUzLCJqdGkiOiI5OTQ4OTIzZi04YWRiLTQyZWYtYTFmNS1jZTE1YTllMzY1OTQiLCJpc3MiOiJodHRwczovL2F1dGguZnVzaW9uYnJhaW4uYWkvcmVhbG1zL0ZCIiwiYXVkIjoiYWNjb3VudCIsInN1YiI6ImU0MzA5NWU5LTQ0YWMtNDQxYi1hMWQ4LTQ0NTljYjEwMzZlYSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImZ1c2lvbi13ZWIiLCJzZXNzaW9uX3N0YXRlIjoiYTRjZGEyOTgtYTQ0My00YWMzLWIxZmUtYzIwZDRiOTU5ODg5IiwiYWNyIjoiMCIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsImRlZmF1bHQtcm9sZXMtZmIiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwiLCJzaWQiOiJhNGNkYTI5OC1hNDQzLTRhYzMtYjFmZS1jMjBkNGI5NTk4ODkiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicHJlZmVycmVkX3VzZXJuYW1lIjoiODkxcGRyZWtjQG1vem1haWwuY29tIiwiZW1haWwiOiI4OTFwZHJla2NAbW96bWFpbC5jb20ifQ.nZxyKKfWQjMrmqAFHRLy4UlHxe8uCOANu5k8WIEMlK2yr26nEct_3iavg_U0AVbm3vIt9LoHSMmNEeUmZ37KEDhvfcwrQwjD8p9M2IxVNuwQ0IVZ2jcjAId3JzYgT6BHhrcRh2Ot_49VsprWYTmwghuPYbX5al_qD0tnTKfC6p0zWLHO9ykV8ybvjrAPp7f2B7--tgrGiVKbLlY7-69bIjmc1jTiW5lg6sUTUpR3nMNRKbcNEznZljkzWVzvWAuscDDFenIhxTIPVdEhtap9OXRvjWNnPbi25gxkglItq_zoCWLYx6jRf4q6mDmtC5zG2NaMG7D2y-tU3gVdMY7t2Q",
            "Content-Type": "multipart/form-data; boundary=---------------------------155928840523742471961848431689",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Sec-GPC": "1"
    }
    data = '{"credentials": "omit", "referrer": "https://editor.fusionbrain.ai/", "body": "-----------------------------155928840523742471961848431689\r\nContent-Disposition: form-data; name=\"params\"; filename=\"blob\"\r\nContent-Type: application/json\r\n\r\n{\"type\":\"GENERATE\",\"width\":1024,\"height\":1024,\"generateParams\":{\"query\":\"' + word + '\"}}\r\n-----------------------------155928840523742471961848431689--\r\n","method": "POST","mode": "cors"}'
    data = data.encode('utf-8')
    # 2. Sending the POST request
    response = requests.post(url, headers=headers, data=data)
    # session = requests.Session()
    # session.auth = ('891pdrekc@mozmail.com', 'Cq7&Np3Npts*47E')
    # auth = session.post('http://' + 'https://api.fusionbrain.ai/')
    # response = session.post(url, headers=headers, data=data)
    # 3. Handling the response from the server
    print(requests.codes.ok)
    if response.status_code > 201:
        print(response.status_code)
        return None
    return response.json()["uuid"]


def make_get(uuid):
    # 1. Setting parameters for the POST request
    url = 'https://api.fusionbrain.ai/web/api/v1/text2image/status/' + uuid
    # "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-GB,en;q=0.5",
        "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJUeUFOUUV0TkFsZ0pjaWNfcE01ZTBwV3pWbXUyNG1zV0dLa2h1OXZpbzFFIn0.eyJleHAiOjE2OTk4ODkwNTYsImlhdCI6MTY5OTg4NTQ1NiwiYXV0aF90aW1lIjoxNjk5ODg1NDUzLCJqdGkiOiI5OTQ4OTIzZi04YWRiLTQyZWYtYTFmNS1jZTE1YTllMzY1OTQiLCJpc3MiOiJodHRwczovL2F1dGguZnVzaW9uYnJhaW4uYWkvcmVhbG1zL0ZCIiwiYXVkIjoiYWNjb3VudCIsInN1YiI6ImU0MzA5NWU5LTQ0YWMtNDQxYi1hMWQ4LTQ0NTljYjEwMzZlYSIsInR5cCI6IkJlYXJlciIsImF6cCI6ImZ1c2lvbi13ZWIiLCJzZXNzaW9uX3N0YXRlIjoiYTRjZGEyOTgtYTQ0My00YWMzLWIxZmUtYzIwZDRiOTU5ODg5IiwiYWNyIjoiMCIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsImRlZmF1bHQtcm9sZXMtZmIiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwiLCJzaWQiOiJhNGNkYTI5OC1hNDQzLTRhYzMtYjFmZS1jMjBkNGI5NTk4ODkiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicHJlZmVycmVkX3VzZXJuYW1lIjoiODkxcGRyZWtjQG1vem1haWwuY29tIiwiZW1haWwiOiI4OTFwZHJla2NAbW96bWFpbC5jb20ifQ.nZxyKKfWQjMrmqAFHRLy4UlHxe8uCOANu5k8WIEMlK2yr26nEct_3iavg_U0AVbm3vIt9LoHSMmNEeUmZ37KEDhvfcwrQwjD8p9M2IxVNuwQ0IVZ2jcjAId3JzYgT6BHhrcRh2Ot_49VsprWYTmwghuPYbX5al_qD0tnTKfC6p0zWLHO9ykV8ybvjrAPp7f2B7--tgrGiVKbLlY7-69bIjmc1jTiW5lg6sUTUpR3nMNRKbcNEznZljkzWVzvWAuscDDFenIhxTIPVdEhtap9OXRvjWNnPbi25gxkglItq_zoCWLYx6jRf4q6mDmtC5zG2NaMG7D2y-tU3gVdMY7t2Q",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "Sec-GPC": "1"
    }
    data = '{"credentials": "omit", "referrer": "https://editor.fusionbrain.ai/","method": "GET","mode": "cors"}'
    # 2. Sending the POST request
    response = requests.get(url, headers=headers, data=data)
    # 3. Handling the response from the server
    # print(requests.codes.ok)
    if response.status_code > 200:
        print(response.status_code)
        return None
    return response.json()


def decode_base64_to_image(image_base64: str):
    imgdata = base64.b64decode(image_base64)
    return imgdata


def download_image_by_text(text: str, filename: str):
    uuid = make_post(text)
    response_jsn = make_get(uuid)
    while response_jsn["images"] is None:
        response_jsn = make_get(uuid)
    image_base64 = response_jsn["images"][0]
    imagedata = decode_base64_to_image(image_base64)
    with open(filename, 'wb') as f:
        f.write(imagedata)

