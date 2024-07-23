import json
import requests
import base64


def make_post(word):
    # 1. Setting parameters for the POST request
    url = 'https://api.fusionbrain.ai/web/api/v1/text2image/run?model_id=1'
    # "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0",
    headers ={
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-GB,en;q=0.5",
            "Content-Type": "multipart/form-data; boundary=---------------------------204776833312792758952050705153",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Sec-GPC": "1"
    }
    data = '{"credentials": "omit", "referrer": "https://editor.fusionbrain.ai/", "body": "-----------------------------204776833312792758952050705153\r\nContent-Disposition: form-data; name=\"params\"; filename=\"blob\"\r\nContent-Type: application/json\r\n\r\n{\"type\":\"GENERATE\",\"width\":1024,\"height\":1024,\"generateParams\":{\"query\":\"' + word + '\"}}\r\n-----------------------------204776833312792758952050705153--\r\n","method": "POST","mode": "cors"}'
    data = data.encode('utf-8')
    # 2. Sending the POST request
    response = requests.post(url, headers=headers, data=data)
    # 3. Handling the response from the server
    # print(requests.codes.ok)
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

