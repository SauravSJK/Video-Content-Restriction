import requests
r = requests.post("http://192.168.31.92:5000/", files = {'image' : open('1f0.jpg', 'rb')})
print r.json()['age'], r.json()['gender']