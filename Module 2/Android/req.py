import requests
r = requests.post("http://18.191.193.32:5000/", files = {'image' : open('1f0.jpg', 'rb')})
print r.json()['age'], r.json()['gender']