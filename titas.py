import urllib.request

url = 'https://www.youtube.com/watch?v=1Mku90VzJ0s'
req = urllib.request.Request(url)
req.add_header('User-Agent', 'Chrome/128.0.6613.120 (Windows 11; Win64; x64; rv:68.0)')
req.add_header('Range', 'bytes=1024-2047') # <=== range header
res = urllib.request.urlopen(req)
with open('titas.bin', 'wb') as f:
    f.write(res.read())