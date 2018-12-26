from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import json
import random
import cgi
import imageio
import cnn_model as model
import torch
import numpy as np

class S(BaseHTTPRequestHandler):
    result = 0
    file_bytes = b""
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
    def respond(self, response, status=200):
        self.send_response(status)
        self.send_header("Content-type","text/html")
        #self.send_header("Content-length",len(response))
        self.end_headers()
        self.wfile.write(bytes(response,"utf-8"))
        
    def do_GET(self):

        if self.path == "/":
            self.path = "/index.html"
        try:
            if self.path.endswith("/index.html"):
                response = """
<!DOCTYPE html>
<html>
<body>

<form enctype="multipart/form-data" method="post" action="/">
  <input name="file" type="file"/>
  <input type="submit" value="Upload"/>
</form>

</body>
</html>
        """
                self.respond(response)
            elif self.path.endswith("/results.html"):
                response = """ 
<!DOCTYPE html>
<html>
<body>
<img src="images/number.png">

Predicted: {}
</body>
</html>
""".format(S.result)
                self.respond(response)
            elif self.path.endswith(".png"):
                self.send_response(200)
                self.send_header('Content-type', 'image/png')
                self.end_headers()
                self.wfile.write(S.file_bytes)
        except IOError:
            self.send_error(404,'File Not Found: %s' % self.path)

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):

        print("in post method")

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     })
        S.file_bytes = form['file'].file.read()
        img = imageio.imread(S.file_bytes)

        mdl = model.Model({})
        mdl.load_state_dict(torch.load('./results/model.pth'))
        mdl.eval()
        output = mdl(torch.from_numpy(np.expand_dims(np.expand_dims(img,0),0)).float())
        S.result = str(torch.argmax(output).item())
        print(S.result)
        print(S.file_bytes)

        self.send_response(302)
        self.send_header('Location', "results.html")
        self.end_headers()
        return


def run(server_class=HTTPServer, handler_class=S, port=80):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...')
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

if len(argv) == 2:
    run(port=int(argv[1]))
else:
    run()
