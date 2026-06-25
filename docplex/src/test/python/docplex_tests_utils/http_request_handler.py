'''
Created on Jun 22, 2018

@author: kong
'''
try:
    from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
except ImportError:
    from http.server import HTTPServer, BaseHTTPRequestHandler

import json

sample_response = [
  {
    "applicationId": "sample",
    "applicationVersion": "string",
    "applicationVersionUsed": "string",
    "executionStatus": "CREATED",
    "createdAt": 0,
    "startedAt": 0,
    "endedAt": 0,
    "endRecordedAt": 0,
    "updatedAt": 0,
    "parameters": {},
    "processingOwner": "string",
    "lastProcessingOwner": "string",
    "owner": "string",
    "attachments": [
      {
        "name": "string",
        "type": "OUTPUT_ATTACHMENT",
        "length": 0
      }
    ],
    "failureInfo": {
      "type": "UNKNOWN",
      "message": "string"
    },
    "language": "string",
    "details": {},
    "solveStatus": "UNKNOWN",
    "interruptionStatus": "STOP",
    "interruptedAt": 0,
    "submittedAt": 0,
    "nbLogItems": 0,
    "subscription": {
      "id": "string",
      "type": "string",
      "customerId": "string",
      "customerIBMId": "string"
    },
    "subscriber": {
      "ibmid": "string",
      "name": "string",
      "email": "string"
    },
    "apiClient": {
      "id": "string",
      "name": "string",
      "email": "string"
    },
    "_rev": "string",
    "_id": "string"
  }
]


class HttpRequestHandler502(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/jobs':
            # If self.server hasa  retry count, use it, otherwise always
            # return 502
            req_before_ok = self.server.req_before_ok if hasattr(self.server, 'req_before_ok') else 100000
            if req_before_ok <= 0:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(sample_response).encode())
            else:
                self.server.req_before_ok = req_before_ok - 1
                self.send_error(502, 'simulated 502')
        else:
            self.send_error(404, 'No such page')


def create_server_502(hostname='127.0.0.1', port=0):
    httpd = HTTPServer((hostname, port), HttpRequestHandler502)
    httpd.req_before_ok = 2  # returns 502 two times, then ok after that
    return httpd
