"""
This code allows you to serve static files from the same port as the websocket connection
This is only suitable for small files and as a development server!
open(full_path, 'rb').read() call that is used to send files will block the whole asyncio loop!
"""

import os
import asyncio
import datetime
import random
import websockets
import posixpath
import mimetypes
import base64
from http import HTTPStatus


class WebSocketServerProtocolWithHTTP(websockets.WebSocketServerProtocol):
    """Implements a simple static file server for WebSocketServer"""

    async def process_request(self, path, request_headers):
        """Serves a file when doing a GET request with a valid path"""

        if "Upgrade" in request_headers:
            return  # Probably a WebSocket connection

        if path == '/':
            path = '/index.html'

        response_headers = [
            ('Server', 'asyncio'),
            ('Connection', 'close'),
        ]
        server_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),"www")
        full_path = os.path.realpath(os.path.join(server_root, path[1:]))

        print("GET", path, end=' ')

        # Validate the path
        if os.path.commonpath((server_root, full_path)) != server_root or \
                not os.path.exists(full_path) or not os.path.isfile(full_path):
            print("404 NOT FOUND")
            return HTTPStatus.NOT_FOUND, [], b'404 NOT FOUND'

        print("200 OK")
        body = open(full_path, 'rb').read()
        ctype = self.guess_type(path)
        response_headers.append(("Content-type", ctype))
        response_headers.append(('Content-Length', str(len(body))))
        return HTTPStatus.OK, response_headers, body

    def guess_type(self, path):
        """Guess the type of a file.

        Argument is a PATH (a filename).

        Return value is a string of the form type/subtype,
        usable for a MIME Content-type header.

        The default implementation looks the file's extension
        up in the table self.extensions_map, using application/octet-stream
        as a default; however it would be permissible (if
        slow) to look inside the data to make a better guess.

        """
 
        base, ext = posixpath.splitext(path)
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        ext = ext.lower()
        if ext in self.extensions_map:
            return self.extensions_map[ext]
        else:
            return self.extensions_map['']
 
    if not mimetypes.inited:
        mimetypes.init() # try to read system mime.types
    extensions_map = mimetypes.types_map.copy()
    extensions_map.update({
        '': 'application/octet-stream', # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
        })

async def on_connection(websocket, path):
    while True:
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        await websocket.send(now)
        data = await websocket.recv()
        print("Received: {}".format(base64.b64decode(data)))
        await asyncio.sleep(random.random() * 2)


if __name__ == "__main__":
    start_server = websockets.serve(on_connection, 'localhost', 80,
                                    create_protocol=WebSocketServerProtocolWithHTTP)
    print("Running server at http://localhost:80/")

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()