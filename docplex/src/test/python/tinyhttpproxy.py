'''
'''
__version__ = "0.2.1"

import select, socket
import threading, time
import traceback

try:
    # python 2
    from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
    import BaseHTTPServer
    import urlparse
    import SocketServer
except:
    # python 3
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse
    import socketserver as SocketServer



class ProxyHandler (BaseHTTPRequestHandler):
    """Code based on: http://www.oki-osk.jp/esc/python/proxy/
    """
    __base = BaseHTTPRequestHandler
    __base_handle = __base.handle

    server_version = "TinyHTTPProxy/" + __version__
    rbufsize = 0                        # self.rfile Be unbuffered

    shutdown = False

    out_byte_count = 0
    in_byte_count = 0

    @staticmethod
    def init():
        ProxyHandler.shutdown = False
        ProxyHandler.out_byte_count = 0
        ProxyHandler.in_byte_count= 0

    def handle(self):
        (ip, port) = self.client_address
        if hasattr(self, 'allowed_clients') and ip not in self.allowed_clients:
            self.raw_requestline = self.rfile.readline()
            if self.parse_request(): self.send_error(403)
        else:
            self.__base_handle()

    def _connect_to(self, netloc, soc):
        i = netloc.find(':')
        if i >= 0:
            host_port = netloc[:i], int(netloc[i+1:])
        else:
            host_port = netloc, 80
        print("\tconnect to %s:%d" % host_port)
        try:
            soc.connect(host_port)
        except socket.error as arg:
            try:
                msg = arg[1]
            except:
                msg = arg
            print('socket error: %s' % msg)
            self.send_error(404, msg)
            return 0
        return 1

    def do_CONNECT(self):
        ProxyHandler.out_byte_count = 0
        ProxyHandler.in_byte_count = 0
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if self._connect_to(self.path, soc):
                self.log_request(200)
                self.wfile.write((self.protocol_version +
                                  " 200 Connection established\r\n").encode())
                self.wfile.write(("Proxy-agent: %s\r\n" % self.version_string()).encode())
                self.wfile.write("\r\n".encode())
                self._read_write(soc, 300)
        except Exception as exc:
            traceback.print_exc()
        finally:
            print("\tbye")
            print("\tsent = %d bytes" % ProxyHandler.out_byte_count)
            print("\treceived = %d bytes" % ProxyHandler.in_byte_count)
            soc.close()
            self.connection.close()

    def do_GET(self):
        (scm, netloc, path, params, query, fragment) = urlparse.urlparse(
            self.path, 'http')
        if scm != 'http' or fragment or not netloc:
            self.send_error(400, "bad url %s" % self.path)
            return
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if self._connect_to(netloc, soc):
                self.log_request()
                soc.send("%s %s %s\r\n" % (
                    self.command,
                    urlparse.urlunparse(('', '', path, params, query, '')),
                    self.request_version))
                self.headers['Connection'] = 'close'
                del self.headers['Proxy-Connection']
                for key_val in self.headers.items():
                    soc.send("%s: %s\r\n" % key_val)
                soc.send("\r\n")
                self._read_write(soc)
        finally:
            print("\tbye")
            soc.close()
            self.connection.close()

    def _read_write(self, soc, max_idling=20):
        iw = [self.connection, soc]
        ow = []
        count = 0
        while not ProxyHandler.shutdown:
            count += 1
            (ins, _, exs) = select.select(iw, ow, iw, 3)
            if exs: break
            if ins:
                for i in ins:
                    direction = 0
                    if i is soc:
                        out = self.connection
                    else:
                        direction = 1
                        out = soc
                    data = i.recv(8192)
                    if data:
                        out.send(data)
                        if direction == 1:
                            ProxyHandler.out_byte_count += len(data)
                        else:
                            ProxyHandler.in_byte_count += len(data)
                        count = 0
            if count == max_idling: break

    do_HEAD = do_GET
    do_POST = do_GET
    do_PUT  = do_GET
    do_DELETE=do_GET


class TinyHTTPProxy():
    def __init__(self):
        """A simple http proxy

        Attributes:
            p: The port this proxy is bound to
        """
        self.port = 0
        pass

    def start(self):
        """Starts the proxy"""
        self.httpd = HTTPServer(("", 0), ProxyHandler)
        ProxyHandler.init()
        self.t = threading.Thread(target=self.httpd.serve_forever)
        self.t.start()
        time.sleep(2)
        self.port = self.httpd.socket.getsockname()[1]
        print("Proxy serving on port %s" % self.port)

    def shutdown(self):
        """shut this proxy down.
        """
        ProxyHandler.shutdown = True
        self.httpd.shutdown()
        self.t.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.shutdown()

    @property
    def out_byte_count(self):
        return ProxyHandler.out_byte_count

    @property
    def in_byte_count(self):
        return ProxyHandler.in_byte_count

class ThreadingHTTPServer (SocketServer.ThreadingMixIn,
                           HTTPServer): pass

if __name__ == '__main__':
    from sys import argv
    if argv[1:] and argv[1] in ('-h', '--help'):
        print(argv[0] + "[port [allowed_client_name ...]]")
    else:
        if argv[2:]:
            allowed = []
            for name in argv[2:]:
                client = socket.gethostbyname(name)
                allowed.append(client)
                print("Accept: %s (%s)" % (client, name))
            ProxyHandler.allowed_clients = allowed
            del argv[2:]
        else:
            print("Any clients will be served...")
        BaseHTTPServer.test(ProxyHandler, ThreadingHTTPServer)

