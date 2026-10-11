"""Offline regressions: python -m unittest discover -s inference/llm/tests -v."""
import contextlib
import http.client
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import struct
import threading
import types
import unittest
from unittest import mock
import urllib.request


SOURCE = Path(__file__).resolve().parents[1] / "call_api.py"
SPEC = importlib.util.spec_from_file_location("call_api", SOURCE)
call_api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(call_api)
FRAME = 'data: {"choices":[{"delta":{"content":"你好"}}]}\r\n\r\n'.encode("utf-8")
DONE = b"data: [DONE]\r\n\r\n"


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


@contextlib.contextmanager
def serve(handler):
    server = Server(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever,
                              kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


class Upstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        case = self.server.case
        self.close_connection = True
        self.send_response(case if isinstance(case, int) else 200)
        if case == "json" or isinstance(case, int):
            body = b'{"choices":[{"message":{"content":"ok"}}]}'
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Retry-After", "3")
        else:
            self.send_header("Content-Type", "text/event-stream")
            if case == "chunked_cut":
                self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            if case == "small":
                # Fragment UTF-8 and CRLF; withhold DONE until the client
                # receives this complete event, rather than timing a benchmark.
                for byte in FRAME:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                self.server.release.wait(1)
                self.wfile.write(DONE)
            elif case == "stall":
                self.server.release.wait(1)
            elif case == "chunked_cut":
                self.wfile.write(("%x\r\n" % len(FRAME)).encode() + FRAME + b"\r\n")
                # Intentionally omit the terminating zero-length chunk.
            elif case == "disconnect":
                for _ in range(100):
                    self.wfile.write(FRAME * 100)
                    self.wfile.flush()
                    if self.server.release.wait(0.01):
                        break
            else:
                self.wfile.write(body if case == "json" or isinstance(case, int) else FRAME)
        except OSError:
            pass


class ProxyTests(unittest.TestCase):
    @contextlib.contextmanager
    def proxy(self, case, fail=None):
        statuses, responses = [], []
        finished = threading.Event()
        with serve(Upstream) as upstream:
            upstream.case = case
            upstream.release = threading.Event()
            captured = []
            # The official handler is nested inside run_proxy_server. Capture
            # its class without changing the production entry point.
            def capture(address, handler):
                captured.append(handler)
                return mock.Mock()
            with mock.patch.object(call_api.http.server, "ThreadingHTTPServer", capture), \
                    contextlib.redirect_stdout(io.StringIO()):
                call_api.run_proxy_server(api_base=f"http://127.0.0.1:{upstream.server_port}/v1")
            real_open = urllib.request.urlopen
            def open_upstream(request, timeout):
                response = real_open(request, timeout=0.2)
                responses.append(response)
                return response
            original_urllib = call_api.urllib
            call_api.urllib = types.SimpleNamespace(error=original_urllib.error,
                request=types.SimpleNamespace(Request=urllib.request.Request, urlopen=open_upstream))
            class Handler(captured[0]):
                def log_message(self, *args):
                    pass
                def send_response(self, code, message=None):
                    statuses.append(code)
                    return super().send_response(code, message)
                def end_headers(self):
                    if fail == "headers":
                        self.wfile.write(b"HTTP/1.0 200 OK\r\n")
                        raise BrokenPipeError("injected partial header write")
                    super().end_headers()
                    if fail == "body":
                        self.wfile.write = mock.Mock(side_effect=RuntimeError("injected body write"))
                def do_POST(self):
                    try:
                        super().do_POST()
                    finally:
                        finished.set()
            try:
                with serve(Handler) as proxy:
                    yield proxy.server_port, upstream.release, statuses, responses, finished
            finally:
                upstream.release.set()
                call_api.urllib = original_urllib

    def request(self, port, stream=True):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        self.addCleanup(connection.close)
        connection.request("POST", "/v1/chat/completions", json.dumps({"stream": stream}).encode(),
                           {"Content-Type": "application/json"})
        return connection.getresponse()

    def test_nonstream_json(self):
        with self.proxy("json") as (port, release, codes, responses, finished):
            response = self.request(port, stream=False)
            self.assertEqual(response.status, 200)
            self.assertEqual(json.loads(response.read())["choices"][0]["message"]["content"], "ok")
            self.assertEqual(codes, [200])

    def test_small_fragmented_event_arrives_before_done(self):
        with self.proxy("small") as (port, release, codes, responses, finished):
            response = self.request(port)
            received = b""
            while len(received) < len(FRAME):
                data = response.read1(4096)
                self.assertTrue(data)
                received += data
            self.assertEqual(received, FRAME)
            release.set()
            self.assertEqual(response.read(), DONE)
            self.assertEqual(codes, [200])

    def test_clean_cut_never_invents_done(self):
        with self.proxy("cut") as (port, release, codes, responses, finished):
            self.assertEqual(self.request(port).read(), FRAME)
            self.assertEqual(codes, [200])

    def test_truncated_chunked_no_second_response(self):
        with self.proxy("chunked_cut") as (port, release, codes, responses, finished):
            response = self.request(port)
            self.assertIsNone(response.getheader("Transfer-Encoding"))
            self.assertNotIn(b"HTTP/", response.read())
            self.assertEqual(codes, [200])

    def test_stall_after_headers_closes_without_502(self):
        with self.proxy("stall") as (port, release, codes, responses, finished):
            self.assertEqual(self.request(port).read(), b"")
            self.assertEqual(codes, [200])
            self.assertTrue(finished.wait(1))
            self.assertTrue(responses[0].closed)

    def test_http_error_codes_preserved(self):
        for code in (429, 500, 502, 504):
            with self.subTest(code=code), self.proxy(code) as (port, release, codes, responses, finished):
                response = self.request(port)
                self.assertEqual(response.status, code)
                self.assertEqual(response.getheader("Retry-After"), "3")
                json.loads(response.read())
                self.assertEqual(codes, [code])

    def test_downstream_reset_closes_upstream(self):
        with self.proxy("disconnect") as (port, release, codes, responses, finished):
            with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
                sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}")
                received = b""
                while b"\r\n\r\n" not in received or not received.split(b"\r\n\r\n", 1)[1]:
                    received += sock.recv(8192)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                struct.pack("hh" if os.name == "nt" else "ii", 1, 0))
            self.assertTrue(finished.wait(2))
            self.assertEqual(codes, [200])
            self.assertTrue(responses[0].closed)

    def test_header_and_body_write_failures_close_without_second_status(self):
        for fail in ("headers", "body"):
            with self.subTest(fail=fail), self.proxy("cut", fail) as (port, release, codes, responses, finished):
                with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
                    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}")
                    while sock.recv(4096):
                        pass
                self.assertTrue(finished.wait(1))
                self.assertEqual(codes, [200])
                self.assertTrue(responses[0].closed)

    def test_server_closed_when_serve_forever_raises(self):
        server = mock.Mock()
        server.serve_forever.side_effect = RuntimeError("injected server failure")
        with mock.patch.object(call_api.http.server, "ThreadingHTTPServer", return_value=server), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            call_api.run_proxy_server()
        server.server_close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
