import threading
import zmq
from typing import Callable, Any, Optional

class ZMQRepServer:
    """
    Simple ZeroMQ REP server wrapper.

    Usage:
      server = ZMQRepServer("tcp://*:5555", handler=my_handler)
      server.start()
      ...
      server.stop()
    """

    def __init__(
        self,
        bind_addr: str,
        handler: Optional[Callable[[Any], Any]] = None,
        context: Optional[zmq.Context] = None,
        # poll_timeout_ms: int = 500,
        poll_timeout_ms = None
    ):
        """
        :param bind_addr: e.g. "tcp://*:5555"
        :param handler: callable(request) -> reply. Can be sync; called in server thread.
        :param context: optional shared zmq.Context (otherwise one is created)
        :param poll_timeout_ms: poll timeout for the socket in milliseconds
        """
        self.bind_addr = bind_addr
        self.context = context or zmq.Context()
        self.handler = handler
        self.poll_timeout_ms = poll_timeout_ms

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._sock: Optional[zmq.Socket] = None

    def start(self):
        """Start server loop in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the server and wait for thread to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        # ensure socket is closed
        if self._sock:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass
            self._sock = None
        # Do not terminate shared context here if provided externally
        # If we created context ourselves, terminate it.
        if self.context is not None:
            try:
                # only terminate if there are no other sockets/threads using it,
                # but generally safe to leave the context running if shared externally.
                pass
            except Exception:
                pass

    def _serve_loop(self):
        sock = self.context.socket(zmq.REP)
        self._sock = sock
        sock.bind(self.bind_addr)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        try:
            while not self._stop_event.is_set():
                events = dict(poller.poll(self.poll_timeout_ms))
                if sock in events and events[sock] == zmq.POLLIN:
                    try:
                        # Use recv_json for convenience; adapt if you need binary frames.
                        request = sock.recv_json(flags=0)
                    except zmq.ZMQError as e:
                        # interrupted or socket error; break if requested
                        if self._stop_event.is_set():
                            break
                        continue

                    # Call handler (user-supplied) to produce a reply
                    if self.handler:
                        try:
                            reply = self.handler(request)
                        except Exception as e:
                            # if handler raises, send an error structure back
                            reply = {"error": str(e)}
                    else:
                        # no handler -> echo the request
                        reply = {"echo": request}

                    # send the reply (must follow recv on REP socket)
                    try:
                        sock.send_json(reply)
                    except zmq.ZMQError:
                        # if send fails, continue loop
                        continue
                # else: loop continues and checks stop_event
        finally:
            # Clean up
            try:
                poller.unregister(sock)
            except Exception:
                pass
            try:
                sock.close(linger=0)
            except Exception:
                pass
            self._sock = None