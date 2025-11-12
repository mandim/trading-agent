import zmq
from typing import Any, Optional
import time
from trading_env_merged import TradingEnv

class ZMQReqClient:
    """Tiny REQ client for testing the server."""

    def __init__(self, connect_addr: str, context: Optional[zmq.Context] = None, timeout_ms: int = None):
        self.connect_addr = connect_addr
        self.context = context or zmq.Context()
        self.timeout_ms = timeout_ms
        self._sock = self.context.socket(zmq.REQ)
        self._sock.connect(self.connect_addr)
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)

    def request(self, payload: Any) -> Any:
        """Send payload (JSON-serializable) and wait for reply or raise TimeoutError."""
        self._sock.send_json(payload)
        socks = dict(self._poller.poll(self.timeout_ms))
        if self._sock in socks and socks[self._sock] == zmq.POLLIN:
            return self._sock.recv_json()
        else:
            raise TimeoutError("No reply from server (timeout)")

    def close(self):
        try:
            self._sock.close(linger=0)
        except Exception:
            pass

if __name__ == '__main__':

    env = TradingEnv(0.0001, 'EURUSD_Daily.csv', 'EURUSD_Ticks.csv', bind_address="tcp://127.0.0.1:5566")
    env.start_server()

    # Give some time to the server to bind
    time.sleep(0.1)

    client = ZMQReqClient("tcp://127.0.0.1:5566")

    try:
        for index in range(5):
            print(f"Opening long position")
            print(client.request({"cmd": "BUY"}))
            print(f"Opening short position")
            print(client.request({"cmd": "SELL"}))

    finally:
        client.close()
        env.stop_server()

    # env.open_position("BUY", 50, 50)