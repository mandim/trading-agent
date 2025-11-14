# zmq_client_test.py
import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://127.0.0.1:6000")

print("Client: sending hello")
socket.send(b"hello from client")
reply = socket.recv()
print("Client: got reply:", reply)
