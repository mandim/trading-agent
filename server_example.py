import zmq

context = zmq.Context()
socket = context.socket(zmq.REP)

# Bind on all interfaces, port 6000
socket.bind("tcp://*:6000")

print("Server: listening on tcp://*:6000")

while True:
    msg = socket.recv()
    print("Server: received:", msg)
    socket.send(b"ok")
