import socket
import sys
import time

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('127.0.0.1', 10000)
print('connecting to %s port %s' % server_address, file=sys.stderr)
sock.connect(server_address)

try:
    while True:
        for i in range(1, 101):
            # Send data
            message = str(i)
            print('sending "%s"' % message, file=sys.stderr)
            sock.sendall(message.encode())

            # Wait for the response
            data = sock.recv(16)
            print('received "%s"' % data.decode(), file=sys.stderr)

            # Optional: add a short delay for readability
            time.sleep(0.5)

finally:
    print('closing socket', file=sys.stderr)
    sock.close()
