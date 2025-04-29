import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the server's address and port
server_address = ('172.26.40.196', 10000)
print('connecting to %s port %s' % server_address, file=sys.stderr)
sock.connect(server_address)

try:
    while True:
        # Receive data from server
        data = sock.recv(1024)
        if data:
            # Split received data by commas and convert to integers
            displacement = [float(value) for value in data.decode('utf-8').split(',')]
            print('received displacement:', displacement, file=sys.stderr)
        else:
            print('no more data from server', file=sys.stderr)
            break

finally:
    print('closing socket', file=sys.stderr)
    sock.close()
