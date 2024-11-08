import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the server's address and port
server_address = ('172.26.99.133', 10000)
print >>sys.stderr, 'connecting to %s port %s' % server_address
sock.connect(server_address)

try:
    while True:
        # Receive data from server
        data = sock.recv(32)
        if data:
            # Split received data by commas and convert to integers
            displacement = [int(value) for value in data.decode('utf-8').split(',')]
            print >>sys.stderr, 'received displacement:', displacement
        else:
            print >>sys.stderr, 'no more data from server'
            break

finally:
    print >>sys.stderr, 'closing socket'
    sock.close()
