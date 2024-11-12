import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the server's address and port
server_address = ('172.26.211.137', 10000)
print >>sys.stderr, 'connecting to %s port %s' % server_address
sock.connect(server_address)

try:
    while True:
        # Receive data from server
        data = sock.recv(1024)
        if data:
            # Split received data by commas and convert to floats
            values = [float(value) for value in data.decode('utf-8').split(',')]
            
            # Assuming the first three values are the translation differences
            translation_difference = values[:3]
            # The next nine values are the rotation matrix (3x3)
            rotation_matrix = values[3:]
            rotation_matrix = [rotation_matrix[:3], rotation_matrix[3:6], rotation_matrix[6:]]
            
            # Print the received translation and rotation matrices
            print >>sys.stderr, 'Received translation difference:', translation_difference
            print >>sys.stderr, 'Received rotation matrix:'
            for row in rotation_matrix:
                print >>sys.stderr, row
        else:
            print >>sys.stderr, 'No more data from server'
            break

finally:
    print >>sys.stderr, 'Closing socket'
    sock.close()
