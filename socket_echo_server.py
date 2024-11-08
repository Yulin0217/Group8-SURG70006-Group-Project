import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the port
server_address = ('172.26.99.133', 10000)
print('starting up on %s port %s' % server_address, file=sys.stderr)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True:
    # Wait for a connection
    print('waiting for a connection', file=sys.stderr)
    connection, client_address = sock.accept()

    try:
        print('connection from', client_address, file=sys.stderr)

        # Send a single array of displacement values and then stop
        displacement = [0.005, 0.005, 0.005]
        message = ','.join(map(str, displacement))  # Convert list to "0.005,0.005,0.005"
        print(f'sending displacement: {message}', file=sys.stderr)
        connection.sendall(message.encode())

    finally:
        # Close the connection after sending the data
        print('closing connection', file=sys.stderr)
        connection.close()
        break  # Exit the server loop to stop the server after sending one message
