import socket
import sys
import time

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

        # Send (x, y, z) displacement data in a loop
        x, y, z = 0, 0, 0
        while True:
            # Format displacement data as a comma-separated string
            message = f"{x},{y},{z}"
            print(f'sending displacement: {message}', file=sys.stderr)
            connection.sendall(message.encode())

            # Update (x, y, z) values (increment as a demonstration)
            x += 1
            y += 2
            z += 3

            # Delay for readability (optional)
            time.sleep(1)

    finally:
        # Clean up the connection
        connection.close()
