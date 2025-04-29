import socket
import numpy 
import sys

# ##################     Create a TCP/IP socket   ####################
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the server's address and port
server_address = ('172.30.28.152', 13000)
print(f'connecting to {server_address[0]} port {server_address[1]}', file=sys.stderr)
sock.connect(server_address)


buffer = ""

try:                                                                                                                                                                                                                                                                 
    while True:
        data = sock.recv(1024)  # Receive data from the server
        if not data:
            print("Disconnection", file=sys.stderr)
            break

  
        buffer += data.decode('utf-8')  # Decode the received data
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)  # Split the buffer into lines
            try:
                # Parse the string to a 3x4 matrix (12 values expected)
                matrix_values = [float(value) for value in line.split(',')]
                if len(matrix_values) != 12:
                    raise ValueError("Expected 12 values for a 3x4 matrix")

                # Convert to a 3x4 matrix and extract the last column
                matrix = numpy.array(matrix_values).reshape(3, 4)
                translation_matrix = matrix[:, 3]
            except ValueError as e:
                print(f"Error: {line} ({e})", file=sys.stderr)
                continue

            # Translation matrix contains the offsets in x, y, z
            offset_x = translation_matrix[0]
            offset_y = translation_matrix[1]
            offset_z = translation_matrix[2]

            # Control the dVRK robot using the extracted offsets
            # application.run_servo_cp_steps(offset_x, offset_y, offset_z, steps, dt)

            print(f"Received translation matrix: {translation_matrix}", file=sys.stderr)

except socket.error as e:
    print(f"wrong: {e}", file=sys.stderr)
finally:
    print("Close", file=sys.stderr)
    sock.close()
