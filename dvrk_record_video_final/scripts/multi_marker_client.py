import socket
import numpy as np


def start_client():
    # Create a TCP/IP client socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Define the server address and port (ensure it matches the server configuration)
    server_address = ('172.26.40.196', 12000)
    print('Connecting to server...')

    try:
        # Connect to the server
        client_socket.connect(server_address)
        print('Connected to server.')

        while True:
            # Receive data from the server (buffer size set to 4096 bytes)
            data = client_socket.recv(4096)

            if not data:
                print("No data received. Closing connection.")
                break

            # Decode the received data
            received_str = data.decode('utf-8')
            print(f"Raw received data: {received_str}")

            # Parse data into a NumPy array
            try:
                # Convert the string to a Python list
                translation_list = eval(received_str)

                # Convert the list into a 2D NumPy matrix, each row containing a translation vector
                translation_matrix = np.array(translation_list).reshape(-1, 3)
                print(f"Translation Matrix:\n{translation_matrix}")
            except Exception as e:
                print(f"Error parsing received data: {e}")
                continue

    except Exception as e:
        print(f"Error connecting to server: {e}")
    finally:
        # Close the connection
        client_socket.close()
        print("Connection closed.")


if __name__ == "__main__":
    start_client()
