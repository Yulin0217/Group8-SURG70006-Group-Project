# import socket
# import numpy 
# import sys

# # ##################     Create a TCP/IP socket   ####################
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# # Connect the socket to the server's address and port
# server_address = ('172.30.28.152', 5555)
# print(f'connecting to {server_address[0]} port {server_address[1]}', file=sys.stderr)
# sock.connect(server_address)


# buffer = ""

# try:                                                                                                                                                                                                                                                                 
#     while True:
#         data = sock.recv(1024)  # Receive data from the server
#         if not data:
#             print("Disconnection", file=sys.stderr)
#             break

  
#         buffer += data.decode('utf-8')  # Decode the received data
#         while '\n' in buffer:
#             line, buffer = buffer.split('\n', 1)  # Split the buffer into lines
#             try:
#                 # Parse the string to a 3x4 matrix (12 values expected)
#                 matrix_values = [float(value) for value in line.split(',')]
#                 if len(matrix_values) != 12:
#                     raise ValueError("Expected 12 values for a 3x4 matrix")

#                 # Convert to a 3x4 matrix and extract the last column
#                 matrix = numpy.array(matrix_values).reshape(3, 4)
#                 translation_matrix = matrix[:, 3]
#             except ValueError as e:
#                 print(f"Error: {line} ({e})", file=sys.stderr)
#                 continue

#             # Translation matrix contains the offsets in x, y, z
#             offset_x = translation_matrix[0]
#             offset_y = translation_matrix[1]
#             offset_z = translation_matrix[2]

#             # Control the dVRK robot using the extracted offsets
#             # application.run_servo_cp_steps(offset_x, offset_y, offset_z, steps, dt)

#             print(f"Received translation matrix: {translation_matrix}", file=sys.stderr)

# except socket.error as e:
#     print(f"wrong: {e}", file=sys.stderr)
# finally:
#     print("Close", file=sys.stderr)
#     sock.close()











import socket
import numpy
import sys

def connect_and_receive():
    # 创建一个 TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 服务器地址和端口
    server_address = ('172.30.28.152', 5554)
    print(f'Connecting to {server_address[0]} port {server_address[1]}', file=sys.stderr)

    try:
        # 连接到服务器
        sock.connect(server_address)

        # 接收数据
        data = sock.recv(1024)  # 每次接收 1024 字节
        if not data:
            print("No data received from server", file=sys.stderr)
            return

        # 打印原始数据
        print('Raw data:', data.decode('utf-8'), file=sys.stderr)

        # 解析数据为 3x4 矩阵
        try:
            line = data.decode('utf-8').strip()  # 解码并去除多余的空白字符
            rows = line[1:-1].split('],[')  # 按 "],[" 分割行数据
            matrix = numpy.array([[float(value) for value in row.split()] for row in rows])  # 转换为 2D 数组

            if matrix.shape != (3, 4):
                raise ValueError("Expected a 3x4 matrix")

            # 提取最后一列作为 translation_matrix
            translation_matrix = matrix[:, 3]

            # 打印解析结果
            print(f"Received matrix:\n{matrix}", file=sys.stderr)
            print(f"Translation matrix: {translation_matrix}", file=sys.stderr)

        except ValueError as e:
            print(f"Error parsing matrix: {line} ({e})", file=sys.stderr)

    except socket.error as e:
        print(f"Socket error: {e}", file=sys.stderr)
    finally:
        # 关闭连接
        print("Closing connection", file=sys.stderr)
        sock.close()

# 主循环，每次按下 Enter 触发一次连接
if __name__ == "__main__":
    try:
        while True:
            input("Press Enter to connect to server and receive data (type 'exit' to quit)...")
            connect_and_receive()
    except KeyboardInterrupt:
        print("\nExiting program.", file=sys.stderr)