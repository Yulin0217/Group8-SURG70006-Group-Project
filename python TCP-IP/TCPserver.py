from socket import *

serverSocket = socket(AF_INET,SOCK_STREAM) # 该socket用于倾听连接
serverSocket.bind((gethostname(),1200))
# serverSocket.bind(gethostname(),1200)

serverSocket.listen(3) # 允许的最大队列数量
print('the server is ready to accept information...')

connectionSocket, address = serverSocket.accept() # 该socket用于传输数据
message = connectionSocket.recv(1024).decode() # bite 改string
print('got the message from the client: ' + message)

modifiedMessage = message.upper().encode() # string改
connectionSocket.send(modifiedMessage)

connectionSocket.close()