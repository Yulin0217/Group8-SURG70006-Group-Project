from socket import *

host_name = 'LAPTOP-OE6L04IQ' #172.26.104.146
# socket.getbyname(socket.getname())
# print(gethostname())
port_num = 1200

clientSocket = socket(AF_INET,SOCK_STREAM) # ipv4,tcp
clientSocket.connect((host_name,port_num))

message = input('enter something: ')
clientSocket.send(message.encode()) # 变量变成bite

upperMessage = clientSocket.recv(1024).decode() # buffer size
print('the message from the server: ' + upperMessage)

clientSocket.close()