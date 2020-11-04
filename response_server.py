"""
    Social-Distancing

    IIT : Istituto italiano di tecnologia

    Pattern Analysis and Computer Vision (PAVIS) research line

    Description: Social distancing json response server

    Disclaimer:
    The information and content provided by this application is for information purposes only.
    You hereby agree that you shall not make any health or medical related decision based in whole
    or in part on anything contained within the application without consulting your personal doctor.
    The software is provided "as is", without warranty of any kind, express or implied,
    including but not limited to the warranties of merchantability,
    fitness for a particular purpose and noninfringement. In no event shall the authors,
    PAVIS or IIT be liable for any claim, damages or other liability, whether in an action of contract,
    tort or otherwise, arising from, out of or in connection with the software
    or the use or other dealings in the software.

    LICENSE:
"""

import os
import sys
import time
from sys import platform
import socket
import threading
import queue
import signal
import datetime
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
import json
import threading
import glob


class ResponseServer:
    """ResponseServer class, send images/json to remote clients, single response
    """

    commands = ["restart", "reboot"]

    def __init__(self, port, content_type):
        """Initialize resposne server

        Args:
            port (string): listen port
            mt ([type]): [description]
        """    
        self.port = port
        self.run = True
        self.content_type = content_type
        self.key_lock = threading.Lock()
        self.restart = False

    def activate(self):
        """Listen on port
        """        
        self.run = True
        # Start listen thread
        threading.Thread(target=self.listen).start()

    def disconnect(self):
        """Stop listening and close socket
        """        
        self.s.shutdown(socket.SHUT_RDWR)
        self.s.close()
        self.run = False
        self.block = None

    def listen(self):
        """Listen on selected port and start new client response
        """        
        # Create server socket
        port = self.port

        # Configure server and reuse address
        self.s = socket.socket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind(('0.0.0.0', port))
        self.s.listen()
        self.block = bytes("{}", "UTF-8")

        # Wait for new connections
        while self.run:
            try:
                # Wait new connection
                c, addr = self.s.accept()

                # Print connection
                #print("Connection from {0} to local port {1}".format(addr, self.port), flush=True)

                # Crete new server thread
                threading.Thread(target=self.client_handler, args=(c,)).start()
            except socket.error as e:
                print("Error while listening :{0}".format(e), flush=True)

        print("thermal listen stop", flush=True)

    def client_handler(self, c):
        """Response thread, send json to connected client

        Args:
            c (socket): socket descriptor
        """
        # Read request from remote web client
        data = c.recv(1024)

        # Decode data to eventually use it
        data = data.decode("UTF-8")

        # Decode REST command (if any, this class can be used as rest command receiver)
        lines = data.split("\r\n")
        
        try:
            # Split received lines
            restful = lines[0].split(" ")[1]
        
            # Read rest command
            if len(restful)>2 and restful != "favicon.ico":
                finded = False
                # Find receive rest command in command list
                for command in self.commands:
                    # If command is present in command list check command value
                    if command in restful:
                        value = restful.split("?")[1].split("=")[1]
                        if command == "restart" and value == "1":
                            # Remote client request a restart
                            print("Restart requeste received!", flush=True)
                            self.restart = True
                            block = '{"status":"ok"}'
                            finded = True
                            break
                # Unable to find command in command list
                if not finded:
                    block = '{"status":"failed"}'
                
                self.put(bytes(block, "UTF-8"))
        except Exception as e: 
            print(lines)
            print(e, flush=True)

        # Critical region
        self.key_lock.acquire()

        # Create a fake header to send to remote client
        response = "HTTP/1.0 200 OK\r\n" \
            "Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n" \
            "Pragma: no-cache\r\n" \
            "Expires: Thu, 01 Dec 1994 16:00:00 GMT\r\n" \
            "Connection: close\r\n" \
            "Content-Type: " + self.content_type + "\r\n" \
            "Content-Length: " + str(len(self.block)) + "\r\n\r\n"

        # Print sending response
        # print(response)

        # Try to send data until socket is valid
        try:
            c.send(bytes(response, "UTF-8"))
        except socket.error as e:
            print(e, flush=True)

        # Try to send data until socket is valid
        try:
            c.send(self.block)
        except socket.error as e:
            print(e, flush=True)

        self.key_lock.release()

        c.close()

    def put(self, dt):
        """Put into sending block, client will receive these data

        Args:
            dt (json data): various data in json format
        """        
        self.key_lock.acquire()
        self.block = dt
        self.key_lock.release()

    def restart_status(self):
        """restart command receved as REST

        Returns:
            command: restart command
        """
        return self.restart