#! /usr/bin/env python3
import paho.mqtt.client as mqtt
from sys import argv

if len(argv) != 4:
        print('not enough arguments given')
            print('''
            subscribe = argv[1]
            publish = argv[2]
            qos = argv[3] # default is 1''')
                print(argv)
                    exit(1)

                    subscribe = argv[1]
                    publish = argv[2]
                    qos = int(argv[3])
                    requestForInput = 'enter message (q to quit): '
                    name = str(input('what is your name?: '))

                    def on_connect(client, userdata, flags, rc):
                            print("\x1b[1K\rConnection returned result: "+str(rc)+'\n'+requestForInput,end='')
                                client.subscribe(subscribe, qos=qos)

                                def on_disconnect(client, userdata, rc):
                                        if rc != 0:
                                                    print('Unexpected Disconnect')
                                                        else:
                                                                    print('Expected Disconnect')

                                                                    def on_message(client, userdata, message):
                                                                            print(f'\x1b[1K\r==={message.topic}:{message.qos}===\n{str(message.payload.decode())}\n===\n{requestForInput}', end='')

                                                                            # 1. create a client instance.
                                                                            client = mqtt.Client()

                                                                            # add additional client options (security, certifications, etc.)
                                                                            # many default options should be good to start off.
                                                                            # add callbacks to client.
                                                                            client.on_connect = on_connect
                                                                            client.on_disconnect = on_disconnect
                                                                            client.on_message = on_message

                                                                            # 2. connect to a broker using one of the connect*() functions.
                                                                            client.connect_async("test.mosquitto.org")

                                                                            # 3. call one of the loop*() functions to maintain network traffic flow with the broker.
                                                                            client.loop_start()
                                                                            # client.loop_forever()
                                                                            while True: # perhaps add a stopping condition using some break or something.
                                                                                    s = str(input(requestForInput))
                                                                                        if s == 'q':
                                                                                                    break
                                                                                                    client.publish(publish,f'{name}: {s}', qos=qos)
                                                                                                    # use subscribe() to subscribe to a topic and receive messages.
                                                                                                    # use publish() to publish messages to the broker.
                                                                                                    # use disconnect() to disconnect from the broker.

                                                                                                    client.loop_stop()
                                                                                                    client.disconnect()

