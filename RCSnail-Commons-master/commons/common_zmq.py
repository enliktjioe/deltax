import numpy as np
import zmq
import zmq.asyncio


async def initialize_publisher(queue: zmq.asyncio.Socket, port):
    queue.sndhwm = 10000
    queue.bind('tcp://127.0.0.1:{}'.format(port))


async def initialize_subscriber(queue: zmq.asyncio.Socket, port):
    queue.connect('tcp://127.0.0.1:{}'.format(port))
    queue.setsockopt(zmq.SUBSCRIBE, b'')


async def initialize_synced_publisher(context: zmq.asyncio.Context, queue: zmq.asyncio.Socket, port):
    queue.sndhwm = 10000
    queue.bind('tcp://127.0.0.1:{}'.format(port))

    synchronizer = context.socket(zmq.REP)
    synchronizer.bind('tcp://127.0.0.1:5560')
    await synchronizer.recv()
    synchronizer.send(b'')

    synchronizer.close()


async def initialize_synced_subscriber(context: zmq.asyncio.Context, queue: zmq.asyncio.Socket, port):
    queue.connect('tcp://127.0.0.1:{}'.format(port))
    queue.setsockopt(zmq.SUBSCRIBE, b'')

    synchronizer = context.socket(zmq.REQ)
    synchronizer.connect('tcp://127.0.0.1:5560')
    synchronizer.send(b'')
    await synchronizer.recv()

    synchronizer.close()


def send_array_with_json(queue: zmq.asyncio.Socket, data, json_data, flags=0, copy=True, track=False):
    """send a json and numpy array with metadata"""
    queue.send_json(json_data, flags | zmq.SNDMORE)
    metadata = dict(dtype=str(data.dtype), shape=data.shape, )
    queue.send_json(metadata, flags | zmq.SNDMORE)
    return queue.send(data, flags, copy=copy, track=track)


async def recv_array_with_json(queue: zmq.asyncio.Socket, flags=0, copy=True, track=False):
    """recv a json and numpy array"""
    json_data = await queue.recv_json(flags=flags)
    metadata = await queue.recv_json(flags=flags)
    msg = await queue.recv(flags=flags, copy=copy, track=track)
    buf = memoryview(msg)
    data = np.frombuffer(buf, dtype=metadata['dtype'])
    return data.reshape(metadata['shape']), json_data
