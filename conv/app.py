import tornado.ioloop
import tornado.web
import tornado.websocket
import src.cifarUtils as cifarUtils
import numpy
from PIL import Image, ImageFilter
from scipy.misc import toimage
import io
import json
import tornado.autoreload
import os
import signal
import uuid
from src.convNNRunner import ConvNNRunner

from tornado.options import define, options, parse_command_line

define("port", default=8888, help="run on the given port", type=int)
define("watch", default=False, help="Auto reload changes in web dir", type=bool)

clients = dict()

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

    def data_received(self, chunk):
        pass

class WebSocketHandler(tornado.websocket.WebSocketHandler):

    def __init__(self, *args, **kwargs):
        super(WebSocketHandler, self).__init__(*args, **kwargs)

    def data_received(self, chunk):
        pass

    def open(self, *args):
        self.id = str(uuid.uuid4())
        clients[self.id] = None
        self.write_message({"uid": self.id}, False)

    def on_message(self, message):
        event = json.loads(message)
        print(message)
        if {'pause', 'id'} <= set(event):
            self.handle_pause(event)
        elif {'stop', 'id'} <= set(event):
            self.handle_stop(event)
        elif {'save', 'id'} <= set(event):
            self.handle_save(event)
        elif {'CNN', 'id'} <= set(event):
            self.handle_save(event)

    def handle_save(self, event):
        clients[event['id']].save()

    def handle_stop(self, event):
        if event['stop']:
            if event['id'] in clients:
                clients[event['id']].stop()
                del clients[event['id']]

    def handle_pause(self, event):
        if event['pause']:
            clients[event['id']].pause()
        else:
            if clients[event['id']] is not None:
                clients[event['id']].resume()
            else:
                print("starting new thread")
                clients[event['id']] = ConvNNRunner(self, event['id'])
                clients[event['id']].start()

    def on_close(self):
        pass

    def on_convnet_save(self, json_convent):
        self.write_message(json_convent, False)

    def on_forward_prop(self, img, stats):
        output = io.BytesIO()
        img.save(output, format='JPEG')
        self.write_message(output.getvalue(), True)
        self.write_message(json.dumps(stats), False)

    def on_train(self, stats):
        self.write_message(json.dumps(stats), False)

    def on_test_set_validation(self, stats):
        self.write_message(json.dumps(stats), False)

public_root = os.path.join(os.path.dirname(__file__), 'web')

def on_shutdown():
    print('Shutting down')
    for key, runner in clients.iteritems():
        print("stopping " + key)
        runner.stop()
    clients.clear()
    tornado.ioloop.IOLoop.instance().stop()


settings = dict(
    debug=True,
    static_path=public_root,
    template_path=public_root
)

app = tornado.web.Application([
    (r'/', IndexHandler),
    (r'/ws', WebSocketHandler),
    (r'/(.*)', tornado.web.StaticFileHandler, {'path': public_root})
], **settings)

if __name__ == '__main__':
    tornado.autoreload.start()
    parse_command_line()
    if options.watch:
        for dir, _, files in os.walk('./web'):
            [tornado.autoreload.watch(dir + '/' + f) for f in files if not f.startswith('.')]

    app.listen(options.port)
    io_loop = tornado.ioloop.IOLoop.instance()
    signal.signal(signal.SIGINT, lambda sig, frame: io_loop.add_callback_from_signal(on_shutdown))
    io_loop.start()

