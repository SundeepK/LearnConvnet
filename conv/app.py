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
        self.conv_runner = ConvNNRunner(self)

    def data_received(self, chunk):
        pass

    def open(self, *args):
        self.conv_runner.start()

    def on_message(self, message):
        print "Client %s received a message : %s" % (self.id, message)

    def on_close(self):
        pass

    def onForwardProp(self, img, stats):
        output = io.BytesIO()
        img.save(output, format='JPEG')
        self.write_message(output.getvalue(), True)
        self.write_message(json.dumps(stats.__dict__), False)

public_root = os.path.join(os.path.dirname(__file__), 'web')

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
    #TODO remove in prod
    tornado.autoreload.start()
    parse_command_line()
    if options.watch:
        for dir, _, files in os.walk('./web'):
            [tornado.autoreload.watch(dir + '/' + f) for f in files if not f.startswith('.')]

    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

