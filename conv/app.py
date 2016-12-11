import tornado.ioloop
import tornado.web
import tornado.websocket
import src.cifarUtils as cifarUtils
import numpy
from PIL import Image, ImageFilter
from scipy.misc import toimage
import io
from src.convNNRunner import ConvNNRunner

from tornado.options import define, options, parse_command_line

define("port", default=8888, help="run on the given port", type=int)

clients = dict()

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("./web/conv.html")

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

app = tornado.web.Application([
    (r'/', IndexHandler),
    (r'/ws', WebSocketHandler),
])

if __name__ == '__main__':
    parse_command_line()
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

