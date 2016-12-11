import tornado.ioloop
import tornado.web
import tornado.websocket


from tornado.options import define, options, parse_command_line

define("port", default=8888, help="run on the given port", type=int)

clients = dict()


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("./web/conv.html")

    def data_received(self, chunk):
        pass



class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def data_received(self, chunk):
        pass

    def open(self, *args):
        with open("ex.png", "rb") as imageFile:
            f = imageFile.read()
            self.write_message(f, True)


    def on_message(self, message):
        print "Client %s received a message : %s" % (self.id, message)

    def on_close(self):
        pass

app = tornado.web.Application([
    (r'/', IndexHandler),
    (r'/ws', WebSocketHandler),
])

if __name__ == '__main__':
    parse_command_line()
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

