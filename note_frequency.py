class Const(object):
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__: # 判断是否已经被赋值，如果是则报错
            raise self.ConstError("Can't change const.%s" % name)
        # if not name.isupper(): # 判断所赋值是否是全部大写，用来做第一次赋值的格式判断，也可以根据需要改成其他判断条件
        #     raise self.ConstCaseError('const name "%s" is not all supercase' % name)

        self.__dict__[name] = value


# do re mi fa so la xi
const = Const()
const.do = 262
const.do_up = 277
const.re = 294
const.re_up = 311
const.mi = 330
const.fa = 349
const.fa_up =  370
const.so = 392
const.so_up = 415
const.la = 440
const.la_up = 466
const.xi = 494