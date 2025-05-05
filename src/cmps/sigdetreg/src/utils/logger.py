from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch

USE_TENSORBOARD = True
try:
    import tensorboardX

    print('Using tensorboardX')
except:
    USE_TENSORBOARD = False


class Logger(object):
    def __init__(self, arg):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(arg.output_dir):
            os.makedirs(arg.output_dir)

        time_str = time.strftime('%Y-%m-%d-%H-%M')

        args = dict((name, getattr(arg, name)) for name in dir(arg)
                    if not name.startswith('_'))
        file_name = os.path.join(arg.output_dir, 'arg.txt')
        with open(file_name, 'wt') as arg_file:
            arg_file.write('==> torch version: {}\n'.format(torch.__version__))
            arg_file.write('==> cudnn version: {}\n'.format(
                torch.backends.cudnn.version()))
            arg_file.write('==> Cmd:\n')
            arg_file.write(str(sys.argv))
            arg_file.write('\n==> arg:\n')
            for k, v in sorted(args.items()):
                arg_file.write('  %s: %s\n' % (str(k), str(v)))

        log_dir = arg.output_dir + '/logs_{}'.format(time_str)
        if USE_TENSORBOARD:
            self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        else:
            if not os.path.exists(os.path.dirname(log_dir)):
                os.mkdir(os.path.dirname(log_dir))
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
        self.log = open(log_dir + '/log.txt', 'w')
        try:
            os.system('cp {}/arg.txt {}/'.format(arg.output_dir, log_dir))
        except:
            pass
        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def write_epoch(self, stats, epoch, phase='train'):

        txt = '{} [{}]'.format(phase, epoch)
        for k, v in stats.items():
            txt += ' {}: {}'.format(k, v)

        self.write(txt + '\n')



    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)
