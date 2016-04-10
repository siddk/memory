"""
analytics.py

File containing class definitions for analytic helpers, like the training Progress bar.
"""
import sys
import time


class Progress(object):
    """
    Progress bar
    """
    def __init__(self, iterable, bar_length=50):
        self.iterable = iterable
        self.bar_length = bar_length
        self.total_length = len(iterable)
        self.start_time = time.time()
        self.count = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.count += 1
            percent = self.count / self.total_length
            print_length = int(percent * self.bar_length)
            progress = "=" * print_length + " " * (self.bar_length - print_length)
            elapsed_time = time.time() - self.start_time
            print_msg = "\r|%s| %.0f%% %.1fs" % (progress, percent * 100, elapsed_time)
            sys.stdout.write(print_msg)
            if self.count == self.total_length:
                sys.stdout.write("\r" + " " * len(print_msg) + "\r")
            sys.stdout.flush()