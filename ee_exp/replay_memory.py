"""
Replay memories for storing training samples
"""


import sys
import numpy

class ReplayMemory(object):
    """
    Replay memories to store training sample with a time window
    """
    def __init__(self, memory_size, shape):
        self._memory_size = memory_size
        self._value_number = len(shape)
        self._memory_pool = []
        self._shape = []
        for i in range(self._value_number):
            self._shape.append((memory_size, shape[i]))
            self._memory_pool.append(numpy.zeros(
                shape = self._shape[-1], dtype = 'float32'))
        self._end = 0
        self._first_round = True

    def push(self, *values):
        """
        push a sample, values must match that of args in initialization
        """
        assert len(values) == self._value_number, \
            "value length do not match %d" % self._value_number
        for i in range(len(values)):
            if(numpy.shape(values[i]) != self._shape[i][1:]):
                raise ValueError(("The %dth argument do not match the shape of %s,"
                    + "received shape = %s")%
                    (i, self._shape[i][1:], numpy.shape(values[i])))
            self._memory_pool[i][self._end] = values[i]
        self._end += 1
        if(self._end >= self._memory_size):
            self._end -= self._memory_size
            self._first_round = False

    def draw(self, batch_size):
        """
        draw a batch samples of batch_size
        """
        if(self._first_round and self._end < batch_size):
            return None
        if(self._first_round):
            find_array = numpy.arange(self._end)
        else:
            find_array = numpy.arange(self._memory_size)
        numpy.random.shuffle(find_array)
        sel_array = find_array[:batch_size]
        ret_tuple = ()
        for i in range(self._value_number):
            ret_tuple += (self._memory_pool[i][sel_array], )
        return ret_tuple

    def draw_by_all(self, batch_size):
        """
        draw a batch samples of batch_size
        """
        if self._first_round and batch_size > self._end:
          batch_size = self._end

        if(self._first_round):
            find_array = numpy.arange(self._end)
        else:
            find_array = numpy.arange(self._memory_size)
        numpy.random.shuffle(find_array)
        sel_array = find_array[:batch_size]
        ret_tuple = ()
        for i in range(self._value_number):
            ret_tuple += (self._memory_pool[i][sel_array], )
        return ret_tuple

    def ready(self, check_size):
        """
        check if there are enough examples for drawing batch_size
        """
        return (self._end > check_size or self._first_round == False)

    def get_size(self):
      if self._first_round:
        return self._end
      else:
        return self._memory_size
