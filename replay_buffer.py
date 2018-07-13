import random
from collections import deque
import numpy as np

class ReplayBuffer(object):


    def __init__(self, capacity):
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity


    def add(self, frame):
        self.buffer.append(frame)

    def get(self, batchsz):

        if len(self.buffer) < batchsz:
            batchsz = len(self.buffer)

        choices = random.sample(self.buffer, batchsz)

        sb_1 = []
        ab_1 = []
        rb_1 = []
        db_1 = []
        sb_2 = []

        while batchsz:
            sb, ab, rb, db, sb_ = choices.pop()

            sb_1.append(sb)
            ab_1.append(ab)
            rb_1.append(rb)
            db_1.append(db)
            sb_2.append(sb_)

            batchsz   -= 1

        """ numpyfy """
        sb_1 = np.array(sb_1)
        ab_1 = np.array(ab_1)
        rb_1 = np.array(rb_1)
        db_1 = np.array(db_1)
        sb_2 = np.array(sb_2)

        return sb_1, ab_1, rb_1, db_1, sb_2




