"""Keep local 20-update saves; bound expensive server publications separately."""


class CheckpointCadence:
    """For the pinned pilot: periodic saves are every20, final save repeats or is off-grid."""

    def __init__(self, writer, initial_step, every):
        if type(every) is not int or every < 20 or every % 20:
            raise ValueError('Durable cadence must be a positive multiple of20')
        self.writer = writer
        self.last_step = initial_step
        self.every = every

    @property
    def last_server_receipt(self):
        return self.writer.last_server_receipt

    def __call__(self, path, model, optimizer, step, order, masks, identity, history):
        if step < self.last_step:
            raise ValueError('Checkpoint steps must not go backwards')
        final_save = step == self.last_step or step % 20 != 0
        if final_save or step % self.every == 0:
            self.writer(path, model, optimizer, step, order, masks, identity, history)
        else:
            self.writer.store.check_time()
            self.writer.original_save(path, model, optimizer, step, order, masks, identity, history)
        self.last_step = step
